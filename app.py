#!/usr/bin/env python3
"""
HashNet Studio — локальная оболочка для hashnet_proto: данные, обучение, слияние, чат, логи.
Только стандартная библиотека Python. Запуск:  python3 app.py   (откроется браузер)

Сервер слушает ТОЛЬКО 127.0.0.1, все запросы к /api требуют одноразовый токен из адреса,
запускаются лишь заранее известные действия (никакого произвольного shell).
"""
import argparse, codecs, json, os, re, secrets, shlex, shutil, signal, struct, subprocess, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
CONF_PATH = os.path.join(ROOT, "studio_config.json")
UI_DIR = os.path.join(ROOT, "ui")
TOOLS = os.path.join(ROOT, "tools")
DEFAULT_CONF = {
    "project_dir": os.path.join(ROOT, "rust"),
    "workspace": os.path.join(ROOT, "workspace"),
    "cargo": "cargo",
    "python": sys.executable or "python3",
}
CONF = dict(DEFAULT_CONF)
TOKEN = secrets.token_urlsafe(24)
PORT = 8765


def load_conf():
    try:
        with open(CONF_PATH, encoding="utf-8") as f:
            CONF.update({k: v for k, v in json.load(f).items() if k in DEFAULT_CONF and isinstance(v, str) and v})
    except Exception:
        pass


def save_conf():
    with open(CONF_PATH, "w", encoding="utf-8") as f:
        json.dump(CONF, f, ensure_ascii=False, indent=2)


def ws(*parts):
    return os.path.join(CONF["workspace"], *parts)


# ───────────────────────────── задачи ─────────────────────────────
JOBS = {}
JOB_SEQ = [0]
JOBS_LOCK = threading.Lock()


def init_job_seq():
    """номера задач продолжаются после перезапуска (по именам файлов в logs/)"""
    mx = 0
    try:
        for fn in os.listdir(ws("logs")):
            m = re.match(r"^\d{8}_\d{6}_(\d+)_", fn)
            if m:
                mx = max(mx, int(m.group(1)))
    except Exception:
        pass
    JOB_SEQ[0] = mx


class Job:
    def __init__(self, kind, title, steps, gpu=False):
        with JOBS_LOCK:
            JOB_SEQ[0] += 1
            self.id = JOB_SEQ[0]
        self.kind, self.title, self.steps, self.gpu = kind, title, steps, gpu
        self.status = "queued"
        self.lines = []
        self.live = ""
        self.t0 = time.time()
        self.t1 = None
        self.rc = None
        self.proc = None
        self.stop_flag = False
        self.lock = threading.Lock()
        os.makedirs(ws("logs"), exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.t0))
        self.log_path = ws("logs", f"{stamp}_{self.id:03d}_{kind}.log")
        self.cmd_text = "\n".join(self._cmd_str(s) for s in steps)
        self._logf = None

    @staticmethod
    def _cmd_str(step):
        if step[0] == "call":
            return "(внутренний шаг) " + step[2]
        return " ".join(shlex.quote(a) for a in step[0])

    def _emit(self, line):
        with self.lock:
            self.lines.append(line)
            self.live = ""
        if self._logf:
            self._logf.write(line + "\n")
            self._logf.flush()

    def _feed(self, text, state):
        buf = state["buf"] + text
        while True:
            m = re.search(r"[\r\n]", buf)
            if not m:
                break
            i = m.start()
            seg, sep = buf[:i], buf[i]
            buf = buf[i + 1:]
            if sep == "\r":
                if buf.startswith("\n"):
                    buf = buf[1:]
                    self._emit(seg)
                elif buf == "":
                    state["buf"] = seg + "\r"      # возможный \r\n на границе чтения
                    return
                else:
                    with self.lock:
                        self.live = seg
            else:
                self._emit(seg)
        state["buf"] = buf
        with self.lock:
            if buf:
                self.live = buf

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        self.status = "running"
        try:
            self._logf = open(self.log_path, "w", encoding="utf-8")
        except Exception:
            self._logf = None
        rc = 0
        for step in self.steps:
            if self.stop_flag:
                break
            if step[0] == "call":
                try:
                    step[1](self._emit)
                    rc = 0
                except Exception as e:
                    self._emit(f"ОШИБКА: {e}")
                    rc = 1
                if rc:
                    break
                continue
            argv, cwd = step
            self._emit("$ " + " ".join(shlex.quote(a) for a in argv))
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            try:
                self.proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True, env=env, bufsize=0)
            except Exception as e:
                self._emit(f"НЕ УДАЛОСЬ ЗАПУСТИТЬ: {e}")
                rc = 127
                break
            dec = codecs.getincrementaldecoder("utf-8")("replace")
            state = {"buf": ""}
            fd = self.proc.stdout
            while True:
                chunk = fd.read1(65536) if hasattr(fd, "read1") else fd.read(65536)
                if not chunk:
                    break
                self._feed(dec.decode(chunk), state)
            tail = state["buf"].rstrip("\r")
            if tail:
                self._emit(tail)
            rc = self.proc.wait()
            if rc != 0:
                break
        self.rc = rc
        self.t1 = time.time()
        if self.stop_flag and rc in (0, 130, -2, -15, -9, 2):
            self.status = "stopped"
        else:
            self.status = "done" if rc == 0 else "failed"
        self._emit(f"── завершено: {self.status} (код {rc}), {self.t1 - self.t0:.0f} с ──")
        if self._logf:
            self._logf.close()

    def stop(self, force=False):
        self.stop_flag = True
        p = self.proc
        if p and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL if force else signal.SIGINT)
            except Exception:
                pass

    def summary(self):
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status, "gpu": self.gpu,
                "t0": self.t0, "t1": self.t1, "rc": self.rc, "log_path": self.log_path}


def running_jobs(kind=None):
    return [j for j in list(JOBS.values()) if j.status in ("queued", "running") and (kind is None or j.kind == kind)]


# ───────────────────────────── Rust-бинарник ─────────────────────────────
def project_ok():
    return os.path.isfile(os.path.join(CONF["project_dir"], "Cargo.toml"))


def bin_path():
    return os.path.join(CONF["project_dir"], "target", "release", "hashnet_proto")


def need_build():
    b = bin_path()
    if not os.path.isfile(b):
        return True
    bt = os.path.getmtime(b)
    proj = CONF["project_dir"]
    for root, _, files in os.walk(os.path.join(proj, "src")):
        for fn in files:
            if os.path.getmtime(os.path.join(root, fn)) > bt:
                return True
    return os.path.getmtime(os.path.join(proj, "Cargo.toml")) > bt


def rust_steps(args):
    if not project_ok():
        raise ValueError(f"в папке проекта нет Cargo.toml: {CONF['project_dir']} (Настройки → папка проекта)")
    steps = []
    if need_build():
        steps.append(([CONF["cargo"], "build", "--release", "--bin", "hashnet_proto"], CONF["project_dir"]))
    steps.append(([bin_path()] + [str(a) for a in args], CONF["project_dir"]))
    return steps


def py_tool(name, args):
    p = os.path.join(TOOLS, name)
    if not os.path.isfile(p):
        raise ValueError(f"нет инструмента {name}")
    return [([CONF["python"], "-u", p] + [str(a) for a in args], ROOT)]


# ───────────────────────────── чекпойнты ─────────────────────────────
def read_ckpt_meta(path):
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"HN4DCKP1":
                return None
            u32 = lambda: struct.unpack("<I", f.read(4))[0]
            u64 = lambda: struct.unpack("<Q", f.read(8))[0]
            def rstr():
                n = u64()
                if n > 1 << 20:
                    raise ValueError
                return f.read(n).decode("utf-8", "replace")
            u32()
            d, L, h, seq = u64(), u64(), u64(), u64()
            step = u64(); phase = u32(); rng = u64(); tokens = u64()
            done = f.read(1)[0]
            pstart = u64(); psteps = u64(); lr = struct.unpack("<f", f.read(4))[0]
            files = []
            for _ in range(u64()):
                files.append({"path": rstr(), "len": u64()})
            segs = []
            for _ in range(u64()):
                p = rstr(); s0 = u64(); e0 = u64(); vs = u64(); ve = u64(); ph = u32()
                w = struct.unpack("<d", f.read(8))[0]
                segs.append({"path": p, "start": s0, "end": e0, "phase": ph, "weight": w})
        params = 12 * L * d * d + 4 * L * d + 256 * d + seq * d + d * 256 + 2 * d
        return {"d_model": d, "layers": L, "heads": h, "seq": seq, "step": step, "phase": phase, "tokens": tokens,
                "phase_done": bool(done), "phase_start": pstart, "phase_steps": psteps, "phase_lr": lr,
                "params": params, "files": files, "segs": segs}
    except Exception:
        return None


def scan_ckpts():
    roots = {os.path.abspath(ws("checkpoints")), os.path.abspath(CONF["project_dir"]), os.path.abspath(os.path.dirname(CONF["project_dir"]))}
    seen, out = set(), []
    for r in roots:
        if not os.path.isdir(r):
            continue
        base_depth = r.rstrip("/").count("/")
        for root, dirs, files in os.walk(r):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("target", "node_modules", "__pycache__", "logs", "src")]
            if root.count("/") - base_depth >= 2:
                dirs[:] = []
            cks = sorted(f for f in files if f.startswith("ckpt_") and f.endswith(".bin"))
            if cks and root not in seen:
                seen.add(root)
                latest = os.path.join(root, cks[-1])
                meta = read_ckpt_meta(latest)
                if meta:
                    st = os.stat(latest)
                    out.append({"dir": root, "name": os.path.basename(root), "file": cks[-1], "size": st.st_size,
                                "mtime": st.st_mtime, "count": len(cks), "meta": meta})
    out.sort(key=lambda c: -c["mtime"])
    return out


# ───────────────────────────── система ─────────────────────────────
_gpu_cache = {"t": 0, "v": None}


def gpu_info():
    if time.time() - _gpu_cache["t"] < 2:
        return _gpu_cache["v"]
    v = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
        if r.returncode == 0 and r.stdout.strip():
            p = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            f = lambda s: float(s) if re.match(r"^[\d.]+$", s or "") else None
            v = {"name": p[0], "used": f(p[1]), "total": f(p[2]), "temp": f(p[3]), "util": f(p[4]), "power": f(p[5]) if len(p) > 5 else None}
    except Exception:
        v = None
    _gpu_cache.update(t=time.time(), v=v)
    return v


def state():
    try:
        du = shutil.disk_usage(CONF["workspace"] if os.path.isdir(CONF["workspace"]) else ROOT)
        disk = {"free": du.free, "total": du.total}
    except Exception:
        disk = None
    return {
        "conf": CONF, "gpu": gpu_info(), "disk": disk,
        "project_ok": project_ok(), "binary": os.path.isfile(bin_path()) and project_ok(), "need_build": project_ok() and need_build(),
        "cargo": shutil.which(CONF["cargo"]) is not None, "root": ROOT,
        "jobs": [j.summary() for j in sorted(JOBS.values(), key=lambda j: -j.id)[:40]],
        "ckpts": scan_ckpts(),
    }


# ───────────────────────────── действия ─────────────────────────────
def num(p, key, typ=float, default=None, lo=None, hi=None, required=False):
    v = p.get(key)
    if v is None or v == "":
        if required:
            raise ValueError(f"поле «{key}» не заполнено")
        return default
    try:
        x = typ(v)
    except Exception:
        raise ValueError(f"поле «{key}»: не число ({v!r})")
    if lo is not None and x < lo or hi is not None and x > hi:
        raise ValueError(f"поле «{key}»: значение вне диапазона {lo}…{hi}")
    return x


def path_arg(p, key, must_exist=False, required=True):
    v = p.get(key)
    if not v:
        if required:
            raise ValueError(f"не указан путь: {key}")
        return None
    v = os.path.expanduser(str(v))
    if "\x00" in v:
        raise ValueError("недопустимый путь")
    v = os.path.abspath(v)
    if must_exist and not os.path.exists(v):
        raise ValueError(f"не найдено: {v}")
    return v


def action_train(p):
    ckdir = path_arg(p, "ckpt_dir")
    resume = bool(p.get("resume"))
    a = ["train", "--ckpt-dir", ckdir, "--steps", num(p, "steps", int, required=True, lo=1)]
    if resume:
        a.append("--resume")
        if p.get("new_phase"):
            a.append("--new-phase")
    else:
        for k, flag in (("d_model", "--d-model"), ("layers", "--layers"), ("heads", "--heads"), ("seq", "--seq")):
            a += [flag, num(p, k, int, required=True, lo=1)]
        if int(p["d_model"]) % int(p["heads"]):
            raise ValueError("d_model должно делиться на число голов")
    for k, flag, typ in (("batch", "--batch", int), ("accum", "--accum", int), ("lr", "--lr", float), ("warmup", "--warmup", int),
                         ("replay_frac", "--replay-frac", float), ("ckpt_every", "--ckpt-every", int), ("eval_every", "--eval-every", int),
                         ("keep", "--keep", int), ("seed", "--seed", int), ("log_every", "--log-every", int),
                         ("val_windows", "--val-windows", int)):
        v = num(p, k, typ)
        if v is not None:
            a += [flag, v]
    if p.get("no_quant"):
        a.append("--no-quant")
    data = p.get("data") or []
    if not resume and not data:
        raise ValueError("для новой модели нужен хотя бы один файл с данными")
    for d in data:
        path = path_arg({"x": d.get("path")}, "x", must_exist=True)
        w = d.get("weight")
        a += ["--data", f"{path}@{float(w)}" if w not in (None, "") else path]
    return Job("train", f"Обучение: {os.path.basename(ckdir)}", rust_steps(a), gpu=True)


def action_gen(p):
    ck = path_arg(p, "ckpt", must_exist=True)
    prompt = str(p.get("prompt", ""))
    if prompt.startswith("--"):
        prompt = " " + prompt
    a = ["gen", "--ckpt", ck, "--len", num(p, "len", int, 200, 1, 4000), "--temp", num(p, "temp", float, 0.8, 0, 5),
         "--top-k", num(p, "top_k", int, 40, 0, 256)]
    nr = num(p, "no_repeat", int, 0, 0, 512)
    if nr:
        a += ["--no-repeat", nr]
    ln = num(p, "loop_n", int, 0, 0, 512)
    if ln:
        a += ["--loop-n", ln]
    if p.get("verify"):
        a.append("--verify")
    if prompt:
        a += ["--prompt", prompt]
    g = gpu_info()
    if g and g["total"] and g["total"] - (g["used"] or 0) < 1200:
        raise RuntimeError(f"мало свободной видеопамяти ({g['total'] - g['used']:.0f} МБ): остановите обучение или подождите")
    return Job("gen", "Генерация", rust_steps(a), gpu=True)


def action_merge(p):
    method = p.get("method")
    if method not in ("avg", "slerp", "task", "ties", "dare", "stack", "grow"):
        raise ValueError("неизвестный метод")
    models = p.get("models") or []
    if not models:
        raise ValueError("не выбрано ни одной модели")
    out = path_arg(p, "out")
    a = ["merge", "--method", method, "--out", out]
    if method == "grow" and len(models) != 1:
        raise ValueError("grow работает с одной моделью")
    for m in models:
        a += ["--model", path_arg({"x": m}, "x", must_exist=True)]
    if method == "grow":
        a += ["--add-layers", num(p, "add_layers", int, required=True, lo=1, hi=256)]
        for k, flag, allowed in (("place", "--place", ("end", "spread")), ("init", "--init", ("ident", "copy"))):
            v = str(p.get(k) or "")
            if v:
                if v not in allowed:
                    raise ValueError(f"поле «{k}»: допустимо {'/'.join(allowed)}")
                a += [flag, v]
        if str(p.get("init") or "ident") == "ident":
            eps = num(p, "eps", float, None, 0, 10)
            if eps is not None:
                a += ["--eps", eps]
    if method in ("task", "ties", "dare"):
        a += ["--base", path_arg(p, "base", must_exist=True)]
    for k, flag, typ in (("density", "--density", float), ("lambda", "--lambda", float), ("drop", "--drop", float), ("t", "--t", float), ("seed", "--seed", int)):
        v = num(p, k, typ)
        if v is not None:
            a += [flag, v]
    for k, flag in (("alpha", "--alpha"), ("take", "--take"), ("globals", "--globals")):
        v = str(p.get(k) or "").strip()
        if v:
            if not re.match(r"^[0-9a-z.,\- ]+$", v):
                raise ValueError(f"поле «{k}»: недопустимые символы")
            a += [flag, v.replace(" ", "")]
    return Job("merge", f"Слияние ({method})", rust_steps(a), gpu=False)


def action_eval(p):
    src = path_arg(p, "ckpt", must_exist=True)
    meta_file = src
    if os.path.isdir(src):
        cks = sorted(f for f in os.listdir(src) if f.startswith("ckpt_") and f.endswith(".bin"))
        if not cks:
            raise ValueError("в папке нет чекпойнтов")
        meta_file = os.path.join(src, cks[-1])
    tmp = ws("tmp_eval", f"e{int(time.time())}")

    def prep(emit):
        os.makedirs(tmp, exist_ok=True)
        emit(f"копирую чекпойнт во временную папку {tmp} (оригинал не меняется)")
        shutil.copy2(meta_file, os.path.join(tmp, os.path.basename(meta_file)))

    def cleanup(emit):
        shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)
        emit("временная папка удалена")

    args = ["train", "--resume", "--ckpt-dir", tmp, "--steps", 1, "--lr", "1e-12", "--warmup", 1, "--batch", 1, "--accum", 1, "--ckpt-every", 0,
            "--val-windows", num(p, "val_windows", int, 64, 1, 100000)]
    steps = [("call", prep, "копирование чекпойнта")] + rust_steps(args) + [("call", cleanup, "уборка")]
    return Job("eval", "Проверка val", steps, gpu=True)


def action_info(p):
    return Job("info", "Сведения о чекпойнте", rust_steps(["info", "--ckpt", path_arg(p, "ckpt", must_exist=True)]), gpu=False)


def action_build(p):
    if not project_ok():
        raise ValueError(f"в папке проекта нет Cargo.toml: {CONF['project_dir']}")
    return Job("build", "Сборка программы", [([CONF["cargo"], "build", "--release"], CONF["project_dir"])], gpu=False)


def action_split(p):
    a = [path_arg(p, "src", must_exist=True), "--out", path_arg(p, "out")]
    if p.get("dry_run"):
        a.append("--dry-run")
    if p.get("keep_generated"):
        a.append("--keep-generated")
    lim = num(p, "limit_gb", float, 0, 0)
    if lim:
        a += ["--limit-gb", lim]
    return Job("split", "Нарезка файла", py_tool("split_corpus.py", a))


def action_sample(p):
    a = [path_arg(p, "src", must_exist=True), path_arg(p, "dst"), "--gb", num(p, "gb", float, required=True, lo=0.001),
         "--chunk-mb", num(p, "chunk_mb", float, 1.0, 0.01, 1000)]
    return Job("sample", "Выборка N ГБ", py_tool("sample_corpus.py", a))


def action_mix(p):
    files = p.get("files") or []
    if len(files) < 1:
        raise ValueError("добавьте файлы")
    a = ["--out", path_arg(p, "out")]
    tgt = num(p, "target_gb", float, 0, 0)
    if tgt:
        a += ["--target-gb", tgt]
    for f in files:
        path = path_arg({"x": f.get("path")}, "x", must_exist=True)
        w = f.get("weight")
        a.append(f"{path}@{float(w)}" if w not in (None, "") else path)
    return Job("mix", "Смешивание файлов", py_tool("mix_files.py", a))


def action_starcoder(p):
    a = [path_arg(p, "src", must_exist=True), "--out", path_arg(p, "out")]
    only = str(p.get("only") or "").strip()
    if only:
        if not re.match(r"^[\w,\-+# ]+$", only):
            raise ValueError("поле «языки»: недопустимые символы")
        a += ["--only", only.replace(" ", "")]
    g = num(p, "max_gb", float, 0, 0)
    if g:
        a += ["--max-gb-per-lang", g]
    return Job("starcoder", "StarCoder → текст", py_tool("starcoder_to_txt.py", a))


def action_shards(p):
    n = num(p, "n", int, required=True, lo=2, hi=64)
    a = [path_arg(p, "src", must_exist=True), "--out", path_arg(p, "out"), "--n", n]
    g = num(p, "max_gb_each", float, 0, 0)
    if g:
        a += ["--max-gb-each", g]
    mode = str(p.get("mode") or "chunks")
    if mode not in ("chunks", "contiguous"):
        raise ValueError("режим: chunks или contiguous")
    a += ["--mode", mode]
    return Job("shards", "Нарезка на непересекающиеся куски", py_tool("make_shards.py", a))


def _latest_ckpt_file(d):
    cks = sorted(f for f in os.listdir(d) if f.startswith("ckpt_") and f.endswith(".bin"))
    if not cks:
        raise ValueError(f"в папке нет чекпойнтов: {d}")
    return os.path.join(d, cks[-1])


def action_branches(p):
    """Пакет: ветки от общей базы на разных кусках данных → слияние → проверка val у всех."""
    base = path_arg(p, "base", must_exist=True)
    base_file = _latest_ckpt_file(base)
    bmeta = read_ckpt_meta(base_file)
    if not bmeta or not bmeta["files"]:
        raise ValueError("не удалось прочитать чекпойнт базы")
    key = os.path.basename(bmeta["files"][0]["path"])
    shards = [path_arg({"x": (s.get("path") if isinstance(s, dict) else s)}, "x", must_exist=True) for s in (p.get("shards") or [])]
    if len(shards) < 2:
        raise ValueError("нужно минимум 2 куска данных")
    out = path_arg(p, "out")
    if os.path.isdir(out) and os.listdir(out):
        raise ValueError(f"папка результата не пуста: {out} (укажите новую)")
    steps = num(p, "steps", int, 4000, 1)
    lr = num(p, "lr", float, 3e-4, 1e-9, 1)
    warm = num(p, "warmup", int, 100, 0)
    batch = num(p, "batch", int, 8, 1)
    merges = [m for m in ("avg", "ties") if p.get("m_" + m)]
    do_eval = bool(p.get("eval", True))
    if not project_ok():
        raise ValueError(f"в папке проекта нет Cargo.toml: {CONF['project_dir']}")
    exe = bin_path()
    cwd = CONF["project_dir"]
    holder = {}
    steps_l = []
    if need_build():
        steps_l.append(([CONF["cargo"], "build", "--release", "--bin", "hashnet_proto"], cwd))
    bdirs = [os.path.join(out, f"branch_{i + 1}") for i in range(len(shards))]

    def prep(emit):
        for bd in bdirs:
            os.makedirs(bd, exist_ok=True)
            shutil.copy2(base_file, os.path.join(bd, os.path.basename(base_file)))
        emit(f"скопирована база ({os.path.basename(base_file)}) в {len(bdirs)} папок веток")
    steps_l.append(("call", prep, "подготовка веток"))
    for i, (bd, sh) in enumerate(zip(bdirs, shards)):
        args = ["train", "--resume", "--ckpt-dir", bd, "--data", sh, "--steps", steps, "--lr", lr, "--warmup", warm,
                "--batch", batch, "--replay-frac", 0, "--ckpt-every", 1000, "--eval-every", 0]
        steps_l.append(([exe] + [str(x) for x in args], cwd))
    merged = []
    for m in merges:
        md = os.path.join(out, "merged_" + m)
        args = ["merge", "--method", m, "--out", md] + sum([["--model", bd] for bd in bdirs], [])
        if m == "ties":
            args += ["--base", base, "--density", 0.2, "--lambda", 1.0]
        steps_l.append(([exe] + [str(x) for x in args], cwd))
        merged.append(("merged_" + m, md))
    if do_eval:
        targets = [("base", base)] + [(f"branch_{i + 1}", bd) for i, bd in enumerate(bdirs)] + merged
        for name, d in targets:
            tmp = ws("tmp_eval_exp", name)

            def mk(name=name, d=d, tmp=tmp):
                def pre(emit):
                    shutil.rmtree(tmp, ignore_errors=True)
                    os.makedirs(tmp)
                    shutil.copy2(_latest_ckpt_file(d), os.path.join(tmp, os.path.basename(_latest_ckpt_file(d))))
                    emit(f"### ПРОВЕРКА {name}")
                def post(emit):
                    shutil.rmtree(tmp, ignore_errors=True)
                return pre, post
            pre, post = mk()
            steps_l.append(("call", pre, "проверка " + name))
            steps_l.append(([exe] + [str(x) for x in ["train", "--resume", "--ckpt-dir", tmp, "--steps", 1, "--lr", "1e-12", "--warmup", 1,
                                                       "--batch", 1, "--accum", 1, "--ckpt-every", 0, "--eval-every", 0]], cwd))
            steps_l.append(("call", post, "уборка"))

        def summary(emit):
            lines = list(holder["job"].lines)
            res, cur, wait = [], None, False
            for ln in lines:
                m = re.match(r"^### ПРОВЕРКА (\S+)", ln)
                if m:
                    cur, wait = m.group(1), False
                    continue
                if cur and "валидация ДО фазы" in ln:
                    wait = True
                    continue
                if cur and wait and key in ln:
                    v = re.search(r"=\s*([\d.]+)\s*бит/байт", ln)
                    if v:
                        res.append((cur, float(v.group(1))))
                        cur, wait = None, False
            emit("══ ИТОГ: val на данных базы (" + key + "), бит/байт — меньше лучше ══")
            b = dict(res).get("base")
            for name, v in res:
                emit(f"  {name:<16} {v:.3f}" + ("" if b is None or name == "base" else f"   ({v - b:+.3f} к базе)"))
            br = [v for n, v in res if n.startswith("branch_")]
            if br:
                emit(f"  лучшая ветка: {min(br):.3f}, средняя: {sum(br) / len(br):.3f}")
                for n, v in res:
                    if n.startswith("merged_"):
                        emit(f"  {n}: {'ЛУЧШЕ' if v < min(br) else 'хуже'} лучшей ветки на {abs(v - min(br)):.3f}")
        steps_l.append(("call", summary, "итоговая таблица"))
    job = Job("branches", f"Ветки ({len(shards)}) → слияние", steps_l, gpu=True)
    holder["job"] = job
    return job


def action_syntax(p):
    ck = path_arg(p, "ckpt", must_exist=True)
    if not os.path.isfile(bin_path()):
        raise ValueError("программа не собрана: нажмите «Собрать программу» на Панели")
    if not shutil.which("rustfmt"):
        raise ValueError("не найден rustfmt (rustup component add rustfmt)")
    g = gpu_info()
    if g and g["total"] and g["total"] - (g["used"] or 0) < 1200:
        raise RuntimeError(f"мало свободной видеопамяти ({g['total'] - g['used']:.0f} МБ): остановите обучение или подождите")
    a = ["--ckpt", ck, "--n", num(p, "n", int, 40, 1, 400), "--len", num(p, "len", int, 400, 50, 4000), "--bin", bin_path()]
    ln = num(p, "loop_n", int, 0, 0, 512)
    if ln:
        a += ["--loop-n", ln]
    tp = num(p, "temp", float, None, 0.05, 3)
    if tp is not None:
        a += ["--temp", tp]
    return Job("syntax", "Метрика: синтаксис Rust", py_tool("eval_syntax.py", a), gpu=True)


ACTIONS = {"train": action_train, "gen": action_gen, "merge": action_merge, "eval": action_eval, "info": action_info,
           "build": action_build, "split": action_split, "sample": action_sample, "mix": action_mix, "starcoder": action_starcoder, "shards": action_shards, "branches": action_branches, "syntax": action_syntax}


def run_action(name, params):
    fn = ACTIONS.get(name)
    if not fn:
        raise ValueError("неизвестное действие")
    if name in ("train", "branches") and (running_jobs("train") or running_jobs("branches")):
        j0 = (running_jobs("train") + running_jobs("branches"))[0]
        raise RuntimeError(f"уже идёт обучение (задача №{j0.id}); остановите его кнопкой «Стоп» (чекпойнт сохранится)")
    if name in ("eval", "gen", "syntax") and running_jobs(name):
        raise RuntimeError("такая задача уже выполняется")
    if name == "build" and running_jobs("build"):
        raise RuntimeError("сборка уже идёт")
    job = fn(params)
    JOBS[job.id] = job
    job.start()
    return job


# ───────────────────────────── файловый браузер ─────────────────────────────
def list_dir(path, show_hidden=False):
    path = os.path.abspath(os.path.expanduser(path or ROOT))
    if not os.path.isdir(path):
        path = os.path.dirname(path) if os.path.isdir(os.path.dirname(path)) else ROOT
    out = []
    try:
        with os.scandir(path) as it:
            for e in it:
                if not show_hidden and e.name.startswith("."):
                    continue
                try:
                    st = e.stat()
                    out.append({"name": e.name, "dir": e.is_dir(), "size": 0 if e.is_dir() else st.st_size, "mtime": st.st_mtime})
                except OSError:
                    continue
                if len(out) >= 4000:
                    break
    except PermissionError:
        return {"path": path, "parent": os.path.dirname(path), "entries": [], "error": "нет доступа"}
    out.sort(key=lambda x: (not x["dir"], x["name"].lower()))
    mounts = []
    for base in ("/media", "/mnt", "/run/media"):
        if os.path.isdir(base):
            for root, dirs, _ in os.walk(base):
                depth = root[len(base):].count("/")
                if depth >= 2:
                    dirs[:] = []
                    continue
                for d in dirs:
                    mounts.append(os.path.join(root, d))
    return {"path": path, "parent": os.path.dirname(path), "entries": out, "home": os.path.expanduser("~"),
            "mounts": sorted(set(m for m in mounts if not os.path.basename(m).startswith(".")))[:12], "workspace": CONF["workspace"], "project": CONF["project_dir"]}


# ───────────────────────────── HTTP ─────────────────────────────
class H(BaseHTTPRequestHandler):
    server_version = "HashNetStudio/1.0"

    def log_message(self, fmt, *args):
        pass

    def _host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        return host in ("127.0.0.1", "localhost", "[::1]")

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        return secrets.compare_digest(self.headers.get("X-Token", ""), TOKEN)

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "bad host"})
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            with open(os.path.join(UI_DIR, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/readme":
            if not secrets.compare_digest(q.get("t", ""), TOKEN):
                return self._send(403, "нужен токен", "text/plain; charset=utf-8")
            with open(os.path.join(ROOT, "docs", "GUIDE.md"), "rb") as f:
                return self._send(200, f.read(), "text/plain; charset=utf-8")
        if not u.path.startswith("/api/") or not self._auth():
            return self._send(403 if u.path.startswith("/api/") else 404, {"error": "forbidden"})
        try:
            if u.path == "/api/state":
                return self._send(200, state())
            if u.path == "/api/fs":
                return self._send(200, list_dir(q.get("path", ""), q.get("hidden") == "1"))
            if u.path == "/api/job":
                j = JOBS.get(int(q.get("id", 0)))
                if not j:
                    return self._send(404, {"error": "нет такой задачи"})
                frm = max(0, int(q.get("from", 0)))
                with j.lock:
                    lines = j.lines[frm:frm + 4000]
                    live = j.live
                return self._send(200, dict(j.summary(), lines=lines, next=frm + len(lines), live=live, cmd=j.cmd_text, total=len(j.lines)))
            return self._send(404, {"error": "unknown"})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        if not self._host_ok() or not self._auth():
            return self._send(403, {"error": "forbidden"})
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._send(415, {"error": "ожидался JSON"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1 << 20:
            return self._send(413, {"error": "слишком большой запрос"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "некорректный JSON"})
        u = urlparse(self.path)
        try:
            if u.path == "/api/run":
                job = run_action(body.get("action"), body.get("params") or {})
                return self._send(200, {"id": job.id, "cmd": job.cmd_text})
            if u.path == "/api/stop":
                j = JOBS.get(int(body.get("id", 0)))
                if not j:
                    return self._send(404, {"error": "нет такой задачи"})
                j.stop(bool(body.get("force")))
                return self._send(200, {"ok": True})
            if u.path == "/api/conf":
                for k in DEFAULT_CONF:
                    v = str(body.get(k) or "").strip()
                    if v:
                        CONF[k] = os.path.expanduser(v) if k.endswith("dir") or k == "workspace" else v
                for sub in ("data", "checkpoints", "logs"):
                    os.makedirs(ws(sub), exist_ok=True)
                save_conf()
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "unknown"})
        except (ValueError, RuntimeError) as e:
            return self._send(409 if isinstance(e, RuntimeError) else 400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    global PORT
    ap = argparse.ArgumentParser(description="HashNet Studio")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    load_conf()
    for sub in ("data", "checkpoints", "logs"):
        os.makedirs(ws(sub), exist_ok=True)
    init_job_seq()
    srv = None
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H)
            PORT = port
            break
        except OSError:
            continue
    if not srv:
        sys.exit("не удалось занять порт")
    url = f"http://127.0.0.1:{PORT}/?t={TOKEN}"
    print("HashNet Studio запущена.")
    print("Открой в браузере (адрес одноразовый, содержит ключ доступа):")
    print("  " + url)
    print("Остановить: Ctrl+C (идущие задачи будут остановлены корректно, чекпойнт сохранится).")
    if not a.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    live = running_jobs()
    if live:
        print(f"останавливаю задачи: {len(live)} …")
        for j in live:
            j.stop()
        t0 = time.time()
        while running_jobs() and time.time() - t0 < 180:
            time.sleep(0.5)
    print("до свидания")


if __name__ == "__main__":
    main()
