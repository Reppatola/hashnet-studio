#!/usr/bin/env python3
"""
fetch_repos.py — скачать самые популярные Rust-репозитории с GitHub для датасета.

Только стандартная библиотека Python 3 + установленный git. Запуск:

    export GITHUB_TOKEN=ghp_...        # желательно: без токена поиск ограничен ~10 запросов/мин
    python3 fetch_repos.py --out "/media/tola/Новый том/rust_repos" --count 500

Что делает:
  1. Ищет репозитории language:Rust (без форков и архивных) по убыванию звёзд, диапазонами звёзд
     (у поиска GitHub потолок 1000 результатов на запрос, поэтому диапазоны).
  2. Оставляет только разрешённые лицензии (по умолчанию permissive: MIT, Apache-2.0, BSD, ISC, ...).
  3. Клонирует поверхностно (--depth 1), удаляет всё кроме .rs/.toml/.md и файлов лицензий и .git
     (чтобы не занимать диск картинками и бинарниками).
  4. Пишет repos.jsonl (звёзды, лицензия, коммит, дата) и skipped.txt (что пропущено и почему).
Возобновляемо: уже скачанные репозитории пропускаются при повторном запуске.
После — сканируй: cargo run --release --bin scan -- --paths "<папка out>"
"""
import argparse, json, os, shutil, subprocess, sys, threading, time
import urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = os.environ.get("GITHUB_API", "https://api.github.com")
PERMISSIVE = "MIT,Apache-2.0,BSD-2-Clause,BSD-3-Clause,ISC,Unlicense,0BSD,Zlib,CC0-1.0"
EDGES = [10**7, 30000, 15000, 10000, 7000, 5000, 3500, 2500, 1800, 1300, 1000, 750, 500, 350, 250, 180, 120, 80, 50, 30, 20, 10]
KEEP_EXT = {".rs", ".toml", ".md"}
KEEP_PREFIX = ("license", "licence", "copying", "notice", "unlicense")
lock = threading.Lock()


def log(*a):
    with lock:
        print(*a, flush=True)


def star_ranges(min_stars):
    """убывающие диапазоны (lo, hi), не ниже min_stars"""
    out = []
    for hi, lo in zip(EDGES, EDGES[1:]):
        if hi <= min_stars:
            break
        out.append((max(lo, min_stars), hi if not out else hi - 1))
    return out


BAD_TOKEN = False  # токен отвергнут (401) — дальше работаем без него


def api_get(url, token, tries=6):
    global BAD_TOKEN
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "hashnet-fetch-repos", "X-GitHub-Api-Version": "2022-11-28"}
    for attempt in range(tries):
        if token and not BAD_TOKEN:
            headers["Authorization"] = "Bearer " + token
        else:
            headers.pop("Authorization", None)
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                remaining = r.headers.get("X-RateLimit-Remaining")
                reset = r.headers.get("X-RateLimit-Reset")
                body = json.loads(r.read().decode("utf-8"))
                return body, remaining, reset
        except urllib.error.HTTPError as e:
            if e.code == 401 and token and not BAD_TOKEN:
                BAD_TOKEN = True
                log("⚠ GitHub отверг токен (HTTP 401): он неверный или просрочен. Продолжаю БЕЗ токена — это медленнее (лимит ~10 запросов/мин).")
                continue
            if e.code in (403, 429):
                ra = e.headers.get("Retry-After")
                rem = e.headers.get("X-RateLimit-Remaining")
                reset = e.headers.get("X-RateLimit-Reset")
                if ra:
                    wait = int(ra) + 1
                elif rem == "0" and reset:
                    wait = max(1, int(reset) - int(time.time())) + 1
                else:
                    wait = 60
                wait = min(wait, 900)
                log(f"  ⏳ лимит GitHub (HTTP {e.code}), жду {wait} с…")
                time.sleep(wait)
                continue
            if e.code in (500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("GitHub API не отвечает: " + url)


def find_candidates(args, token, allowed):
    """возвращает (кандидаты, пропущенные[(имя, причина)])"""
    cands, skipped, seen = [], [], set()
    pause_tok, pause_anon = 2.3, 6.5
    for lo, hi in star_ranges(args.min_stars):
        for page in range(1, 11):
            q = f"language:Rust fork:false archived:false stars:{lo}..{hi}"
            url = f"{API}/search/repositories?" + urllib.parse.urlencode({"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page})
            body, remaining, reset = api_get(url, token)
            items = body.get("items", [])
            for it in items:
                name = it["full_name"]
                if name in seen:
                    continue
                seen.add(name)
                lic = (it.get("license") or {}).get("spdx_id") or "NONE"
                size_mb = it.get("size", 0) / 1024
                if allowed is not None and lic not in allowed:
                    skipped.append((name, f"лицензия {lic}"))
                elif size_mb > args.max_size_mb:
                    skipped.append((name, f"размер истории {size_mb:.0f} МБ > {args.max_size_mb}"))
                else:
                    cands.append({"name": name, "url": it["clone_url"], "stars": it["stargazers_count"], "license": lic,
                                  "size_mb": round(size_mb, 1), "pushed_at": it.get("pushed_at"), "branch": it.get("default_branch")})
                if len(cands) >= args.count:
                    return cands, skipped
            log(f"  поиск: звёзды {lo}..{hi}, стр. {page}: +{len(items)}, подходит {len(cands)}/{args.count}")
            if len(items) < 100:
                break
            if remaining is not None and remaining.isdigit() and int(remaining) <= 1 and reset:
                time.sleep(max(1, int(reset) - int(time.time())) + 1)
            else:
                time.sleep(pause_anon if (not token or BAD_TOKEN) else pause_tok)
    return cands, skipped


def prune(dest):
    """оставить только .rs/.toml/.md и лицензии; удалить пустые каталоги"""
    for root, dirs, files in os.walk(dest, topdown=True):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in KEEP_EXT or f.lower().startswith(KEEP_PREFIX):
                continue
            try:
                os.remove(os.path.join(root, f))
            except OSError:
                pass
    for root, dirs, files in os.walk(dest, topdown=False):
        if root != dest and not os.listdir(root):
            try:
                os.rmdir(root)
            except OSError:
                pass


def clone_one(c, out, args):
    final = os.path.join(out, c["name"].replace("/", "__"))
    if os.path.isdir(final):
        return c, "exists", None
    free_gb = shutil.disk_usage(out).free / 2**30
    if free_gb < args.min_free_gb:
        return c, "nodisk", f"свободно {free_gb:.1f} ГБ < {args.min_free_gb}"
    tmp = final + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    cmd = ["git", "clone", "--quiet", "--depth", str(args.depth), "--single-branch", c["url"], tmp]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp, ignore_errors=True)
        return c, "fail", "таймаут клонирования"
    if r.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return c, "fail", r.stderr.strip()[-200:]
    sha = subprocess.run(["git", "-C", tmp, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not args.keep_git:
        shutil.rmtree(os.path.join(tmp, ".git"), ignore_errors=True)
    if not args.keep_all:
        prune(tmp)
    c = dict(c, commit=sha)
    os.rename(tmp, final)
    return c, "ok", None


def main():
    ap = argparse.ArgumentParser(description="Скачать популярные Rust-репозитории с GitHub")
    ap.add_argument("--out", required=True, help="куда складывать")
    ap.add_argument("--count", type=int, default=300, help="сколько репозиториев взять")
    ap.add_argument("--min-stars", type=int, default=50)
    ap.add_argument("--licenses", default=PERMISSIVE, help="через запятую SPDX-идентификаторы, или 'any'")
    ap.add_argument("--max-size-mb", type=float, default=3000, help="пропускать репозитории с историей больше (МБ по данным GitHub)")
    ap.add_argument("--depth", type=int, default=1, help="глубина клона (1 = только последний коммит)")
    ap.add_argument("--keep-git", action="store_true", help="оставить .git (нужно, если depth>1 и нужна история коммитов)")
    ap.add_argument("--keep-all", action="store_true", help="не удалять нерусты/картинки/бинарники")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--min-free-gb", type=float, default=10)
    ap.add_argument("--dry-run", action="store_true", help="только показать, что будет скачано")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        log("⚠ GITHUB_TOKEN не задан — поиск будет медленным (лимит ~10 запросов/мин). Создай токен без прав: github.com/settings/tokens")
    allowed = None if args.licenses.strip().lower() == "any" else {x.strip() for x in args.licenses.split(",") if x.strip()}
    out = args.out
    os.makedirs(out, exist_ok=True)
    if not shutil.which("git"):
        sys.exit("❌ git не найден")

    log(f"Ищу {args.count} репозиториев Rust (звёзды ≥ {args.min_stars}, лицензии: {'любые' if allowed is None else ', '.join(sorted(allowed))})…")
    cands, skipped = find_candidates(args, token, allowed)
    with open(os.path.join(out, "skipped.txt"), "a", encoding="utf-8") as f:
        for n, why in skipped:
            f.write(f"{n}\t{why}\n")
    log(f"Подходит: {len(cands)}, пропущено по лицензии/размеру: {len(skipped)}")
    if args.dry_run:
        for c in cands:
            log(f"  ★{c['stars']:>6}  {c['license']:<14} {c['size_mb']:>8} МБ  {c['name']}")
        return

    ok = fail = exists = 0
    meta_path = os.path.join(out, "repos.jsonl")
    t0 = time.time()
    stop = False
    with ThreadPoolExecutor(max_workers=args.jobs) as ex, open(meta_path, "a", encoding="utf-8") as meta:
        for i, (c, st, err) in enumerate(ex.map(lambda c: clone_one(c, out, args), cands), 1):
            if st == "ok":
                ok += 1
                meta.write(json.dumps(c, ensure_ascii=False) + "\n")
                meta.flush()
                log(f"[{i}/{len(cands)}] ✓ {c['name']}  (★{c['stars']}, {c['license']})")
            elif st == "exists":
                exists += 1
            elif st == "nodisk":
                log(f"⛔ остановка: {err}")
                stop = True
                break
            else:
                fail += 1
                log(f"[{i}/{len(cands)}] ✗ {c['name']}: {err}")
    log(f"\nГотово за {time.time()-t0:.0f} с: скачано {ok}, уже было {exists}, ошибок {fail}{', остановлено из-за места' if stop else ''}.")
    log(f"Метаданные: {meta_path}")
    log(f'Дальше: cargo run --release --bin scan -- --paths "{out}"')


if __name__ == "__main__":
    main()
