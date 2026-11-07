//! Утилиты: ГСЧ, форматирование, корректная остановка по Ctrl+C.
use std::sync::atomic::{AtomicBool, Ordering};

pub fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// splitmix64 — нормальный ГСЧ (старые LCG `% N` брали слабые младшие биты)
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(mix(seed ^ 0xA076_1D64_78BD_642F))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// равномерное [0,1)
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 16_777_216.0
    }
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }
    pub fn normal(&mut self) -> f32 {
        let u1 = self.f32().max(1e-7);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    /// число в [0, n)
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

pub fn dur(secs: f64) -> String {
    if !secs.is_finite() {
        return "—".to_string();
    }
    if secs < 60.0 {
        format!("{:.1}с", secs)
    } else if secs < 3600.0 {
        format!("{:.0}м {:02.0}с", (secs / 60.0).floor(), secs % 60.0)
    } else {
        format!("{:.0}ч {:02.0}м", (secs / 3600.0).floor(), ((secs % 3600.0) / 60.0).floor())
    }
}

pub fn mb(bytes: u64) -> String {
    let m = bytes as f64 / 1024.0 / 1024.0;
    if m >= 1024.0 {
        format!("{:.2} ГБ", m / 1024.0)
    } else {
        format!("{:.1} МБ", m)
    }
}

pub fn count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2} млрд", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2} млн", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1} тыс", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

// ───────────── Ctrl+C → сохранить чекпойнт и выйти ─────────────
pub static STOP: AtomicBool = AtomicBool::new(false);

pub fn stop_requested() -> bool {
    STOP.load(Ordering::Relaxed)
}

#[cfg(unix)]
pub fn install_ctrlc() {
    extern "C" fn handler(_sig: i32) {
        STOP.store(true, Ordering::Relaxed);
    }
    extern "C" {
        fn signal(sig: i32, handler: usize) -> usize;
    }
    unsafe {
        let f: extern "C" fn(i32) = handler;
        signal(2, f as usize);
    }
}

#[cfg(windows)]
pub fn install_ctrlc() {
    unsafe extern "system" fn handler(_ctrl_type: u32) -> i32 {
        STOP.store(true, Ordering::Relaxed);
        1
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn SetConsoleCtrlHandler(h: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
    }
    unsafe {
        SetConsoleCtrlHandler(Some(handler), 1);
    }
}

#[cfg(not(any(unix, windows)))]
pub fn install_ctrlc() {}
