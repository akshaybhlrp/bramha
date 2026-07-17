use rayon::ThreadPool;

/// Gets the estimated physical core count (logical cores / 2) to prevent hyperthreading resource contention.
pub fn get_physical_core_count() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    if logical > 1 { logical / 2 } else { 1 }
}

/// Pins the current thread to a specific physical core ID (Linux only, cross-platform fallback).
pub fn pin_thread_to_core(core_id: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(core_id, &mut set);
        let tid = libc::syscall(libc::SYS_gettid) as i32;
        if libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            eprintln!("⚠️ Failed to pin thread to physical core {}", core_id);
        } else {
            // println!("📌 Pinned Rayon worker thread to physical core {}", core_id);
        }
    }
}

/// Constructs a physical-core-pinned Rayon thread pool to enforce high-performance compute bounds.
pub fn build_pinned_rayon_pool() -> ThreadPool {
    let thread_count = get_physical_core_count();
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .start_handler(move |i| {
            pin_thread_to_core(i);
        })
        .build()
        .unwrap_or_else(|e| {
            eprintln!(
                "⚠️ Failed to build pinned rayon pool: {}. Falling back to default pool.",
                e
            );
            rayon::ThreadPoolBuilder::new().build().unwrap()
        })
}

/// Reusable global static reference to the pinned Rayon pool to ensure zero-cost sharing
pub fn global_rayon_pool() -> &'static ThreadPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(build_pinned_rayon_pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_pool_creation() {
        let pool = global_rayon_pool();
        let cores = get_physical_core_count();
        assert_eq!(pool.current_num_threads(), cores);

        let sum: i32 = pool.install(|| (0..100).sum());
        assert_eq!(sum, 4950);
    }
}
