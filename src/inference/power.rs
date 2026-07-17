use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

pub static POWER_LIMIT: AtomicU32 = AtomicU32::new(100);

pub fn set_power_limit(val: u32) {
    POWER_LIMIT.store(val.min(100), Ordering::SeqCst);
}

pub fn get_power_limit() -> u32 {
    if let Ok(env_val) = std::env::var("BRAMHA_POWER_LIMIT")
        && let Ok(parsed) = env_val.parse::<u32>() {
            return parsed.min(100);
        }
    POWER_LIMIT.load(Ordering::SeqCst)
}

/// Dynamic sleep throttler based on active calculation time.
/// Target: Limit utilization to N% (e.g., 50% power doubles total elapsed duration).
pub fn throttle_power(work_time: Duration) {
    let limit = get_power_limit();
    if limit >= 100 || limit == 0 {
        return;
    }

    let work_ns = work_time.as_nanos() as f64;
    let total_ns = work_ns / (limit as f64 / 100.0);
    let sleep_ns = total_ns - work_ns;

    if sleep_ns > 0.0 {
        std::thread::sleep(Duration::from_nanos(sleep_ns as u64));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_power_limit_set_get() {
        set_power_limit(50);
        assert_eq!(get_power_limit(), 50);
        set_power_limit(150); // Clamped to 100
        assert_eq!(get_power_limit(), 100);
    }

    #[test]
    fn test_throttle_power_duration() {
        set_power_limit(50); // 50% utilization, active time = sleep time
        let start = Instant::now();
        let work = Duration::from_millis(10);
        throttle_power(work);
        let elapsed = start.elapsed();
        // Should be at least 10ms of work + 10ms of sleep = 20ms total (give some room for timing jitter)
        assert!(elapsed >= Duration::from_millis(9));
    }
}
