/// Runtime performance profiler for CPU inference pipeline.
/// Tracks timing of key operations to identify bottlenecks.
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct TimingStats {
    pub count: u32,
    pub total_us: u64,
    pub min_us: u64,
    pub max_us: u64,
}

impl Default for TimingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            total_us: 0,
            min_us: u64::MAX,
            max_us: 0,
        }
    }

    pub fn record(&mut self, duration_us: u64) {
        self.count += 1;
        self.total_us += duration_us;
        self.min_us = self.min_us.min(duration_us);
        self.max_us = self.max_us.max(duration_us);
    }

    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_us as f64 / self.count as f64
        }
    }

    pub fn total_ms(&self) -> f64 {
        self.total_us as f64 / 1000.0
    }
}

pub struct ProfileScope {
    name: String,
    start: Instant,
    profiler: &'static Profiler,
}

pub struct Profiler {
    stats: Mutex<HashMap<String, TimingStats>>,
}

static GLOBAL_PROFILER: std::sync::OnceLock<Profiler> = std::sync::OnceLock::new();

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            stats: Mutex::new(HashMap::new()),
        }
    }

    pub fn global() -> &'static Profiler {
        GLOBAL_PROFILER.get_or_init(Profiler::new)
    }

    pub fn record(&self, name: &str, duration: Duration) {
        let duration_us = duration.as_micros() as u64;
        let mut stats = self.stats.lock().unwrap();
        stats
            .entry(name.to_string())
            .or_insert_with(TimingStats::new)
            .record(duration_us);
    }

    pub fn scope(name: &'static str) -> ProfileScope {
        ProfileScope {
            name: name.to_string(),
            start: Instant::now(),
            profiler: Profiler::global(),
        }
    }

    pub fn reset(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.clear();
    }

    pub fn report(&self) -> String {
        let stats = self.stats.lock().unwrap();
        let mut lines = vec![
            "═══════════════════════════════════════════════════════════════".to_string(),
            "Performance Profile Report".to_string(),
            "═══════════════════════════════════════════════════════════════".to_string(),
        ];

        let mut entries: Vec<_> = stats.iter().collect();
        entries.sort_by(|a, b| b.1.total_us.cmp(&a.1.total_us));

        let total_time: u64 = stats.values().map(|s| s.total_us).sum();
        let total_time_ms = total_time as f64 / 1000.0;

        for (name, stat) in entries {
            let percentage = if total_time > 0 {
                (stat.total_us as f64 / total_time as f64) * 100.0
            } else {
                0.0
            };
            lines.push(format!(
                "  {:50} │ {:6} calls │ {:10.2}ms │ {:6.2}% │ avg: {:8.2}µs │ min: {:8}µs │ max: {:8}µs",
                name,
                stat.count,
                stat.total_ms(),
                percentage,
                stat.avg_us(),
                stat.min_us,
                stat.max_us
            ));
        }

        lines.push("═══════════════════════════════════════════════════════════════".to_string());
        lines.push(format!("Total Time: {:.2}ms", total_time_ms));
        lines.join("\n")
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.profiler.record(&self.name, duration);
    }
}

/// Macro for easy scoped profiling
#[macro_export]
macro_rules! profile {
    ($name:expr) => {
        $crate::inference::profiler::Profiler::scope($name)
    };
}
