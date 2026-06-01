use std::env;
use std::sync::OnceLock;
pub mod drift_thresholds;
use crate::core::drift_metrics::{CategoricalDriftMeasurement, ContinuousDriftMeasurement};

static MAX_THREADS: OnceLock<usize> = OnceLock::new();

pub(crate) const DEFAULT_STREAM_FLUSH_CADENCE: u64 = 3600 * 24;
pub(crate) const DEFAULT_MAX_STREAM_SIZE: u64 = 1_000_000_u64;
pub(crate) const DEFAULT_DECAY_HALF_LIFE: u64 = 86400; // Default half life 1 day
pub(crate) const FLUSH_CHECK_OFFSET: usize = 255;
const MAX_THREADS_ENV_KEY: &'static str = "DDRIFT_MAX_THREADS";

const MIN_EXAMPLES_PER_THREAD: usize = 10_000_usize;

pub const ALL_CONTINUOUS_DRIFT_MEASUREMENTS: [ContinuousDriftMeasurement; 6] = [
    ContinuousDriftMeasurement::JensenShannon,
    ContinuousDriftMeasurement::PopulationStabilityIndex,
    ContinuousDriftMeasurement::WassersteinDistance,
    ContinuousDriftMeasurement::KullbackLeibler,
    ContinuousDriftMeasurement::KolmogorovSmirnov,
    ContinuousDriftMeasurement::Hellinger,
];

pub const ALL_CATEGORICAL_DRIFT_MEASUREMENTS: [CategoricalDriftMeasurement; 7] = [
    CategoricalDriftMeasurement::JensenShannon,
    CategoricalDriftMeasurement::PopulationStabilityIndex,
    CategoricalDriftMeasurement::WassersteinDistance,
    CategoricalDriftMeasurement::KullbackLeibler,
    CategoricalDriftMeasurement::ChiSquared,
    CategoricalDriftMeasurement::Hellinger,
    CategoricalDriftMeasurement::GTest,
];

fn get_max_threads() -> usize {
    *MAX_THREADS.get_or_init(|| {
        let max_available_threads = if let Ok(nz_count) = std::thread::available_parallelism() {
            nz_count.get()
        } else {
            1
        };

        if let Ok(user_defined_threads_str) = env::var(MAX_THREADS_ENV_KEY) {
            user_defined_threads_str
                .parse::<usize>()
                .unwrap_or(max_available_threads)
        } else {
            max_available_threads
        }
    })
}

/// Get the available number of threads.
/// If the dataset size is less than the number of threads, use a single thread.
pub(crate) fn get_thread_count(n: usize) -> usize {
    let thread_count = get_max_threads();

    if thread_count > n {
        return 1;
    }
    thread_count.min((n / MIN_EXAMPLES_PER_THREAD).max(1))
}
