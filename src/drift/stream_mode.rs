use crate::constants;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[non_exhaustive]
#[derive(Debug, Serialize, Deserialize)]
pub enum StreamingDriftMode {
    Flush { size: u64, cadence: u64 },
    ExponentialDecay(f64),
}

impl Default for StreamingDriftMode {
    fn default() -> StreamingDriftMode {
        StreamingDriftMode::Flush {
            size: constants::DEFAULT_MAX_STREAM_SIZE,
            cadence: constants::DEFAULT_STREAM_FLUSH_CADENCE,
        }
    }
}

#[derive(Debug)]
pub(crate) enum StreamModeInner {
    Flush {
        size: f64,
        cadence: Duration,
        last_flush_ts: Instant,
    },
    ExponentialDecay(f64),
}

impl From<StreamModeInner> for StreamingDriftMode {
    fn from(mode: StreamModeInner) -> StreamingDriftMode {
        match mode {
            StreamModeInner::Flush { size, cadence, .. } => StreamingDriftMode::Flush {
                size: size as u64,
                cadence: cadence.as_secs(),
            },
            StreamModeInner::ExponentialDecay(decay_factor) => {
                StreamingDriftMode::ExponentialDecay(decay_factor)
            }
        }
    }
}

impl From<StreamingDriftMode> for StreamModeInner {
    fn from(mode: StreamingDriftMode) -> StreamModeInner {
        match mode {
            StreamingDriftMode::Flush { size, cadence } => StreamModeInner::Flush {
                size: size as f64,
                cadence: Duration::from_secs(cadence),
                last_flush_ts: Instant::now(),
            },
            StreamingDriftMode::ExponentialDecay(decay_factor) => {
                StreamModeInner::ExponentialDecay(decay_factor)
            }
        }
    }
}

impl StreamModeInner {
    /// Resets state on flush.
    pub(crate) fn perform_flush(&mut self) {
        match self {
            StreamModeInner::Flush { last_flush_ts, .. } => {
                *last_flush_ts = Instant::now();
            }
            _ => {}
        }
    }

    /// Determine if a flush is needed. When mode is using ExponentialDecay, this should be
    /// compiled out in release mode.
    pub(crate) fn needs_flush(&self, total_stream_size: f64) -> bool {
        match self {
            StreamModeInner::Flush {
                size,
                cadence,
                last_flush_ts,
            } => {
                // First check size.
                // If size is valid, mask the Instant check to only check every 255
                // This introduces small error and amortizes the Instant check. Instant check is
                // non trivial expensive.
                total_stream_size >= *size
                    || (total_stream_size as usize & 255 == 0
                        && Instant::now().duration_since(*last_flush_ts) >= *cadence)
            }
            StreamModeInner::ExponentialDecay(_) => false,
        }
    }

    /// Fetch the number of seconds since last flush.
    pub(crate) fn last_flush(&self) -> u64 {
        match self {
            StreamModeInner::Flush { last_flush_ts, .. } => {
                Instant::now().duration_since(*last_flush_ts).as_secs()
            }
            StreamModeInner::ExponentialDecay(_) => u64::default(),
        }
    }
}
