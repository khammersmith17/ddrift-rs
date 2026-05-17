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
    #[inline]
    pub(crate) fn touch_flush_ts(&mut self) {
        match self {
            StreamModeInner::Flush { last_flush_ts, .. } => {
                *last_flush_ts = Instant::now();
            }
            _ => {}
        }
    }
    /// Resets state on flush.
    #[inline]
    pub(crate) fn perform_flush(&mut self, bins: &mut [f64], n: &mut f64) {
        match self {
            StreamModeInner::Flush { last_flush_ts, .. } => {
                bins.fill(0_f64);
                *n = 0_f64;
                *last_flush_ts = Instant::now();
            }
            _ => {}
        }
    }

    /// Determine if a flush is needed. When mode is using ExponentialDecay, this should be
    /// compiled out in release mode.
    #[inline]
    pub(crate) fn needs_flush(&self, total_stream_size: f64) -> bool {
        match self {
            StreamModeInner::Flush {
                size,
                cadence,
                last_flush_ts,
            } => {
                // Always flush on size.
                // If not size, amortize the time check every 255 items.
                // Will only check size when least signifcant byte is full.
                total_stream_size >= *size
                    || (total_stream_size as usize & constants::FLUSH_CHECK_OFFSET == 0
                        && Instant::now().duration_since(*last_flush_ts) >= *cadence)
            }
            StreamModeInner::ExponentialDecay(_) => false,
        }
    }

    /// Fetch the number of seconds since last flush.
    #[inline]
    pub(crate) fn last_flush(&self) -> u64 {
        match self {
            StreamModeInner::Flush { last_flush_ts, .. } => {
                Instant::now().duration_since(*last_flush_ts).as_secs()
            }
            StreamModeInner::ExponentialDecay(_) => u64::default(),
        }
    }

    #[inline]
    pub(crate) fn apply_decay(&self, bins: &mut [f64], n: &mut f64) {
        match self {
            StreamModeInner::ExponentialDecay(decay_factor) => {
                bins.iter_mut().for_each(|b| *b *= decay_factor);
                *n *= decay_factor
            }
            _ => {}
        }
    }

    #[inline]
    pub(crate) fn apply_nullable_decay(
        &self,
        bins: &mut [f64],
        total_n: &mut f64,
        null_n: &mut f64,
    ) {
        match self {
            StreamModeInner::ExponentialDecay(decay_factor) => {
                bins.iter_mut().for_each(|b| *b *= decay_factor);
                *total_n *= decay_factor;
                *null_n *= decay_factor;
            }
            _ => {}
        }
    }
}
