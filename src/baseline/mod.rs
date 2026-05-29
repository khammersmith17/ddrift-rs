#[cfg(feature = "arrow")]
pub mod arrow;
pub mod categorical;
pub mod continuous;

pub use categorical::{BaselineCategoricalBins, NullableBaselineCategoricalBins};
pub use continuous::{BaselineContinuousBins, NullableBaselineContinuousBins};
