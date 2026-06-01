pub mod categorical;
pub mod continuous;
#[cfg(feature = "arrow")]
pub mod table;

pub use categorical::{BaselineCategoricalBins, NullableBaselineCategoricalBins};
pub use continuous::{BaselineContinuousBins, NullableBaselineContinuousBins};
