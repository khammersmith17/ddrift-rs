# ddrift
This crate implements utilities to compute drift between one dataset. The use case that inspires this crate is data drift monitoring in the context of production machine learning, where data drift can serve as an effective monitoring lever to determine when a model may need to be retrained.

This crate provides a wide array of statistical methods to determine the drift, or "distance", between two datasets. Drift in a dataset can be useful in cases where a downstream process or business context is sensitive to shifts in the distribution of data, such as a machine learning context. These methods can also be used to enforce the integrity of a dataset, by comparing it against some well known distribution, in this crate this is called the baseline.

This crate attempts to provide efficient implementations that to approximate a dataset using quantile binning for continuous distribtions, and value binning for categorical datasets. These dataset approximations can represent the distribution of a dataset into a space efficient representation. This representation also yield effective in computing the drift between two datasets using statistical methods.

This crate supports both datasets that are expected to be nonnull, and datasets that allow for nullable values.
