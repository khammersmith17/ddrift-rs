pub mod profile {

    use crate::baseline::{
        BaselineCategoricalBins, BaselineContinuousBins, NullableBaselineCategoricalBins,
        NullableBaselineContinuousBins,
    };
    use crate::core::{
        bin_edges::{CategoricalBinEdges, ContinuousBinEdges},
        distribution::QuantileType,
        error::DriftError,
    };
    use ahash::HashMap;
    use num_traits::Float;
    use std::hash::Hash;
    pub struct CategoricalView<T: Ord + Clone + Hash> {
        pub size: usize,
        pub bin_counts: HashMap<T, usize>,
    }

    impl<T: Ord + Clone + Hash> From<BaselineCategoricalBins<T>> for CategoricalView<T> {
        fn from(baseline: BaselineCategoricalBins<T>) -> CategoricalView<T> {
            let BaselineCategoricalBins {
                bin_edges,
                sample_size,
                baseline_bins,
            } = baseline;
            let CategoricalBinEdges(mut idx_map) = bin_edges;
            // For each key, the index in idx_map can be used to resolve the offset in the bin vector
            // of where the value count is stored.
            for (_key, idx) in idx_map.iter_mut() {
                let value_count = baseline_bins[*idx];
                *idx = value_count as usize;
            }
            CategoricalView {
                bin_counts: idx_map,
                size: sample_size as usize,
            }
        }
    }

    impl<T: Ord + Hash + Clone> CategoricalView<T> {
        pub fn new(dataset: &[T]) -> Result<CategoricalView<T>, DriftError> {
            let bl = BaselineCategoricalBins::new(dataset)?;
            Ok(bl.into())
        }
    }

    pub struct NullableCategoricalView<T: Ord + Clone + Hash> {
        pub size: usize,
        pub bin_counts: HashMap<T, usize>,
        pub null_count: usize,
    }

    impl<T: Ord + Clone + Hash> From<NullableBaselineCategoricalBins<T>>
        for NullableCategoricalView<T>
    {
        fn from(baseline: NullableBaselineCategoricalBins<T>) -> NullableCategoricalView<T> {
            let NullableBaselineCategoricalBins {
                bin_edges,
                total_samples,
                null_samples,
                baseline_bins,
            } = baseline;
            let CategoricalBinEdges(mut idx_map) = bin_edges;
            // For each key, the index in idx_map can be used to resolve the offset in the bin vector
            // of where the value count is stored.
            for (_key, idx) in idx_map.iter_mut() {
                let value_count = baseline_bins[*idx];
                *idx = value_count as usize;
            }
            NullableCategoricalView {
                bin_counts: idx_map,
                size: total_samples as usize,
                null_count: null_samples as usize,
            }
        }
    }

    impl<T: Ord + Hash + Clone> NullableCategoricalView<T> {
        pub fn new(dataset: &[Option<T>]) -> Result<NullableCategoricalView<T>, DriftError> {
            let bl = NullableBaselineCategoricalBins::new(dataset)?;
            Ok(bl.into())
        }
    }

    pub struct ContinuousView<T: Float> {
        pub quantile_bins: Vec<f64>,
        pub bin_edges: ContinuousBinEdges<T>,
        pub size: usize,
    }

    impl<T: Float> From<BaselineContinuousBins<T>> for ContinuousView<T> {
        fn from(baseline: BaselineContinuousBins<T>) -> ContinuousView<T> {
            let BaselineContinuousBins {
                baseline_hist: quantile_bins,
                bin_edges,
                sample_size,
                ..
            } = baseline;
            ContinuousView {
                quantile_bins,
                bin_edges,
                size: sample_size as usize,
            }
        }
    }

    impl<T: Float + Send + Sync> ContinuousView<T> {
        pub fn new(
            dataset: &[T],
            quantile_type: Option<QuantileType>,
        ) -> Result<ContinuousView<T>, DriftError> {
            let bl = BaselineContinuousBins::new(dataset, quantile_type)?;
            Ok(bl.into())
        }
    }

    pub struct NullableContinuousView<T: Float> {
        pub quantile_bins: Vec<f64>,
        pub bin_edges: ContinuousBinEdges<T>,
        pub size: usize,
        pub null_count: usize,
    }

    impl<T: Float> From<NullableBaselineContinuousBins<T>> for NullableContinuousView<T> {
        fn from(baseline: NullableBaselineContinuousBins<T>) -> NullableContinuousView<T> {
            let NullableBaselineContinuousBins {
                baseline_hist: quantile_bins,
                bin_edges,
                sample_size,
                null_count,
                ..
            } = baseline;
            NullableContinuousView {
                quantile_bins,
                bin_edges,
                size: sample_size as usize,
                null_count: null_count as usize,
            }
        }
    }

    impl<T: Float + Send + Sync> NullableContinuousView<T> {
        pub fn new(
            dataset: &[Option<T>],
            quantile_type: Option<QuantileType>,
        ) -> Result<NullableContinuousView<T>, DriftError> {
            let bl = NullableBaselineContinuousBins::new(dataset, quantile_type)?;
            Ok(bl.into())
        }
    }
}

pub mod candidate {
    use crate::core::{
        bin_edges::{CategoricalBinEdges, ContinuousBinEdges},
        compute_dataset_from_bins_categorical, compute_dataset_from_bins_categorical_parallel,
        compute_dataset_from_bins_continuous, compute_dataset_from_nullable_bins_categorical,
        compute_dataset_from_nullable_bins_categorical_parallel,
        compute_nullable_dataset_from_bins_continuous,
        error::DriftError,
    };
    use num_traits::Float;
    use std::hash::Hash;

    pub struct ContinuousCandidateView<'a, T: Float> {
        pub bin_edges: &'a ContinuousBinEdges<T>,
        pub quantile_bins: Vec<f64>,
        pub size: usize,
    }

    impl<'a, T: Float + Send + Sync> ContinuousCandidateView<'a, T> {
        pub fn from_bin_edges(
            dataset: &[T],
            bin_edges: &'a ContinuousBinEdges<T>,
        ) -> Result<ContinuousCandidateView<'a, T>, DriftError> {
            if dataset.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }

            let size = dataset.len();
            let quantile_bins = compute_dataset_from_bins_continuous(dataset, bin_edges);
            Ok(ContinuousCandidateView {
                bin_edges,
                quantile_bins,
                size,
            })
        }
    }

    pub struct NullableContinuousCandidateView<'a, T: Float> {
        pub bin_edges: &'a ContinuousBinEdges<T>,
        pub quantile_bins: Vec<f64>,
        pub size: usize,
        pub null_count: usize,
    }

    #[cfg(feature = "arrow")]
    impl<'a, T: Float + Send + Sync> NullableContinuousCandidateView<'a, T> {
        pub fn arrow_from_bin_edges(
            dataset: &[T],
            bin_edges: &'a ContinuousBinEdges<T>,
            null_buffer: Option<&arrow::buffer::NullBuffer>,
        ) -> Result<NullableContinuousCandidateView<'a, T>, DriftError> {
            if dataset.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }

            let size = dataset.len();
            let (quantile_bins, null_c) =
                crate::core::ddrift_arrow::compute_bins_continuous(dataset, bin_edges, null_buffer);
            Ok(NullableContinuousCandidateView {
                bin_edges,
                quantile_bins,
                size,
                null_count: null_c as usize,
            })
        }
    }

    impl<'a, T: Float + Send + Sync> NullableContinuousCandidateView<'a, T> {
        pub fn from_bin_edges(
            dataset: &[Option<T>],
            bin_edges: &'a ContinuousBinEdges<T>,
        ) -> Result<NullableContinuousCandidateView<'a, T>, DriftError> {
            if dataset.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }

            let size = dataset.len();
            let (quantile_bins, null_c) =
                compute_nullable_dataset_from_bins_continuous(dataset, bin_edges);
            Ok(NullableContinuousCandidateView {
                bin_edges,
                quantile_bins,
                size,
                null_count: null_c as usize,
            })
        }
    }

    pub struct CategoricalCandidateView<'a, T: Hash + Ord + Clone> {
        pub bin_edges: &'a CategoricalBinEdges<T>,
        pub quantile_bins: Vec<f64>,
        pub size: usize,
    }

    impl<'a, T: Hash + Ord + Clone> CategoricalCandidateView<'a, T> {
        pub fn from_bin_edges(
            dataset: &[T],
            bin_edges: &'a CategoricalBinEdges<T>,
        ) -> Result<CategoricalCandidateView<'a, T>, DriftError> {
            if dataset.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }
            let size = dataset.len();
            let quantile_bins = compute_dataset_from_bins_categorical(dataset, bin_edges);
            Ok(CategoricalCandidateView {
                bin_edges,
                quantile_bins,
                size,
            })
        }
    }

    impl<'a, T: Hash + Ord + Clone + Send + Sync> CategoricalCandidateView<'a, T> {
        pub fn from_bin_edges_parallel(
            dataset: &[T],
            bin_edges: &'a CategoricalBinEdges<T>,
        ) -> Result<CategoricalCandidateView<'a, T>, DriftError> {
            if dataset.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }
            let size = dataset.len();
            let quantile_bins = compute_dataset_from_bins_categorical_parallel(dataset, bin_edges);
            Ok(CategoricalCandidateView {
                bin_edges,
                quantile_bins,
                size,
            })
        }
    }

    pub struct NullableCategoricalCandidateView<'a, T: Hash + Ord + Clone> {
        pub bin_edges: &'a CategoricalBinEdges<T>,
        pub quantile_bins: Vec<f64>,
        pub size: usize,
        pub null_count: usize,
    }

    #[cfg(feature = "arrow")]
    impl<'a, T: Hash + Ord + Clone + Send + Sync> NullableCategoricalCandidateView<'a, T> {
        pub fn arrow_from_bin_edges(
            dataset: &[T],
            bin_edges: &'a CategoricalBinEdges<T>,
            null_buffer: Option<&arrow::buffer::NullBuffer>,
        ) -> Result<NullableCategoricalCandidateView<'a, T>, DriftError> {
            if dataset.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }
            let size = dataset.len();
            let (quantile_bins, null_c) = crate::core::ddrift_arrow::compute_bins_categorical(
                dataset,
                bin_edges,
                null_buffer,
            );
            Ok(NullableCategoricalCandidateView {
                bin_edges,
                quantile_bins,
                size,
                null_count: null_c as usize,
            })
        }
    }

    #[cfg(feature = "arrow")]
    impl<'a> NullableCategoricalCandidateView<'a, String> {
        pub fn from_string_slice<
            'slice,
            S: crate::table::slice_impl::SliceImpl<&'slice str> + Send + Sync,
        >(
            slice: &S,
            bin_edges: &'a CategoricalBinEdges<String>,
        ) -> Result<NullableCategoricalCandidateView<'a, String>, DriftError> {
            if slice.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }

            let size = slice.len();
            let (quantile_bins, null_count) =
                crate::core::ddrift_arrow::compute_bins_arrow_string_slice(slice, bin_edges);
            Ok(NullableCategoricalCandidateView {
                bin_edges,
                quantile_bins,
                size,
                null_count: null_count as usize,
            })
        }
    }

    #[cfg(feature = "arrow")]
    impl<'a> NullableCategoricalCandidateView<'a, bool> {
        pub fn from_bool_slice<S: crate::table::slice_impl::SliceImpl<bool> + Send + Sync>(
            slice: &S,
            bin_edges: &'a CategoricalBinEdges<bool>,
        ) -> Result<NullableCategoricalCandidateView<'a, bool>, DriftError> {
            if slice.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }

            let size = slice.len();
            let (quantile_bins, null_count) =
                crate::core::ddrift_arrow::compute_bins_arrow_bool_slice(slice, bin_edges);
            Ok(NullableCategoricalCandidateView {
                bin_edges,
                quantile_bins,
                size,
                null_count: null_count as usize,
            })
        }
    }

    impl<'a, T: Hash + Ord + Clone> NullableCategoricalCandidateView<'a, T> {
        pub fn from_bin_edges(
            dataset: &[Option<T>],
            bin_edges: &'a CategoricalBinEdges<T>,
        ) -> Result<NullableCategoricalCandidateView<'a, T>, DriftError> {
            if dataset.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }
            let size = dataset.len();
            let (quantile_bins, null_c) =
                compute_dataset_from_nullable_bins_categorical(dataset, bin_edges);
            Ok(NullableCategoricalCandidateView {
                bin_edges,
                quantile_bins,
                size,
                null_count: null_c as usize,
            })
        }
    }

    impl<'a, T: Hash + Ord + Clone + Send + Sync> NullableCategoricalCandidateView<'a, T> {
        pub fn from_bin_edges_parallel(
            dataset: &[Option<T>],
            bin_edges: &'a CategoricalBinEdges<T>,
        ) -> Result<NullableCategoricalCandidateView<'a, T>, DriftError> {
            if dataset.is_empty() {
                return Err(DriftError::EmptyRuntimeData);
            }
            let size = dataset.len();
            let (quantile_bins, null_c) =
                compute_dataset_from_nullable_bins_categorical_parallel(dataset, bin_edges);
            Ok(NullableCategoricalCandidateView {
                bin_edges,
                quantile_bins,
                size,
                null_count: null_c as usize,
            })
        }
    }
}
