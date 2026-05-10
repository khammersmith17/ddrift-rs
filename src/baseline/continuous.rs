use crate::{
    core::{
        bin_edges::ContinuousBinEdges,
        compute_dataset_from_bins_continuous, compute_new_hist_prob,
        distribution::{MIN_BIN_CLAMP, QuantileType},
        error::{DriftError, DriftExportError},
    },
    export::ContinuousDriftBaselineExport,
};
use num_traits::Float;
use std::cmp::Ordering;

fn dataset_contains_nans<T: Float>(data: &[T]) -> bool {
    data.iter().any(|value| value.is_nan())
}

fn sort_baseline_data_opt<T: Float>(data: &[Option<T>]) -> Result<(Vec<T>, usize), DriftError> {
    // Take user data, filter Nones.
    // Clone is to not mangle users data.
    let mut non_none: Vec<T> = data.iter().filter_map(|entry| entry.clone()).collect();

    // NaNs should be represented as None.
    if dataset_contains_nans(&non_none) {
        return Err(DriftError::NaNValueError);
    }

    // SAFETY: Validated to not have NaNs, thus NaN ordering ambiguity does not apply.
    non_none.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let null_count = data.len() - non_none.len();
    Ok((non_none, null_count))
}

fn sort_baseline_data<T: Float>(data: &[T]) -> Result<Vec<T>, DriftError> {
    if data.len() <= 1 {
        return Err(DriftError::EmptyBaselineData);
    }

    // do not accept NaNs
    if dataset_contains_nans(data) {
        return Err(DriftError::NaNValueError);
    }

    // To not mangle users data.
    let mut sorted_baseline = data.to_vec();

    // SAFETY: Validated to not have NaNs, thus NaN ordering ambiguity does not apply.
    sorted_baseline.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(sorted_baseline)
}

#[derive(Clone, Debug)]
pub(crate) struct NullableBaselineContinuousBins<T: Float> {
    pub bin_edges: ContinuousBinEdges<T>,
    pub baseline_hist: Vec<f64>,
    quantile_type: QuantileType,
    null_n: f64,
}

impl<T: Float> NullableBaselineContinuousBins<T> {
    pub(crate) fn n_bins(&self) -> usize {
        self.bin_edges.n_bins()
    }

    pub fn export_bin_edges(&self) -> Vec<T> {
        self.bin_edges.export_edges()
    }

    // Resolve the bin a particular data example falls into.
    #[inline]
    pub(crate) fn resolve_bin(&self, sample: T) -> usize {
        self.bin_edges.resolve_bin(sample)
    }

    pub(crate) fn export_baseline(&self) -> Vec<f64> {
        self.baseline_hist.clone()
    }
}

impl<T: Float + Send + Sync> NullableBaselineContinuousBins<T> {
    pub(crate) fn new(
        baseline_data: &[Option<T>],
        quantile_type_opt: Option<QuantileType>,
    ) -> Result<NullableBaselineContinuousBins<T>, DriftError> {
        let quantile_type = quantile_type_opt.unwrap_or_default();
        let (sorted_baseline, null_n) = sort_baseline_data_opt(baseline_data)?;
        let bin_edges: ContinuousBinEdges<T> =
            ContinuousBinEdges::new_from_dataset_with_quantile_type(
                &sorted_baseline,
                quantile_type,
            );

        let baseline_hist = compute_new_hist_prob(
            baseline_data.len(),
            &compute_dataset_from_bins_continuous(&sorted_baseline, &bin_edges),
        )?;

        Ok(NullableBaselineContinuousBins {
            bin_edges,
            baseline_hist,
            quantile_type,
            null_n: null_n as f64,
        })
    }

    pub(crate) fn reset(&mut self, baseline_data: &[Option<T>]) -> Result<(), DriftError> {
        let q_type = self.quantile_type;
        *self = Self::new(baseline_data, Some(q_type))?;
        Ok(())
    }
}

// Break out baseline to have shared logic between the discrete and the streaming variants of drift
// utilities.
// Also allows for more elegant composition of different usage
#[derive(Clone, Debug)]
pub(crate) struct BaselineContinuousBins<T: Float> {
    pub bin_edges: ContinuousBinEdges<T>,
    pub baseline_hist: Vec<f64>,
    quantile_type: QuantileType,
}

impl<T: Float + serde::de::DeserializeOwned> TryFrom<ContinuousDriftBaselineExport<T>>
    for BaselineContinuousBins<T>
{
    type Error = DriftExportError;
    fn try_from(export: ContinuousDriftBaselineExport<T>) -> Result<Self, Self::Error> {
        let ContinuousDriftBaselineExport {
            bin_edges: raw_bin_edges,
            baseline_hist,
            quantile_type,
        } = export;
        let n_bins = baseline_hist.len();
        if raw_bin_edges.len() != n_bins - 2 || n_bins < MIN_BIN_CLAMP {
            return Err(DriftExportError::InvalidDataShape);
        }

        let bin_edges = ContinuousBinEdges::new_from_parts(raw_bin_edges);
        Ok(BaselineContinuousBins {
            bin_edges,
            baseline_hist,
            quantile_type,
        })
    }
}

impl<T: Float + serde::Serialize> From<BaselineContinuousBins<T>>
    for ContinuousDriftBaselineExport<T>
{
    fn from(baseline: BaselineContinuousBins<T>) -> ContinuousDriftBaselineExport<T> {
        let BaselineContinuousBins {
            bin_edges: bin_edges_outer,
            baseline_hist,
            quantile_type,
            ..
        } = baseline;

        ContinuousDriftBaselineExport {
            bin_edges: bin_edges_outer.take_edges(),
            baseline_hist,
            quantile_type,
        }
    }
}

impl<T: Float + serde::de::DeserializeOwned> BaselineContinuousBins<T> {
    pub(crate) fn new_from_export(
        export: ContinuousDriftBaselineExport<T>,
    ) -> Result<BaselineContinuousBins<T>, DriftExportError> {
        Self::try_from(export)
    }
}

impl<T: Float> BaselineContinuousBins<T> {
    pub(crate) fn n_bins(&self) -> usize {
        self.bin_edges.n_bins()
    }

    pub fn export_bin_edges(&self) -> Vec<T> {
        self.bin_edges.export_edges()
    }

    // Resolve the bin a particular data example falls into.
    #[inline]
    pub(crate) fn resolve_bin(&self, sample: T) -> usize {
        self.bin_edges.resolve_bin(sample)
    }

    pub(crate) fn export_baseline(&self) -> Vec<f64> {
        self.baseline_hist.clone()
    }
}

impl<T: Float + Send + Sync> BaselineContinuousBins<T> {
    // Constructor on a baseline dataset. Allocates then hyrdates with the provided baseline
    // dataset.
    pub(crate) fn new(
        baseline_data: &[T],
        quantile_resolution: QuantileType,
    ) -> Result<BaselineContinuousBins<T>, DriftError> {
        let sorted_baseline = sort_baseline_data(baseline_data)?;
        let bin_edges = ContinuousBinEdges::new_from_dataset_with_quantile_type(
            &sorted_baseline,
            quantile_resolution,
        );

        let baseline_hist = compute_new_hist_prob(
            baseline_data.len(),
            &compute_dataset_from_bins_continuous(baseline_data, &bin_edges),
        )?;

        Ok(BaselineContinuousBins {
            bin_edges,
            baseline_hist,
            quantile_type: quantile_resolution,
        })
    }

    // call into init method
    pub(crate) fn reset(&mut self, baseline_data: &[T]) -> Result<(), DriftError> {
        let sorted_baseline = sort_baseline_data(baseline_data)?;
        self.bin_edges = ContinuousBinEdges::new_from_dataset_with_quantile_type(
            &sorted_baseline,
            self.quantile_type,
        );

        self.baseline_hist = compute_new_hist_prob(
            baseline_data.len(),
            &compute_dataset_from_bins_continuous(baseline_data, &self.bin_edges),
        )?;
        Ok(())
    }
}
