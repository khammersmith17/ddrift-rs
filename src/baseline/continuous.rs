use crate::{
    core::{
        bin_edges::{ContinuousBinEdges, NullableContinuousBinEdges},
        compute_dataset_from_bins_continuous, compute_new_hist_prob,
        distribution::{MIN_BIN_CLAMP, QuantileType},
        error::{DriftError, DriftExportError},
    },
    export::{ContinuousDriftBaselineExport, NullableContinuousDriftBaselineExport},
};
use num_traits::Float;
use std::cmp::Ordering;

// Non Option NaNs are not supported.
fn dataset_contains_nans<T: Float>(data: &[T]) -> bool {
    data.iter().any(|value| value.is_nan())
}

// Sort optional data. Makes a copy on non None data and returns the number of None elements in the
// provided slice.
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
    bin_edges: NullableContinuousBinEdges<T>,
    baseline_hist: Vec<f64>,
    quantile_type: QuantileType,
    sample_size: f64,
    null_count: f64,
}

impl<T: Float> NullableBaselineContinuousBins<T> {
    pub(crate) fn bin_edges(&self) -> &ContinuousBinEdges<T> {
        self.bin_edges.inner_ref()
    }

    pub(crate) fn baseline_bins(&self) -> &[f64] {
        &self.baseline_hist
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.bin_edges.n_bins()
    }

    pub fn export_bin_edges(&self) -> Vec<T> {
        self.bin_edges.export_edges()
    }

    // Resolve the bin a particular data example falls into.
    #[inline]
    pub(crate) fn resolve_bin(&self, sample: Option<T>) -> Option<usize> {
        self.bin_edges.resolve_bin(sample)
    }

    pub(crate) fn export_baseline(&self) -> Vec<f64> {
        self.baseline_hist.clone()
    }

    /// Non-none population size.
    pub(crate) fn population_size(&self) -> f64 {
        self.sample_size - self.null_count
    }
}

impl<T: Float + Send + Sync> NullableBaselineContinuousBins<T> {
    pub(crate) fn new(
        baseline_data: &[Option<T>],
        quantile_type_opt: Option<QuantileType>,
    ) -> Result<NullableBaselineContinuousBins<T>, DriftError> {
        let sample_size = baseline_data.len() as f64;
        let quantile_type = quantile_type_opt.unwrap_or_default();
        let (sorted_baseline, null_count) = sort_baseline_data_opt(baseline_data)?;
        let bin_edges_inner: ContinuousBinEdges<T> =
            ContinuousBinEdges::new_from_dataset_with_quantile_type(
                &sorted_baseline,
                quantile_type,
            );

        let baseline_hist =
            compute_dataset_from_bins_continuous(&sorted_baseline, &bin_edges_inner);
        let bin_edges = NullableContinuousBinEdges::new(bin_edges_inner);

        Ok(NullableBaselineContinuousBins {
            bin_edges,
            baseline_hist,
            quantile_type,
            sample_size,
            null_count: null_count as f64,
        })
    }

    pub(crate) fn reset(&mut self, baseline_data: &[Option<T>]) -> Result<(), DriftError> {
        let q_type = self.quantile_type;
        *self = Self::new(baseline_data, Some(q_type))?;
        Ok(())
    }
}

impl<T: Float + serde::de::DeserializeOwned> TryFrom<NullableContinuousDriftBaselineExport<T>>
    for NullableBaselineContinuousBins<T>
{
    type Error = DriftExportError;
    fn try_from(export: NullableContinuousDriftBaselineExport<T>) -> Result<Self, Self::Error> {
        let NullableContinuousDriftBaselineExport {
            bin_edges: raw_bin_edges,
            baseline_hist,
            quantile_type,
            null_count,
            sample_size,
        } = export;
        let n_bins = baseline_hist.len();
        if raw_bin_edges.len() != n_bins - 2 || n_bins < MIN_BIN_CLAMP {
            return Err(DriftExportError::InvalidDataShape);
        }
        let bin_edges_inner = ContinuousBinEdges::new_from_parts(raw_bin_edges);
        let bin_edges = NullableContinuousBinEdges::new(bin_edges_inner);

        Ok(NullableBaselineContinuousBins {
            bin_edges,
            baseline_hist,
            quantile_type,
            null_count,
            sample_size,
        })
    }
}

impl<T: Float + serde::Serialize> From<NullableBaselineContinuousBins<T>>
    for NullableContinuousDriftBaselineExport<T>
{
    fn from(baseline: NullableBaselineContinuousBins<T>) -> Self {
        NullableContinuousDriftBaselineExport {
            bin_edges: baseline.bin_edges.take_edges(),
            baseline_hist: baseline.baseline_hist,
            quantile_type: baseline.quantile_type,
            null_count: baseline.null_count,
            sample_size: baseline.sample_size,
        }
    }
}

impl<T: Float + serde::de::DeserializeOwned> NullableBaselineContinuousBins<T> {
    pub(crate) fn new_from_export(
        export: NullableContinuousDriftBaselineExport<T>,
    ) -> Result<NullableBaselineContinuousBins<T>, DriftExportError> {
        Self::try_from(export)
    }
}

// Break out baseline to have shared logic between the discrete and the streaming variants of drift
// utilities.
// Also allows for more elegant composition of different usage
#[derive(Clone, Debug)]
pub(crate) struct BaselineContinuousBins<T: Float> {
    bin_edges: ContinuousBinEdges<T>,
    baseline_hist: Vec<f64>,
    sample_size: f64,
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
            sample_size,
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
            sample_size,
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
            sample_size,
        } = baseline;

        ContinuousDriftBaselineExport {
            bin_edges: bin_edges_outer.take_edges(),
            baseline_hist,
            quantile_type,
            sample_size,
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

    pub(crate) fn bin_edges(&self) -> &ContinuousBinEdges<T> {
        &self.bin_edges
    }

    pub fn export_bin_edges(&self) -> Vec<T> {
        self.bin_edges.export_edges()
    }

    pub fn baseline_bins(&self) -> &[f64] {
        &self.baseline_hist
    }

    // Resolve the bin a particular data example falls into.
    #[inline]
    pub(crate) fn resolve_bin(&self, sample: T) -> usize {
        self.bin_edges.resolve_bin(sample)
    }

    pub(crate) fn export_baseline(&self) -> Vec<f64> {
        self.baseline_hist.clone()
    }

    pub(crate) fn population_size(&self) -> f64 {
        self.sample_size
    }
}

impl<T: Float + Send + Sync> BaselineContinuousBins<T> {
    // Constructor on a baseline dataset. Allocates then hyrdates with the provided baseline
    // dataset.
    pub(crate) fn new(
        baseline_data: &[T],
        quantile_resolution: QuantileType,
    ) -> Result<BaselineContinuousBins<T>, DriftError> {
        let sample_size = baseline_data.len() as f64;
        let sorted_baseline = sort_baseline_data(baseline_data)?;
        let bin_edges = ContinuousBinEdges::new_from_dataset_with_quantile_type(
            &sorted_baseline,
            quantile_resolution,
        );

        let baseline_hist = compute_dataset_from_bins_continuous(baseline_data, &bin_edges);

        Ok(BaselineContinuousBins {
            bin_edges,
            sample_size,
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
