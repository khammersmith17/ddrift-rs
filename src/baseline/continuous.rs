use crate::{
    core::{
        bin_edges::ContinuousBinEdges,
        compute_dataset_from_bins_continuous,
        distribution::{MIN_BIN_CLAMP, QuantileType},
        error::{DriftError, DriftExportError},
    },
    drift::{DriftActor, NullableDriftActor},
    export::{ContinuousDriftBaselineExport, NullableContinuousDriftBaselineExport},
};

#[cfg(feature = "arrow")]
use arrow::array::{Float32Array, Float64Array};
use num_traits::Float;
use std::cmp::Ordering;

#[cfg(feature = "arrow")]
fn sort_float32_array(array: &Float32Array) -> (Vec<f32>, usize) {
    let mut concrete: Vec<f32> = array.iter().filter_map(|entry| entry).collect();
    concrete.sort_by(|a, b| a.total_cmp(b));
    let concrete_len = concrete.len();
    (concrete, array.len() - concrete_len)
}

#[cfg(feature = "arrow")]
fn sort_float64_array(array: &Float64Array) -> (Vec<f64>, usize) {
    let mut concrete: Vec<f64> = array.iter().filter_map(|entry| entry).collect();
    concrete.sort_by(|a, b| a.total_cmp(b));
    let concrete_len = concrete.len();
    (concrete, array.len() - concrete_len)
}

// Non Option NaNs are not supported.
fn dataset_contains_nans<T: Float>(data: &[T]) -> bool {
    data.iter().any(|value| value.is_nan())
}

// Sort optional data. Makes a copy on non None data and returns the number of None elements in the
// provided slice.
fn sort_baseline_data_opt<T: Float>(data: &[Option<T>]) -> Result<(Vec<T>, usize), DriftError> {
    // Take user data, filter Nones.
    // Clone is to not mangle users data.
    let mut non_none: Vec<T> = data.iter().filter_map(|entry| *entry).collect();

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
pub struct NullableBaselineContinuousBins<T: Float> {
    pub(crate) bin_edges: ContinuousBinEdges<T>,
    pub(crate) baseline_hist: Vec<f64>,
    pub(crate) quantile_type: QuantileType,
    pub(crate) sample_size: f64,
    pub(crate) null_count: f64,
}

impl<'a, T: Float> DriftActor<'a> for NullableBaselineContinuousBins<T> {
    fn quantile_bins(&'a self) -> &'a [f64] {
        &self.baseline_hist
    }

    fn example_count(&self) -> usize {
        self.sample_size as usize
    }
}

impl<'a, T: Float> NullableDriftActor<'a> for NullableBaselineContinuousBins<T> {
    fn null_count(&self) -> usize {
        self.null_count as usize
    }
}

impl<T: Float> NullableBaselineContinuousBins<T> {
    pub(crate) fn bin_edges(&self) -> &ContinuousBinEdges<T> {
        &self.bin_edges
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
    pub(crate) fn resolve_bin(&self, sample_opt: Option<T>) -> Option<usize> {
        let sample = sample_opt?;
        Some(self.bin_edges.resolve_bin(sample))
    }

    /// Non-none population size.
    pub(crate) fn population_size(&self) -> f64 {
        self.sample_size - self.null_count
    }
}

#[cfg(feature = "arrow")]
impl NullableBaselineContinuousBins<f32> {
    pub(crate) fn from_arrow32(
        array: &Float32Array,
        quantile_type_opt: Option<QuantileType>,
    ) -> NullableBaselineContinuousBins<f32> {
        let sample_size = array.len() as f64;
        let (sorted_baseline, null_count) = sort_float32_array(array);
        NullableBaselineContinuousBins::from_sorted_baseline(
            sorted_baseline,
            sample_size,
            null_count,
            quantile_type_opt.unwrap_or_default(),
        )
    }
}

#[cfg(feature = "arrow")]
impl NullableBaselineContinuousBins<f64> {
    pub(crate) fn from_arrow64(
        array: &Float64Array,
        quantile_type_opt: Option<QuantileType>,
    ) -> NullableBaselineContinuousBins<f64> {
        let sample_size = array.len() as f64;
        let (sorted_baseline, null_count) = sort_float64_array(array);
        NullableBaselineContinuousBins::from_sorted_baseline(
            sorted_baseline,
            sample_size,
            null_count,
            quantile_type_opt.unwrap_or_default(),
        )
    }
}

impl<T: Float + Send + Sync> NullableBaselineContinuousBins<T> {
    pub fn new(
        baseline_data: &[Option<T>],
        quantile_type_opt: Option<QuantileType>,
    ) -> Result<NullableBaselineContinuousBins<T>, DriftError> {
        let sample_size = baseline_data.len() as f64;
        let quantile_type = quantile_type_opt.unwrap_or_default();
        let (sorted_baseline, null_count) = sort_baseline_data_opt(baseline_data)?;
        Ok(NullableBaselineContinuousBins::from_sorted_baseline(
            sorted_baseline,
            sample_size,
            null_count,
            quantile_type,
        ))
    }

    fn from_sorted_baseline(
        sorted_baseline: Vec<T>,
        sample_size: f64,
        null_count: usize,
        quantile_type: QuantileType,
    ) -> NullableBaselineContinuousBins<T> {
        let bin_edges: ContinuousBinEdges<T> =
            ContinuousBinEdges::new_from_dataset_with_quantile_type(
                &sorted_baseline,
                quantile_type,
            );

        let baseline_hist = compute_dataset_from_bins_continuous(&sorted_baseline, &bin_edges);

        NullableBaselineContinuousBins {
            bin_edges,
            baseline_hist,
            quantile_type,
            sample_size,
            null_count: null_count as f64,
        }
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
        if raw_bin_edges.len() != n_bins - 1 || n_bins < MIN_BIN_CLAMP {
            return Err(DriftExportError::InvalidDataShape);
        }
        let bin_edges = ContinuousBinEdges::new_from_parts(raw_bin_edges);

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
pub struct BaselineContinuousBins<T: Float> {
    pub(crate) bin_edges: ContinuousBinEdges<T>,
    pub(crate) baseline_hist: Vec<f64>,
    pub(crate) sample_size: f64,
    pub(crate) quantile_type: QuantileType,
}

impl<'a, T: Float> DriftActor<'a> for BaselineContinuousBins<T> {
    fn quantile_bins(&'a self) -> &'a [f64] {
        &self.baseline_hist
    }

    fn example_count(&self) -> usize {
        self.sample_size as usize
    }
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
        if raw_bin_edges.len() != n_bins - 1 || n_bins < MIN_BIN_CLAMP {
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

    pub(crate) fn population_size(&self) -> f64 {
        self.sample_size
    }
}

impl<T: Float + Send + Sync> BaselineContinuousBins<T> {
    // Constructor on a baseline dataset. Allocates then hyrdates with the provided baseline
    // dataset.
    pub fn new(
        baseline_data: &[T],
        quantile_resolution: Option<QuantileType>,
    ) -> Result<BaselineContinuousBins<T>, DriftError> {
        let sample_size = baseline_data.len() as f64;
        let sorted_baseline = sort_baseline_data(baseline_data)?;
        let q_type = quantile_resolution.unwrap_or_default();
        let bin_edges =
            ContinuousBinEdges::new_from_dataset_with_quantile_type(&sorted_baseline, q_type);

        let baseline_hist = compute_dataset_from_bins_continuous(baseline_data, &bin_edges);

        Ok(BaselineContinuousBins {
            bin_edges,
            sample_size,
            baseline_hist,
            quantile_type: q_type,
        })
    }

    // call into init method
    pub(crate) fn reset(&mut self, baseline_data: &[T]) -> Result<(), DriftError> {
        let sorted_baseline = sort_baseline_data(baseline_data)?;
        self.bin_edges = ContinuousBinEdges::new_from_dataset_with_quantile_type(
            &sorted_baseline,
            self.quantile_type,
        );

        self.baseline_hist = compute_dataset_from_bins_continuous(baseline_data, &self.bin_edges);
        self.sample_size = baseline_data.len() as f64;
        Ok(())
    }
}
