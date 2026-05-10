use crate::{
    core::distribution::QuantileType, core::error::DriftExportError,
    drift_types::stream_mode::StreamingDriftMode,
};
use num_traits::Float;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::path::PathBuf;

pub trait LoadDataDriftExport: DeserializeOwned {
    fn from_file(filepath: PathBuf) -> Result<Self, DriftExportError> {
        let file_data = std::fs::read_to_string(filepath)?;
        let export: Self = serde_json::from_str(&file_data)?;
        Ok(export)
    }

    fn from_bytes(payload: &[u8]) -> Result<Self, DriftExportError> {
        let export: Self = serde_json::from_slice(payload)?;
        Ok(export)
    }

    fn from_str(payload: &str) -> Result<Self, DriftExportError> {
        let export: Self = serde_json::from_str(payload)?;
        Ok(export)
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ContinuousDriftBaselineExport<T> {
    pub bin_edges: Vec<T>,
    pub baseline_hist: Vec<f64>,
    pub quantile_type: QuantileType,
}

impl<T: Float + DeserializeOwned> LoadDataDriftExport for ContinuousDriftBaselineExport<T> {}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingContinuousBaseExport<T> {
    pub baseline: ContinuousDriftBaselineExport<T>,
    pub stream_mode: StreamingDriftMode,
}

impl<T: Float + DeserializeOwned> LoadDataDriftExport for StreamingContinuousBaseExport<T> {}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingContinuousStatefulExport<T> {
    pub stream_bins: Vec<f64>,
    pub baseline: ContinuousDriftBaselineExport<T>,
    pub stream_mode: StreamingDriftMode,
}

impl<T: Float + DeserializeOwned> LoadDataDriftExport for StreamingContinuousStatefulExport<T> {}

#[derive(Debug, Deserialize, Serialize)]
pub struct CategoricalDriftBaselineExport {
    pub baseline_hist: Vec<f64>,
    pub baseline_values: Vec<Value>,
    pub n: f64,
}

impl LoadDataDriftExport for CategoricalDriftBaselineExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingCategoricalStatefulExport {
    pub baseline: CategoricalDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
    pub stream_bins: Vec<f64>,
    pub total_stream_size: f64,
}

impl LoadDataDriftExport for StreamingCategoricalStatefulExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct NullableCategoricalDriftBaselineExport {
    pub baseline_hist: Vec<f64>,
    pub baseline_values: Vec<Value>,
    pub n: f64,
    pub null_n: f64,
}

impl LoadDataDriftExport for NullableCategoricalDriftBaselineExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct StreamingCategoricalBaseExport {
    pub baseline: CategoricalDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
}

impl LoadDataDriftExport for StreamingCategoricalBaseExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct NullableStreamingCategoricalBaseExport {
    pub baseline: NullableCategoricalDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
}

impl LoadDataDriftExport for NullableStreamingCategoricalBaseExport {}

#[derive(Debug, Deserialize, Serialize)]
pub struct NullableStreamingCategoricalStatefulExport {
    pub stream_bins: Vec<f64>,
    pub baseline: NullableCategoricalDriftBaselineExport,
    pub stream_mode: StreamingDriftMode,
    pub total_n: f64,
    pub null_n: f64,
}

impl LoadDataDriftExport for NullableStreamingCategoricalStatefulExport {}
