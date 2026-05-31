use thiserror::Error;

#[cfg(feature = "arrow")]
use arrow::error::ArrowError;

#[cfg(feature = "arrow")]
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum DriftArrowError {
    #[error("Schema error: {0:?}")]
    SchemaError(crate::ddrift_arrow::schema_view::InvalidSchemaReport),
    #[error("Unsupported Arrow DataType: {0:?}")]
    UnsupportedArrowTypeError(arrow::datatypes::DataType),
    #[error("Drift Error: {0:?}")]
    DriftError(DriftError),
    #[error("Arrow Error: {0:?}")]
    ArrowError(ArrowError),
}

#[cfg(feature = "arrow")]
impl From<DriftError> for DriftArrowError {
    fn from(err: DriftError) -> DriftArrowError {
        DriftArrowError::DriftError(err)
    }
}

#[cfg(feature = "arrow")]
impl From<ArrowError> for DriftArrowError {
    fn from(err: ArrowError) -> DriftArrowError {
        DriftArrowError::ArrowError(err)
    }
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum DriftError {
    #[error("Data used for runtime drift analysis must be non empty")]
    EmptyRuntimeData,
    #[error("Unable to convert internal timestamp into DateTime object")]
    DateTimeError,
    #[error("Internal runtime bins are malformed")]
    MalformedRuntimeData,
    #[error("Baseline data must be non empty")]
    EmptyBaselineData,
    #[error("NaN values are not supported")]
    NaNValueError,
    #[error("Unsupported drift type")]
    UnsupportedDriftType,
    #[error("Operation not supported in current drift mode")]
    UnsupportedOperation,
    #[error("Configuration not supported in current drift mode")]
    UnsupportedConfig,
    #[error("IO error using disk backend: {0:?}")]
    IOError(std::io::Error),
    #[error("No entry found")]
    NoEntryFound,
}

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum DriftExportError {
    #[error("Data shapes in baseline are invalid")]
    InvalidDataShape,
    #[error("Invalid drift mode")]
    InvalidDriftMode,
    #[error("Unable to deserialize export: {0:?}")]
    DeserializationError(serde_json::Error),
    #[error("Could not read export file")]
    IOError(std::io::Error),
}

impl From<serde_json::Error> for DriftExportError {
    fn from(err: serde_json::Error) -> DriftExportError {
        DriftExportError::DeserializationError(err)
    }
}

impl From<std::io::Error> for DriftExportError {
    fn from(err: std::io::Error) -> DriftExportError {
        DriftExportError::IOError(err)
    }
}
