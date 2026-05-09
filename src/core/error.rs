use thiserror::Error;

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
