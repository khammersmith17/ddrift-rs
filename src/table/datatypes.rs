#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum DriftDataType {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
    LargeUtf8,
    Boolean,
}

#[cfg(feature = "arrow")]
impl From<&arrow::datatypes::DataType> for DriftDataType {
    fn from(dt: &arrow::datatypes::DataType) -> Self {
        use arrow::datatypes::DataType;
        match dt {
            DataType::Float32 => Self::Float32,
            DataType::Float64 => Self::Float64,
            DataType::Int8 => Self::Int8,
            DataType::Int16 => Self::Int16,
            DataType::Int32 => Self::Int32,
            DataType::Int64 => Self::Int64,
            DataType::UInt8 => Self::UInt8,
            DataType::UInt16 => Self::UInt16,
            DataType::UInt32 => Self::UInt32,
            DataType::UInt64 => Self::UInt64,
            DataType::Utf8 => Self::Utf8,
            DataType::LargeUtf8 => Self::LargeUtf8,
            DataType::Boolean => Self::Boolean,
            DataType::Dictionary(_, inner) => match inner.as_ref() {
                DataType::Utf8 => Self::Utf8,
                DataType::LargeUtf8 => Self::LargeUtf8,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "arrow")]
impl From<arrow::datatypes::DataType> for DriftDataType {
    fn from(dt: arrow::datatypes::DataType) -> Self {
        Self::from(&dt)
    }
}
