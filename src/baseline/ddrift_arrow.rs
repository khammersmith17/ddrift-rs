use super::{
    categorical::NullableBaselineCategoricalBins, continuous::NullableBaselineContinuousBins,
};
use crate::core::distribution::QuantileType;
use ahash::{HashMap, HashMapExt};
use arrow::{
    array::{self, Array},
    datatypes::DataType,
    record_batch::RecordBatch,
};
use std::sync::Arc;

// TODO: define concrete errors for these cases.
// Define export and import utilities.

pub enum ArrowBaselineContainer {
    FloatingPoint32(NullableBaselineContinuousBins<f32>),
    FloatingPoint64(NullableBaselineContinuousBins<f64>),
    Integer64(NullableBaselineCategoricalBins<i64>),
    Integer32(NullableBaselineCategoricalBins<i32>),
    Integer16(NullableBaselineCategoricalBins<i16>),
    Integer8(NullableBaselineCategoricalBins<i8>),
    UnsignedInteger64(NullableBaselineCategoricalBins<u64>),
    UnsignedInteger32(NullableBaselineCategoricalBins<u32>),
    UnsignedInteger16(NullableBaselineCategoricalBins<u16>),
    UnsignedInteger8(NullableBaselineCategoricalBins<u8>),
    String(NullableBaselineCategoricalBins<String>),
    Boolean(NullableBaselineCategoricalBins<bool>),
}

pub struct ArrowBaselineColumn {
    pub arrow_type: DataType,
    pub container: ArrowBaselineContainer,
}

impl ArrowBaselineColumn {
    pub fn from_array(
        array: Arc<dyn Array>,
        quantile_type: Option<QuantileType>,
    ) -> Result<ArrowBaselineColumn, Box<dyn std::error::Error>> {
        let arrow_type = array.data_type().clone();
        let container = match &arrow_type {
            DataType::Float32 => {
                let typed_arr = array
                    .as_any()
                    .downcast_ref::<array::Float32Array>()
                    .unwrap();
                let inner: NullableBaselineContinuousBins<f32> =
                    NullableBaselineContinuousBins::from_arrow32(&typed_arr, quantile_type);
                ArrowBaselineContainer::FloatingPoint32(inner)
            }
            DataType::Float64 => {
                let typed_arr = array
                    .as_any()
                    .downcast_ref::<array::Float64Array>()
                    .unwrap();
                let inner: NullableBaselineContinuousBins<f64> =
                    NullableBaselineContinuousBins::from_arrow64(&typed_arr, quantile_type);
                ArrowBaselineContainer::FloatingPoint64(inner)
            }
            DataType::Int8 => {
                let typed = array.as_any().downcast_ref::<array::Int8Array>().unwrap();
                let data: Vec<Option<i8>> = typed.iter().collect();
                ArrowBaselineContainer::Integer8(NullableBaselineCategoricalBins::new(&data)?)
            }
            DataType::Int16 => {
                let typed = array.as_any().downcast_ref::<array::Int16Array>().unwrap();
                let data: Vec<Option<i16>> = typed.iter().collect();
                ArrowBaselineContainer::Integer16(NullableBaselineCategoricalBins::new(&data)?)
            }
            DataType::Int32 => {
                let typed = array.as_any().downcast_ref::<array::Int32Array>().unwrap();
                let data: Vec<Option<i32>> = typed.iter().collect();
                ArrowBaselineContainer::Integer32(NullableBaselineCategoricalBins::new(&data)?)
            }
            DataType::Int64 => {
                let typed = array.as_any().downcast_ref::<array::Int64Array>().unwrap();
                let data: Vec<Option<i64>> = typed.iter().collect();
                ArrowBaselineContainer::Integer64(NullableBaselineCategoricalBins::new(&data)?)
            }
            DataType::UInt8 => {
                let typed = array.as_any().downcast_ref::<array::UInt8Array>().unwrap();
                let data: Vec<Option<u8>> = typed.iter().collect();
                ArrowBaselineContainer::UnsignedInteger8(NullableBaselineCategoricalBins::new(
                    &data,
                )?)
            }
            DataType::UInt16 => {
                let typed = array.as_any().downcast_ref::<array::UInt16Array>().unwrap();
                let data: Vec<Option<u16>> = typed.iter().collect();
                ArrowBaselineContainer::UnsignedInteger16(NullableBaselineCategoricalBins::new(
                    &data,
                )?)
            }
            DataType::UInt32 => {
                let typed = array.as_any().downcast_ref::<array::UInt32Array>().unwrap();
                let data: Vec<Option<u32>> = typed.iter().collect();
                ArrowBaselineContainer::UnsignedInteger32(NullableBaselineCategoricalBins::new(
                    &data,
                )?)
            }
            DataType::UInt64 => {
                let typed = array.as_any().downcast_ref::<array::UInt64Array>().unwrap();
                let data: Vec<Option<u64>> = typed.iter().collect();
                ArrowBaselineContainer::UnsignedInteger64(NullableBaselineCategoricalBins::new(
                    &data,
                )?)
            }
            DataType::Utf8 => {
                let typed_array = array.as_any().downcast_ref::<array::StringArray>().unwrap();
                let baseline_bins =
                    NullableBaselineCategoricalBins::from_string_array(&typed_array)?;

                ArrowBaselineContainer::String(baseline_bins)
            }
            DataType::LargeUtf8 => {
                let typed_array = array
                    .as_any()
                    .downcast_ref::<array::LargeStringArray>()
                    .unwrap();
                let baseline_bins =
                    NullableBaselineCategoricalBins::from_string_array(&typed_array)?;
                ArrowBaselineContainer::String(baseline_bins)
            }
            DataType::Dictionary(_, value_type)
                if matches!(value_type.as_ref(), DataType::Utf8 | DataType::LargeUtf8) =>
            {
                let utf8 = arrow::compute::cast(&*array, &DataType::Utf8)?;
                let typed_array = utf8.as_any().downcast_ref::<array::StringArray>().unwrap();

                let baseline_bins =
                    NullableBaselineCategoricalBins::from_string_array(&typed_array)?;

                ArrowBaselineContainer::String(baseline_bins)
            }
            DataType::Boolean => {
                let typed_array = array
                    .as_any()
                    .downcast_ref::<array::BooleanArray>()
                    .unwrap();

                let baseline_bins =
                    NullableBaselineCategoricalBins::from_boolean_array(&typed_array)?;
                ArrowBaselineContainer::Boolean(baseline_bins)
            }
            other => {
                return Err(format!("unsupported Arrow type for drift baseline: {other}").into());
            }
        };
        Ok(ArrowBaselineColumn {
            arrow_type,
            container,
        })
    }
}

pub struct ArrowBaselineTable {
    pub table: HashMap<String, ArrowBaselineColumn>,
}

impl ArrowBaselineTable {
    pub fn from_record_batch(
        batch: &RecordBatch,
        quantile_types_opt: Option<&HashMap<String, QuantileType>>,
    ) -> Result<ArrowBaselineTable, Box<dyn std::error::Error>> {
        // Have an owned map to reference when the user does not provide.
        let fallback_map = HashMap::new();
        let quantile_types = quantile_types_opt.unwrap_or(&fallback_map);
        let schema = batch.schema();
        let table_fields = schema.fields();
        let mut table = HashMap::with_capacity(table_fields.len());
        for field in table_fields.iter() {
            let name = field.name();
            // SAFETY: Field names are acquired from the schema.
            let column_array = batch.column_by_name(name).unwrap();
            // Clone is on an Arc.
            table.insert(
                name.clone(),
                ArrowBaselineColumn::from_array(
                    column_array.clone(),
                    quantile_types.get(name.as_str()).copied(),
                )?,
            );
        }
        Ok(ArrowBaselineTable { table })
    }
}
