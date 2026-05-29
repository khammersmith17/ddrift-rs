use super::{
    categorical::NullableBaselineCategoricalBins, continuous::NullableBaselineContinuousBins,
};
use crate::core::distribution::QuantileType;
use ahash::HashMap;
use arrow::{
    array::{self, Array},
    datatypes::DataType,
};

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

pub struct ArrowColumnBaseline {
    arrow_type: DataType,
    container: ArrowBaselineContainer,
}

impl ArrowColumnBaseline {
    pub fn from_array(
        array: Box<dyn Array>,
        quantile_type: Option<QuantileType>,
    ) -> Result<ArrowColumnBaseline, Box<dyn std::error::Error>> {
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
            DataType::Int8 => todo!(),
            DataType::Int16 => todo!(),
            DataType::Int32 => todo!(),
            DataType::Int64 => todo!(),
            DataType::UInt8 => todo!(),
            DataType::UInt16 => todo!(),
            DataType::UInt32 => todo!(),
            DataType::UInt64 => todo!(),
            DataType::Utf8 | DataType::LargeUtf8 => todo!(),
            DataType::Dictionary(_, value_type)
                if matches!(value_type.as_ref(), DataType::Utf8 | DataType::LargeUtf8) =>
            {
                todo!()
            }
            DataType::Boolean => todo!(),

            other => {
                return Err(format!("unsupported Arrow type for drift baseline: {other}").into());
            }
        };
        #[allow(unreachable_code)]
        Ok(ArrowColumnBaseline {
            arrow_type,
            container,
        })
    }
}

pub struct ArrowTableBaseline {
    pub table: HashMap<String, ArrowColumnBaseline>,
}
