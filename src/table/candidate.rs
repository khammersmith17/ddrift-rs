use super::{
    ColumnTypeClass,
    datatypes::DriftDataType,
    schema_view::{SchemaValidationResult, validate_schema},
};
use crate::{
    baseline::{
        categorical::NullableBaselineCategoricalBins,
        continuous::NullableBaselineContinuousBins,
        table::{BaselineColumn, BaselineContainer, BaselineTable},
    },
    core::{
        dataset_view::candidate::{
            NullableCategoricalCandidateView, NullableContinuousCandidateView,
        },
        error::{DriftError, DriftTableError},
    },
    drift::{NullDriftActorComponents, NullableDriftActor},
};
use ahash::{HashMap, HashMapExt};
use arrow::{
    array::{Array, ArrayRef},
    datatypes::DataType,
    record_batch::RecordBatch,
};
use std::collections::hash_map::Iter;
use std::sync::Arc;

fn canidate_arrow_array_string_insert_dispatch<'a>(
    view: &mut NullableCategoricalCandidateView<'a, String>,
    array: ArrayRef,
) -> Result<(), DriftError> {
    use super::slice_impl::{StringSlice32, StringSlice64};
    match array.data_type() {
        DataType::Utf8 | DataType::Dictionary(_, _) => {
            let typed = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let slice = StringSlice32::from_array(typed);
            view.insert_string_slice(&slice)?;
        }
        DataType::LargeUtf8 => {
            let typed = array
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .unwrap();
            let slice = StringSlice64::from_array(typed);
            view.insert_string_slice(&slice)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn candidate_arrow_array_string_dispatch<'a>(
    array: ArrayRef,
    bin_edges: &'a crate::core::bin_edges::CategoricalBinEdges<String>,
) -> Result<NullableCategoricalCandidateView<'a, String>, crate::core::error::DriftError> {
    use super::slice_impl::{StringSlice32, StringSlice64};

    // Match on inner string type to determine concrete slice type.
    // Dispatch follows the same route given either slice dispatch.
    match array.data_type() {
        DataType::Utf8 | DataType::Dictionary(_, _) => {
            let typed = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let slice = StringSlice32::from_array(typed);
            NullableCategoricalCandidateView::from_string_slice(&slice, bin_edges)
        }
        DataType::LargeUtf8 => {
            let typed = array
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .unwrap();
            let slice = StringSlice64::from_array(typed);
            NullableCategoricalCandidateView::from_string_slice(&slice, bin_edges)
        }
        // Guaranteed to be one of these types at this point.
        _ => unreachable!(),
    }
}

pub enum CandidateContainer<'a> {
    FloatingPoint32(NullableContinuousCandidateView<'a, f32>),
    FloatingPoint64(NullableContinuousCandidateView<'a, f64>),
    Integer64(NullableCategoricalCandidateView<'a, i64>),
    Integer32(NullableCategoricalCandidateView<'a, i32>),
    Integer16(NullableCategoricalCandidateView<'a, i16>),
    Integer8(NullableCategoricalCandidateView<'a, i8>),
    UnsignedInteger64(NullableCategoricalCandidateView<'a, u64>),
    UnsignedInteger32(NullableCategoricalCandidateView<'a, u32>),
    UnsignedInteger16(NullableCategoricalCandidateView<'a, u16>),
    UnsignedInteger8(NullableCategoricalCandidateView<'a, u8>),
    String(NullableCategoricalCandidateView<'a, String>),
    Boolean(NullableCategoricalCandidateView<'a, bool>),
}

impl<'a> CandidateContainer<'a> {
    fn computation_view(&'a self) -> NullDriftActorComponents<'a> {
        match self {
            Self::FloatingPoint32(view) => view.nullable_components(),
            Self::FloatingPoint64(view) => view.nullable_components(),
            Self::Integer64(view) => view.nullable_components(),
            Self::Integer32(view) => view.nullable_components(),
            Self::Integer16(view) => view.nullable_components(),
            Self::Integer8(view) => view.nullable_components(),
            Self::UnsignedInteger64(view) => view.nullable_components(),
            Self::UnsignedInteger32(view) => view.nullable_components(),
            Self::UnsignedInteger16(view) => view.nullable_components(),
            Self::UnsignedInteger8(view) => view.nullable_components(),
            Self::String(view) => view.nullable_components(),
            Self::Boolean(view) => view.nullable_components(),
        }
    }

    fn type_class(&self) -> ColumnTypeClass {
        if matches!(self, Self::FloatingPoint32(_) | Self::FloatingPoint64(_)) {
            ColumnTypeClass::Continuous
        } else {
            ColumnTypeClass::Categorical
        }
    }
}

pub struct CandidateColumn<'a> {
    pub datatype: DriftDataType,
    pub container: CandidateContainer<'a>,
}

impl<'a> CandidateColumn<'a> {
    pub(super) fn new_empty_from_baseline(baseline: &'a BaselineColumn) -> CandidateColumn<'a> {
        let &BaselineColumn {
            datatype: ref baseline_type,
            container: ref baseline_container,
        } = baseline;
        let container = match baseline_container {
            BaselineContainer::FloatingPoint32(bl_bins) => {
                let &NullableBaselineContinuousBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableContinuousCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::FloatingPoint32(inner)
            }
            BaselineContainer::FloatingPoint64(bl_bins) => {
                let &NullableBaselineContinuousBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableContinuousCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::FloatingPoint64(inner)
            }
            BaselineContainer::Integer8(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::Integer8(inner)
            }
            BaselineContainer::Integer16(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::Integer16(inner)
            }
            BaselineContainer::Integer32(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::Integer32(inner)
            }
            BaselineContainer::Integer64(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::Integer64(inner)
            }
            BaselineContainer::UnsignedInteger8(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::UnsignedInteger8(inner)
            }
            BaselineContainer::UnsignedInteger16(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::UnsignedInteger16(inner)
            }
            BaselineContainer::UnsignedInteger32(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::UnsignedInteger32(inner)
            }
            BaselineContainer::UnsignedInteger64(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::UnsignedInteger64(inner)
            }
            BaselineContainer::String(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::String(inner)
            }
            BaselineContainer::Boolean(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = NullableCategoricalCandidateView::empty_from_edges(bin_edges);
                CandidateContainer::Boolean(inner)
            }
        };
        CandidateColumn {
            datatype: *baseline_type,
            container,
        }
    }
    // Caller must guarantee schema parity between baseline and array before calling this.
    pub(super) fn from_baseline_and_array(
        baseline: &'a BaselineColumn,
        array: ArrayRef,
    ) -> Result<CandidateColumn<'a>, DriftTableError> {
        let &BaselineColumn {
            datatype: ref baseline_type,
            container: ref baseline_container,
        } = baseline;

        let candidate_type = array.data_type().into();
        debug_assert_eq!(baseline_type, &candidate_type);

        let container = match baseline_container {
            BaselineContainer::FloatingPoint32(bl_bins) => {
                let &NullableBaselineContinuousBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .unwrap();
                let inner = NullableContinuousCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::FloatingPoint32(inner)
            }
            BaselineContainer::FloatingPoint64(bl_bins) => {
                let &NullableBaselineContinuousBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                let inner = NullableContinuousCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::FloatingPoint64(inner)
            }
            BaselineContainer::Integer8(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int8Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::Integer8(inner)
            }
            BaselineContainer::Integer16(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int16Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::Integer16(inner)
            }
            BaselineContainer::Integer32(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::Integer32(inner)
            }
            BaselineContainer::Integer64(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::Integer64(inner)
            }
            BaselineContainer::UnsignedInteger8(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt8Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::UnsignedInteger8(inner)
            }
            BaselineContainer::UnsignedInteger16(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt16Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::UnsignedInteger16(inner)
            }
            BaselineContainer::UnsignedInteger32(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::UnsignedInteger32(inner)
            }
            BaselineContainer::UnsignedInteger64(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                CandidateContainer::UnsignedInteger64(inner)
            }
            BaselineContainer::String(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = candidate_arrow_array_string_dispatch(array.clone(), bin_edges)?;
                CandidateContainer::String(inner)
            }
            BaselineContainer::Boolean(bl_bins) => {
                use super::slice_impl::BooleanSlice;
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                let slice = BooleanSlice::from_array(&typed_array);
                let inner = NullableCategoricalCandidateView::from_bool_slice(&slice, bin_edges)?;
                CandidateContainer::Boolean(inner)
            }
        };
        Ok(CandidateColumn {
            datatype: candidate_type,
            container,
        })
    }

    pub(super) fn insert(&mut self, array: ArrayRef) -> Result<(), DriftTableError> {
        match &mut self.container {
            CandidateContainer::FloatingPoint32(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::FloatingPoint64(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::Integer8(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int8Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::Integer16(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int16Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::Integer32(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::Integer64(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::UnsignedInteger8(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt8Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::UnsignedInteger16(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt16Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::UnsignedInteger32(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::UnsignedInteger64(view) => {
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .unwrap();
                view.insert_arrow_array(typed.values(), typed.nulls())?;
            }
            CandidateContainer::String(view) => {
                canidate_arrow_array_string_insert_dispatch(view, array)?;
            }
            CandidateContainer::Boolean(view) => {
                use super::slice_impl::BooleanSlice;
                let typed = array
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                let slice = BooleanSlice::from_array(typed);
                view.insert_bool_slice(&slice)?;
            }
        }
        Ok(())
    }

    pub(crate) fn computation_view(&'a self) -> NullDriftActorComponents<'a> {
        self.container.computation_view()
    }

    pub(crate) fn type_class(&self) -> ColumnTypeClass {
        self.container.type_class()
    }
}

pub struct CandidateTable<'a> {
    pub table: HashMap<String, CandidateColumn<'a>>,
}

/*
* All entries start with a validation.
* This provides three benefits:
*   First, we know the schema is correct and has parity.
*   Second, we know there are not unsupported types in the candidate table.
*   Third, we know downstream there are no structural error cases.
* */

impl<'a> CandidateTable<'a> {
    pub fn new_from_baseline_schema(baseline_table: &'a BaselineTable) -> CandidateTable<'a> {
        let mut table = HashMap::with_capacity(baseline_table.len());
        for (name, column) in baseline_table.iter() {
            table.insert(
                name.clone(),
                CandidateColumn::new_empty_from_baseline(column),
            );
        }

        CandidateTable { table }
    }

    pub fn from_arrow_record_batch(
        baseline_table: &'a BaselineTable,
        record_batch: Arc<RecordBatch>,
    ) -> Result<CandidateTable<'a>, DriftTableError> {
        if let SchemaValidationResult::Invalid(diff) =
            validate_schema(baseline_table, record_batch.as_ref())
        {
            return Err(DriftTableError::SchemaError(diff));
        }

        let mut table = HashMap::with_capacity(baseline_table.table.len());
        for (name, baseline_col) in &baseline_table.table {
            // SAFETY: schema validated above ensures column exists with matching type.
            let array = record_batch.column_by_name(name).unwrap().clone();
            table.insert(
                name.clone(),
                CandidateColumn::from_baseline_and_array(baseline_col, array)?,
            );
        }
        Ok(CandidateTable { table })
    }

    pub fn insert_record_batch(
        &mut self,
        record_batch: Arc<RecordBatch>,
    ) -> Result<(), DriftTableError> {
        // Get a temporary exclusive reference from the mut reference.
        if let SchemaValidationResult::Invalid(diff) =
            validate_schema(&(*self), record_batch.as_ref())
        {
            return Err(DriftTableError::SchemaError(diff));
        }
        for (name, column) in self.table.iter_mut() {
            // SAFETY: schema validated above ensures column exists with matching type.
            let array = record_batch.column_by_name(name).unwrap().clone();
            column.insert(array)?;
        }
        Ok(())
    }

    pub fn get_column(&self, column_name: &str) -> Option<&CandidateColumn<'a>> {
        self.table.get(column_name)
    }

    pub(crate) fn iter(&self) -> Iter<String, CandidateColumn<'a>> {
        self.table.iter()
    }
}
