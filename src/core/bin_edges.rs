use super::distribution::QuantileType;
use num_traits::Float;
use std::hash::Hash;

#[derive(Debug, PartialEq, Clone)]
pub struct ContinuousBinEdges<T: Float> {
    pub(crate) bin_edges: Vec<T>,
    n_bins: usize,
}

impl<T: Float> ContinuousBinEdges<T> {
    pub fn new_from_parts(bin_edges: Vec<T>) -> ContinuousBinEdges<T> {
        let n_bins = bin_edges.len() + 2;
        ContinuousBinEdges { bin_edges, n_bins }
    }
    /// Assumes data is sorted.
    pub fn new_from_dataset_with_quantile_type(
        dataset: &[T],
        quantile_type: QuantileType,
    ) -> ContinuousBinEdges<T> {
        let n_bins = quantile_type.compute_num_bins(dataset);
        ContinuousBinEdges::new_from_dataset_with_bin_count(dataset, n_bins)
    }

    /// Assumes data is sorted.
    pub fn new_from_dataset_with_bin_count(dataset: &[T], n_bins: usize) -> ContinuousBinEdges<T> {
        /*
         * - Bin edges will be of size num_bins - 2.
         * - The outer bins, or tail bins in the distribution will be reserved for values observed in the
         *  distribution that fall outsde the mix/max bounds of the baseline distribution.
         *  - Bin/quantile size will have its "step" size determined by evenly diving the difference
         *  between the max and min of the distribution and dividing by the number of bins - 2.
         *  - A value is assigned to a particular quantile if left <= value < right, otherwise it will
         *  be assigned to one of the tail quantile bins.
         *  - Each bin has a constant step size.
         * */
        let mut bin_edges = vec![T::zero(); n_bins - 1];
        let n = dataset.len();
        let n_0 = dataset[0];
        let bin_step = (dataset[n - 1] - n_0) / T::from(n_bins).unwrap();
        let mut edge_value = n_0;

        for edge in bin_edges.iter_mut() {
            *edge = edge_value;
            edge_value = edge_value + bin_step;
        }

        ContinuousBinEdges { bin_edges, n_bins }
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.n_bins
    }

    pub(crate) fn export_edges(&self) -> Vec<T> {
        self.bin_edges.clone()
    }

    pub(crate) fn take_edges(self) -> Vec<T> {
        let Self { bin_edges, .. } = self;
        bin_edges
    }

    #[inline]
    pub fn resolve_bin(&self, sample: T) -> usize {
        // Values are assigned to bin with condition [i, i + 1).
        let i = self.bin_edges.partition_point(|edge| sample >= *edge);
        i.clamp(0, self.n_bins - 1)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct NullableContinuousBinEdges<T: Float> {
    pub(crate) inner: ContinuousBinEdges<T>,
}

impl<T: Float> NullableContinuousBinEdges<T> {
    pub(crate) fn new(inner: ContinuousBinEdges<T>) -> NullableContinuousBinEdges<T> {
        Self { inner }
    }

    pub(crate) fn inner_ref(&self) -> &ContinuousBinEdges<T> {
        &self.inner
    }

    #[inline]
    pub fn resolve_bin(&self, sample: Option<T>) -> Option<usize> {
        if let Some(concrete_sample) = sample {
            Some(self.inner.resolve_bin(concrete_sample))
        } else {
            None
        }
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.inner.n_bins
    }

    pub(crate) fn export_edges(&self) -> Vec<T> {
        self.inner.bin_edges.clone()
    }

    pub(crate) fn take_edges(self) -> Vec<T> {
        self.inner.take_edges()
    }
}

/// Utility wrapper type to encapsulate bin resolution when approximating an entire dataset.
#[derive(Clone, Debug)]
pub struct CategoricalBinEdges<T: Hash + Ord + Clone>(pub ahash::HashMap<T, usize>);

impl<T: Hash + Ord + Clone> CategoricalBinEdges<T> {
    pub fn new(idx_map: ahash::HashMap<T, usize>) -> CategoricalBinEdges<T> {
        Self(idx_map)
    }

    pub fn inner_ref(&self) -> &ahash::HashMap<T, usize> {
        &self.0
    }

    #[inline]
    pub fn resolve_bin<Q>(&self, key: &Q) -> usize
    where
        T: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        *self.0.get(key).unwrap_or(&(self.n_bins() - 1))
    }

    #[inline]
    pub(crate) fn n_bins(&self) -> usize {
        self.0.len() + 1
    }
}

/// Utility wrapper type to encapsulate bin resolution when approximating an entire dataset.
#[derive(Clone, Debug)]
pub struct NullableCategoricalBinEdges<T: Hash + Ord + Clone>(pub ahash::HashMap<T, usize>);

impl<T: Hash + Ord + Clone> NullableCategoricalBinEdges<T> {
    pub fn new(idx_map: ahash::HashMap<T, usize>) -> NullableCategoricalBinEdges<T> {
        Self(idx_map)
    }

    pub fn inner_ref(&self) -> &ahash::HashMap<T, usize> {
        &self.0
    }

    #[inline]
    pub fn resolve_bin<Q>(&self, key_opt: &Option<Q>) -> Option<usize>
    where
        T: std::borrow::Borrow<Q>,
        Q: Hash + Eq,
    {
        let Some(example) = key_opt else { return None };
        Some(*self.0.get(example).unwrap_or(&(self.n_bins() - 1)))
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.0.len() + 1
    }
}

#[cfg(test)]
mod continuous_test {
    use super::*;

    fn define_bins() -> ContinuousBinEdges<f32> {
        let bin_edges: Vec<f32> = vec![0.25, 0.50, 0.75];
        let n_bins = 4_usize;

        ContinuousBinEdges { bin_edges, n_bins }
    }

    fn define_nullable_bins() -> NullableContinuousBinEdges<f32> {
        let inner = define_bins();

        NullableContinuousBinEdges { inner }
    }

    #[test]
    fn continuous_resolution_non_edge() {
        let bins = define_bins();

        assert_eq!(0_usize, bins.resolve_bin(0.1));
        assert_eq!(1_usize, bins.resolve_bin(0.3));
        assert_eq!(2_usize, bins.resolve_bin(0.55));
        assert_eq!(3_usize, bins.resolve_bin(1.0));
    }

    #[test]
    fn continuous_resolution_edge() {
        let bins = define_bins();

        assert_eq!(0_usize, bins.resolve_bin(0.1));
        assert_eq!(1_usize, bins.resolve_bin(0.25));
        assert_eq!(2_usize, bins.resolve_bin(0.50));
        assert_eq!(3_usize, bins.resolve_bin(0.75));
    }

    #[test]
    fn all_bins_resolved() {
        use std::collections::HashSet;
        let mut resolved_set: HashSet<usize> = HashSet::new();
        let bins = define_bins();
        let mut value = 0.1;
        let step = 0.01;

        while value < 1_f32 {
            resolved_set.insert(bins.resolve_bin(value));
            value += step;
        }

        assert_eq!(4_usize, resolved_set.len());
    }

    #[test]
    fn continuous_nullable() {
        let nullable_bins = define_nullable_bins();

        assert_eq!(Some(0_usize), nullable_bins.resolve_bin(Some(0.1)));
        assert_eq!(Some(1_usize), nullable_bins.resolve_bin(Some(0.3)));
        assert_eq!(Some(2_usize), nullable_bins.resolve_bin(Some(0.55)));
        assert_eq!(Some(3_usize), nullable_bins.resolve_bin(Some(1.0)));
        assert_eq!(None, nullable_bins.resolve_bin(None));
    }
}
