use super::distribution::QuantileType;
use num_traits::Float;
use std::hash::Hash;

#[derive(Debug, PartialEq, Clone)]
pub struct ContinuousBinEdges<T: Float> {
    bin_edges: Vec<T>,
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
         *  distribution that fall outsde the bounds of the baseline distribution.
         *  - Bin/quantile size will have its "step" size determined by evenly diving the difference
         *  between the max and min of the distribution and dividing by the number of bins - 2.
         *  - A value is assigned to a particular quantile if left <= value < right, otherwise it will
         *  be assigned to one of the tail quantile bins.
         *  - Each bin has a constant step size.
         * */
        let mut bin_edges = vec![T::zero(); n_bins - 2];
        let n = dataset.len();
        let n_0 = dataset[0];
        let bin_step = (dataset[n - 1] - n_0) / T::from(n).unwrap();
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

    #[inline]
    fn left_bin_edge(&self) -> T {
        self.bin_edges[0]
    }

    #[inline]
    fn right_bin_edge(&self) -> T {
        // bin_edges.len == n_bins - 2
        self.bin_edges[self.len() - 1]
    }

    pub(crate) fn len(&self) -> usize {
        self.bin_edges.len()
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
        if sample < self.left_bin_edge() {
            return 0_usize;
        }

        if sample > self.right_bin_edge() {
            return self.n_bins - 1;
        }
        // find "pivot" point
        // ie the bin where value >= left and < right
        // this incorrectly misses the left and right edge currently
        // as these values would not created a parition within the edges
        let i = self.bin_edges.partition_point(|edge| sample >= *edge);
        i.clamp(0, self.n_bins - 1)
    }
}

/// Utility wrapper type to encapsulate bin resolution when approximating an entire dataset.
pub struct CategoricalBinEdges<'a, T: Hash + Ord + Clone>(pub &'a ahash::HashMap<T, usize>);

impl<T: Hash + Ord + Clone> CategoricalBinEdges<'_, T> {
    pub fn resolve_bin<Q>(&self, key: &Q) -> usize
    where
        T: std::borrow::Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(idx) = self.0.get(key) {
            *idx
        } else {
            self.n_bins() - 1
        }
    }

    pub fn resolve_bin_opt<Q>(&self, key_opt: &Option<Q>) -> Option<usize>
    where
        T: std::borrow::Borrow<Q>,
        Q: Hash + Eq,
    {
        let Some(example) = key_opt else { return None };
        if let Some(idx) = self.0.get(example) {
            Some(*idx)
        } else {
            Some(self.n_bins() - 1)
        }
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.0.len() + 1
    }
}
