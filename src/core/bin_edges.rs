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

#[cfg(test)]
mod continuous_test {
    use super::*;

    fn define_bins() -> ContinuousBinEdges<f32> {
        let bin_edges: Vec<f32> = vec![0.25, 0.50, 0.75];
        let n_bins = 4_usize;

        ContinuousBinEdges { bin_edges, n_bins }
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
    fn below_min_resolves_to_first_bin() {
        let bins = define_bins();
        assert_eq!(0_usize, bins.resolve_bin(-1.0));
        assert_eq!(0_usize, bins.resolve_bin(f32::NEG_INFINITY));
    }

    #[test]
    fn above_max_resolves_to_last_bin() {
        let bins = define_bins();
        assert_eq!(3_usize, bins.resolve_bin(2.0));
        assert_eq!(3_usize, bins.resolve_bin(f32::INFINITY));
    }

    #[test]
    fn dataset_constructor_bin_count() {
        // sorted dataset spanning [0, 1] with 4 bins
        let dataset: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        let bins = ContinuousBinEdges::new_from_dataset_with_bin_count(&dataset, 4);
        assert_eq!(4, bins.n_bins());
    }

    #[test]
    fn dataset_constructor_all_bins_reachable() {
        use std::collections::HashSet;
        let baseline_dataset: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        let bins = ContinuousBinEdges::new_from_dataset_with_bin_count(&baseline_dataset, 4);
        let runtime_dataset: Vec<f32> = (-1..=100).map(|i| i as f32 / 100.0).collect();
        let resolved: HashSet<usize> = runtime_dataset
            .iter()
            .map(|&v| bins.resolve_bin(v))
            .collect();
        dbg!(&resolved);
        assert_eq!(4, resolved.len());
    }
}

#[cfg(test)]
mod categorical_test {
    use super::*;
    use ahash::HashMap;

    const OTHER_BUCKET: usize = 4;

    fn define_map() -> HashMap<String, usize> {
        let mut keys = vec!["Four".into(), "One".into(), "Three".into(), "Two".into()];
        keys.sort();
        let map: HashMap<String, usize> =
            keys.into_iter().enumerate().map(|(i, k)| (k, i)).collect();
        // Four, One, Three, Two
        map
    }

    fn define_bins() -> CategoricalBinEdges<String> {
        CategoricalBinEdges::new(define_map())
    }

    #[test]
    fn other_bucket() {
        let bins = define_bins();

        assert_eq!(OTHER_BUCKET, bins.resolve_bin("random"));
    }

    #[test]
    fn categorical_resolve() {
        let bins = define_bins();

        assert_eq!(0_usize, bins.resolve_bin("Four"));
        assert_eq!(1_usize, bins.resolve_bin("One"));
        assert_eq!(2_usize, bins.resolve_bin("Three"));
        assert_eq!(3_usize, bins.resolve_bin("Two"));
    }

    #[test]
    fn lookup_is_case_sensitive() {
        let bins = define_bins();
        // lowercase variants are not in the baseline, should fall to other bucket
        assert_eq!(OTHER_BUCKET, bins.resolve_bin("four"));
        assert_eq!(OTHER_BUCKET, bins.resolve_bin("one"));
    }

    #[test]
    fn multiple_unknown_keys_all_go_to_other_bucket() {
        let bins = define_bins();
        for key in &["x", "y", "z", "unknown", "FOUR"] {
            assert_eq!(OTHER_BUCKET, bins.resolve_bin(*key));
        }
    }

    #[test]
    fn single_category_baseline() {
        // n_bins should be 2: one known bin + one other bin
        let map: HashMap<String, usize> = [("only".to_string(), 0)].into_iter().collect();
        let bins = CategoricalBinEdges::new(map);
        assert_eq!(2, bins.n_bins());
        assert_eq!(0_usize, bins.resolve_bin("only"));
        assert_eq!(1_usize, bins.resolve_bin("other"));
    }
}
