pub mod baseline;
pub(crate) mod constants;
pub mod core;
pub mod drift;
pub mod export;
use core::{
    bin_edges::ContinuousBinEdges,
    distribution::QuantileType,
    drift_metrics::{CategoricalDriftType, ContinuousDriftType},
    error::DriftError,
};
use drift::{
    DriftComputation,
    discrete::{categorical::CategoricalDataDrift, continuous::ContinuousDataDrift},
};
use num_traits::Float;
use std::hash::Hash;

pub fn compute_approximate_dataset_continuous<T: Float + Send + Sync>(
    dataset: &[T],
    quantile_type: Option<QuantileType>,
) -> Vec<f64> {
    core::compute_dataset_from_bins_continuous(
        dataset,
        &ContinuousBinEdges::new_from_dataset_with_quantile_type(
            dataset,
            quantile_type.unwrap_or_default(),
        ),
    )
}

pub fn compute_approximate_dataset_continuous_with_bin_edges<T: Float + Send + Sync>(
    dataset: &[T],
    bin_edges: &ContinuousBinEdges<T>,
) -> Vec<f64> {
    core::compute_dataset_from_bins_continuous(dataset, bin_edges)
}

pub fn compute_drift_continuous_distribution<T: Float + Send + Sync>(
    baseline_distribution: &[T],
    candidate_distribution: &[T],
    drift_metrics: &[ContinuousDriftType],
    quantile_type: Option<QuantileType>,
) -> Result<Vec<DriftComputation<ContinuousDriftType>>, DriftError> {
    let mut drift_container =
        ContinuousDataDrift::new_from_baseline(quantile_type, baseline_distribution)?;
    let drift_res =
        drift_container.compute_drift_multiple_criteria(candidate_distribution, drift_metrics)?;
    Ok(drift_res)
}

pub fn compute_drift_categorical_distribution<T: Hash + Ord + Clone>(
    baseline_distribution: &[T],
    candidate_distribution: &[T],
    drift_metrics: &[CategoricalDriftType],
) -> Result<Vec<DriftComputation<CategoricalDriftType>>, DriftError> {
    let mut drift_container = CategoricalDataDrift::new(baseline_distribution)?;
    let drift_res =
        drift_container.compute_drift_multiple_criteria(candidate_distribution, drift_metrics)?;
    Ok(drift_res)
}

/// Performs the same computation as [`compute_drift_categorical_distribution`], but if the type is
/// Sync, it can be optimized to perform the bin assignment across many cores. The is method
/// provides that functionaliy.
pub fn compute_drift_categorical_distribution_par<T: Hash + Ord + Clone + Sync>(
    baseline_distribution: &[T],
    candidate_distribution: &[T],
    drift_metrics: &[CategoricalDriftType],
) -> Result<Vec<DriftComputation<CategoricalDriftType>>, DriftError> {
    let mut drift_container = CategoricalDataDrift::new(baseline_distribution)?;
    let drift_res = drift_container
        .compute_drift_multiple_criteria_par(candidate_distribution, drift_metrics)?;
    Ok(drift_res)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- compute_drift_continuous_distribution ---

    #[test]
    fn continuous_no_drift_returns_near_zero() {
        let baseline = [1.0, 2.0, 3.0, 4.0, 5.0];
        let candidate = [1.0, 2.0, 3.0, 4.0, 5.0];
        let metrics = [ContinuousDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_continuous_distribution(&baseline, &candidate, &metrics, None).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].drift_magnitude.abs() < 1e-9);
    }

    #[test]
    fn continuous_shifted_distribution_detects_drift() {
        let baseline = [1.0, 2.0, 3.0, 4.0, 5.0];
        let candidate = [20.0, 21.0, 22.0, 23.0, 24.0];
        let metrics = [ContinuousDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_continuous_distribution(&baseline, &candidate, &metrics, None).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].drift_magnitude > 0.5);
    }

    #[test]
    fn continuous_multiple_metrics_returns_one_value_per_metric() {
        let baseline = [1.0, 2.0, 3.0, 4.0, 5.0];
        let candidate = [1.0, 2.0, 3.0, 4.0, 5.0];
        let metrics = [
            ContinuousDriftType::PopulationStabilityIndex,
            ContinuousDriftType::JensenShannon,
            ContinuousDriftType::KullbackLeibler,
            ContinuousDriftType::WassersteinDistance,
        ];

        let result =
            compute_drift_continuous_distribution(&baseline, &candidate, &metrics, None).unwrap();

        assert_eq!(result.len(), 4);
        for score in &result {
            assert!(score.drift_magnitude.abs() < 1e-9);
        }
    }

    #[test]
    fn continuous_explicit_quantile_type_is_accepted() {
        let baseline = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let candidate = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let metrics = [ContinuousDriftType::JensenShannon];

        let result = compute_drift_continuous_distribution(
            &baseline,
            &candidate,
            &metrics,
            Some(QuantileType::Sturges),
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].drift_magnitude.abs() < 1e-9);
    }

    #[test]
    fn continuous_empty_baseline_returns_error() {
        let baseline: &[f64] = &[];
        let candidate = [1.0, 2.0];
        let metrics = [ContinuousDriftType::PopulationStabilityIndex];

        let result = compute_drift_continuous_distribution(baseline, &candidate, &metrics, None);

        assert!(result.is_err());
    }

    // --- compute_drift_categorical_distribution ---

    #[test]
    fn categorical_no_drift_returns_near_zero() {
        let baseline = ["a", "b", "a", "c", "b", "a"];
        let candidate = ["a", "b", "a", "c", "b", "a"];
        let metrics = [CategoricalDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].drift_magnitude.abs() < 1e-9);
    }

    #[test]
    fn categorical_shifted_distribution_detects_drift() {
        let baseline = ["a", "a", "a", "a", "b"];
        let candidate = ["b", "b", "b", "b", "a"];
        let metrics = [CategoricalDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].drift_magnitude > 0.1);
    }

    #[test]
    fn categorical_multiple_metrics_returns_one_value_per_metric() {
        let baseline = ["x", "y", "x", "z", "y"];
        let candidate = ["x", "y", "x", "z", "y"];
        let metrics = [
            CategoricalDriftType::PopulationStabilityIndex,
            CategoricalDriftType::JensenShannon,
            CategoricalDriftType::KullbackLeibler,
            CategoricalDriftType::WassersteinDistance,
        ];

        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 4);
        for score in &result {
            assert!(score.drift_magnitude.abs() < 1e-9);
        }
    }

    #[test]
    fn categorical_unseen_label_in_candidate_is_handled() {
        let baseline = ["a", "b", "a", "b"];
        let candidate = ["a", "b", "c", "d"];
        let metrics = [CategoricalDriftType::JensenShannon];

        // unseen labels should be bucketed into the overflow bin, not panic
        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].drift_magnitude >= 0.0);
    }

    #[test]
    fn categorical_integer_labels_are_supported() {
        let baseline = [1i32, 2, 1, 3, 2, 1];
        let candidate = [1i32, 2, 1, 3, 2, 1];
        let metrics = [CategoricalDriftType::PopulationStabilityIndex];

        let result =
            compute_drift_categorical_distribution(&baseline, &candidate, &metrics).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].drift_magnitude.abs() < 1e-9);
    }

    #[test]
    fn categorical_empty_baseline_returns_error() {
        let baseline: &[&str] = &[];
        let candidate = ["a", "b"];
        let metrics = [CategoricalDriftType::PopulationStabilityIndex];

        let result = compute_drift_categorical_distribution(baseline, &candidate, &metrics);

        assert!(result.is_err());
    }
}
