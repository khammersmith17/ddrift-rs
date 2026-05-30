use super::{DriftCheck, DriftMeasurementEvaluation, DriftThresholdEvaluation, NullableDriftCheck};
use crate::constants::drift_thresholds as defaults;
use crate::core::drift_metrics::ContinuousDriftMeasurement;
use crate::drift::DriftComputation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ContinuousDriftContract {
    pub jensen_shannon: f64,
    pub population_stability_index: f64,
    pub wasserstein_distance: f64,
    pub kullback_leibler: f64,
    pub kolmogorov_smirnov: f64,
    pub hellinger: f64,
}

impl ContinuousDriftContract {
    /// Returns `true` if every result whose metric has a configured threshold is at or below that
    /// threshold. Results for metrics with no threshold always pass.
    pub fn check(
        &self,
        results: &[DriftComputation<ContinuousDriftMeasurement>],
    ) -> DriftCheck<ContinuousDriftMeasurement> {
        let eval = results
            .iter()
            .map(|r| {
                let upper_threshold = match r.drift_type {
                    ContinuousDriftMeasurement::JensenShannon => self.jensen_shannon,
                    ContinuousDriftMeasurement::PopulationStabilityIndex => {
                        self.population_stability_index
                    }
                    ContinuousDriftMeasurement::WassersteinDistance => self.wasserstein_distance,
                    ContinuousDriftMeasurement::KullbackLeibler => self.kullback_leibler,
                    ContinuousDriftMeasurement::KolmogorovSmirnov => self.kolmogorov_smirnov,
                    ContinuousDriftMeasurement::Hellinger => self.hellinger,
                };
                let delta = r.drift_magnitude - upper_threshold;
                let evaluation_result = if delta > 0_f64 {
                    DriftThresholdEvaluation::Failed(delta)
                } else {
                    DriftThresholdEvaluation::Passed
                };
                DriftMeasurementEvaluation {
                    metric: r.drift_type,
                    evaluation_result,
                }
            })
            .collect();
        DriftCheck::new(eval)
    }
}

#[derive(Default)]
pub struct ContinuousDriftContractBuilder {
    jensen_shannon: Option<f64>,
    population_stability_index: Option<f64>,
    wasserstein_distance: Option<f64>,
    kullback_leibler: Option<f64>,
    kolmogorov_smirnov: Option<f64>,
    hellinger: Option<f64>,
}

impl ContinuousDriftContractBuilder {
    pub fn new() -> ContinuousDriftContractBuilder {
        ContinuousDriftContractBuilder::default()
    }

    pub fn with_jensen_shannon(mut self, value: f64) -> ContinuousDriftContractBuilder {
        self.jensen_shannon = Some(value);
        self
    }

    pub fn with_population_stability_index(mut self, value: f64) -> ContinuousDriftContractBuilder {
        self.population_stability_index = Some(value);
        self
    }

    pub fn with_wasserstein_distance(mut self, value: f64) -> ContinuousDriftContractBuilder {
        self.wasserstein_distance = Some(value);
        self
    }

    pub fn with_kullback_leibler(mut self, value: f64) -> ContinuousDriftContractBuilder {
        self.kullback_leibler = Some(value);
        self
    }

    pub fn with_kolmogorov_smirnov(mut self, value: f64) -> ContinuousDriftContractBuilder {
        self.kolmogorov_smirnov = Some(value);
        self
    }

    pub fn with_hellinger(mut self, value: f64) -> ContinuousDriftContractBuilder {
        self.hellinger = Some(value);
        self
    }

    pub fn build(self) -> ContinuousDriftContract {
        ContinuousDriftContract {
            jensen_shannon: self
                .jensen_shannon
                .unwrap_or(defaults::common::JENSEN_SHANNON),
            population_stability_index: self
                .population_stability_index
                .unwrap_or(defaults::common::POPULATION_STABILITY_INDEX),
            wasserstein_distance: self
                .wasserstein_distance
                .unwrap_or(defaults::common::WASSERSTEIN_DISTANCE),
            kullback_leibler: self
                .kullback_leibler
                .unwrap_or(defaults::common::KULLBACK_LEIBLER),
            kolmogorov_smirnov: self
                .kolmogorov_smirnov
                .unwrap_or(defaults::continuous::KOLMOGOROV_SMIRNOV),
            hellinger: self.hellinger.unwrap_or(defaults::common::HELLINGER),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NullableContinuousDriftContract {
    pub jensen_shannon: f64,
    pub population_stability_index: f64,
    pub wasserstein_distance: f64,
    pub kullback_leibler: f64,
    pub kolmogorov_smirnov: f64,
    pub hellinger: f64,
    pub null_percentage: f64,
}

impl NullableContinuousDriftContract {
    /// Returns `true` if the null rate and every drift result with a configured threshold are at
    /// or below their respective thresholds. Metrics and the null rate with no threshold always
    /// pass.
    ///
    /// `null_percentage` and `results` map directly to the fields of
    /// [`NullableDriftComputation`] and [`NullableDriftComputationMulti`].
    ///
    /// [`NullableDriftComputation`]: crate::drift::NullableDriftComputation
    /// [`NullableDriftComputationMulti`]: crate::drift::NullableDriftComputationMulti
    pub fn check(
        &self,
        observed_null_percentage: f64,
        results: &[DriftComputation<ContinuousDriftMeasurement>],
    ) -> NullableDriftCheck<ContinuousDriftMeasurement> {
        let eval = results
            .iter()
            .map(|r| {
                let upper_threshold = match r.drift_type {
                    ContinuousDriftMeasurement::JensenShannon => self.jensen_shannon,
                    ContinuousDriftMeasurement::PopulationStabilityIndex => {
                        self.population_stability_index
                    }
                    ContinuousDriftMeasurement::WassersteinDistance => self.wasserstein_distance,
                    ContinuousDriftMeasurement::KullbackLeibler => self.kullback_leibler,
                    ContinuousDriftMeasurement::KolmogorovSmirnov => self.kolmogorov_smirnov,
                    ContinuousDriftMeasurement::Hellinger => self.hellinger,
                };
                let delta = r.drift_magnitude - upper_threshold;
                let evaluation_result = if delta > 0_f64 {
                    DriftThresholdEvaluation::Failed(delta)
                } else {
                    DriftThresholdEvaluation::Passed
                };
                DriftMeasurementEvaluation {
                    metric: r.drift_type,
                    evaluation_result,
                }
            })
            .collect();
        NullableDriftCheck::new(eval, self.null_percentage, observed_null_percentage)
    }
}

#[derive(Default)]
pub struct NullableContinuousDriftContractBuilder {
    jensen_shannon: Option<f64>,
    population_stability_index: Option<f64>,
    wasserstein_distance: Option<f64>,
    kullback_leibler: Option<f64>,
    kolmogorov_smirnov: Option<f64>,
    hellinger: Option<f64>,
    null_percentage: Option<f64>,
}

impl NullableContinuousDriftContractBuilder {
    pub fn new() -> NullableContinuousDriftContractBuilder {
        NullableContinuousDriftContractBuilder::default()
    }

    pub fn with_jensen_shannon(mut self, value: f64) -> NullableContinuousDriftContractBuilder {
        self.jensen_shannon = Some(value);
        self
    }

    pub fn with_population_stability_index(
        mut self,
        value: f64,
    ) -> NullableContinuousDriftContractBuilder {
        self.population_stability_index = Some(value);
        self
    }

    pub fn with_wasserstein_distance(
        mut self,
        value: f64,
    ) -> NullableContinuousDriftContractBuilder {
        self.wasserstein_distance = Some(value);
        self
    }

    pub fn with_kullback_leibler(mut self, value: f64) -> NullableContinuousDriftContractBuilder {
        self.kullback_leibler = Some(value);
        self
    }

    pub fn with_kolmogorov_smirnov(mut self, value: f64) -> NullableContinuousDriftContractBuilder {
        self.kolmogorov_smirnov = Some(value);
        self
    }

    pub fn with_hellinger(mut self, value: f64) -> NullableContinuousDriftContractBuilder {
        self.hellinger = Some(value);
        self
    }

    pub fn with_null_percentage(mut self, value: f64) -> NullableContinuousDriftContractBuilder {
        self.null_percentage = Some(value);
        self
    }

    pub fn build(self) -> NullableContinuousDriftContract {
        NullableContinuousDriftContract {
            jensen_shannon: self
                .jensen_shannon
                .unwrap_or(defaults::common::JENSEN_SHANNON),
            population_stability_index: self
                .population_stability_index
                .unwrap_or(defaults::common::POPULATION_STABILITY_INDEX),
            wasserstein_distance: self
                .wasserstein_distance
                .unwrap_or(defaults::common::WASSERSTEIN_DISTANCE),
            kullback_leibler: self
                .kullback_leibler
                .unwrap_or(defaults::common::KULLBACK_LEIBLER),
            kolmogorov_smirnov: self
                .kolmogorov_smirnov
                .unwrap_or(defaults::continuous::KOLMOGOROV_SMIRNOV),
            hellinger: self.hellinger.unwrap_or(defaults::common::HELLINGER),
            null_percentage: self
                .null_percentage
                .unwrap_or(defaults::common::NULL_PERCENTAGE),
        }
    }
}
