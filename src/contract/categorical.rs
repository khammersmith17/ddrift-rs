use super::{DriftCheck, DriftMeasurementEvaluation, DriftThresholdEvaluation, NullableDriftCheck};
use crate::constants::drift_thresholds as defaults;
use crate::core::drift_metrics::CategoricalDriftMeasurement;
use crate::drift::DriftComputation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CategoricalDriftContract {
    pub jensen_shannon: f64,
    pub population_stability_index: f64,
    pub wasserstein_distance: f64,
    pub kullback_leibler: f64,
    pub chi_squared: f64,
    pub hellinger: f64,
    pub g_test: f64,
}

impl CategoricalDriftContract {
    /// Returns `true` if every result whose metric has a configured threshold is at or below that
    /// threshold. Results for metrics with no threshold always pass.
    pub fn check(
        &self,
        results: &[DriftComputation<CategoricalDriftMeasurement>],
    ) -> DriftCheck<CategoricalDriftMeasurement> {
        let eval = results
            .iter()
            .map(|r| {
                let threshold = match r.drift_type {
                    CategoricalDriftMeasurement::JensenShannon => self.jensen_shannon,
                    CategoricalDriftMeasurement::PopulationStabilityIndex => {
                        self.population_stability_index
                    }
                    CategoricalDriftMeasurement::WassersteinDistance => self.wasserstein_distance,
                    CategoricalDriftMeasurement::KullbackLeibler => self.kullback_leibler,
                    CategoricalDriftMeasurement::ChiSquared => self.chi_squared,
                    CategoricalDriftMeasurement::Hellinger => self.hellinger,
                    CategoricalDriftMeasurement::GTest => self.g_test,
                };
                let delta = r.drift_magnitude - threshold;
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
pub struct CategoricalDriftContractBuilder {
    jensen_shannon: Option<f64>,
    population_stability_index: Option<f64>,
    wasserstein_distance: Option<f64>,
    kullback_leibler: Option<f64>,
    chi_squared: Option<f64>,
    hellinger: Option<f64>,
    g_test: Option<f64>,
}

impl CategoricalDriftContractBuilder {
    pub fn new() -> CategoricalDriftContractBuilder {
        CategoricalDriftContractBuilder::default()
    }

    pub fn with_jensen_shannon(mut self, value: f64) -> CategoricalDriftContractBuilder {
        self.jensen_shannon = Some(value);
        self
    }

    pub fn with_population_stability_index(
        mut self,
        value: f64,
    ) -> CategoricalDriftContractBuilder {
        self.population_stability_index = Some(value);
        self
    }

    pub fn with_wasserstein_distance(mut self, value: f64) -> CategoricalDriftContractBuilder {
        self.wasserstein_distance = Some(value);
        self
    }

    pub fn with_kullback_leibler(mut self, value: f64) -> CategoricalDriftContractBuilder {
        self.kullback_leibler = Some(value);
        self
    }

    pub fn with_chi_squared(mut self, value: f64) -> CategoricalDriftContractBuilder {
        self.chi_squared = Some(value);
        self
    }

    pub fn with_hellinger(mut self, value: f64) -> CategoricalDriftContractBuilder {
        self.hellinger = Some(value);
        self
    }

    pub fn with_g_test(mut self, value: f64) -> CategoricalDriftContractBuilder {
        self.g_test = Some(value);
        self
    }

    pub fn build(self) -> CategoricalDriftContract {
        CategoricalDriftContract {
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
            chi_squared: self
                .chi_squared
                .unwrap_or(defaults::categorical::CHI_SQUARED),
            hellinger: self.hellinger.unwrap_or(defaults::common::HELLINGER),
            g_test: self.g_test.unwrap_or(defaults::categorical::G_TEST),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NullableCategoricalDriftContract {
    pub jensen_shannon: f64,
    pub population_stability_index: f64,
    pub wasserstein_distance: f64,
    pub kullback_leibler: f64,
    pub chi_squared: f64,
    pub hellinger: f64,
    pub g_test: f64,
    pub null_percentage: f64,
}

impl NullableCategoricalDriftContract {
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
        results: &[DriftComputation<CategoricalDriftMeasurement>],
    ) -> NullableDriftCheck<CategoricalDriftMeasurement> {
        let eval = results
            .iter()
            .map(|r| {
                let threshold = match r.drift_type {
                    CategoricalDriftMeasurement::JensenShannon => self.jensen_shannon,
                    CategoricalDriftMeasurement::PopulationStabilityIndex => {
                        self.population_stability_index
                    }
                    CategoricalDriftMeasurement::WassersteinDistance => self.wasserstein_distance,
                    CategoricalDriftMeasurement::KullbackLeibler => self.kullback_leibler,
                    CategoricalDriftMeasurement::ChiSquared => self.chi_squared,
                    CategoricalDriftMeasurement::Hellinger => self.hellinger,
                    CategoricalDriftMeasurement::GTest => self.g_test,
                };
                let delta = r.drift_magnitude - threshold;
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
pub struct NullableCategoricalDriftContractBuilder {
    jensen_shannon: Option<f64>,
    population_stability_index: Option<f64>,
    wasserstein_distance: Option<f64>,
    kullback_leibler: Option<f64>,
    chi_squared: Option<f64>,
    hellinger: Option<f64>,
    g_test: Option<f64>,
    null_percentage: Option<f64>,
}

impl NullableCategoricalDriftContractBuilder {
    pub fn new() -> NullableCategoricalDriftContractBuilder {
        NullableCategoricalDriftContractBuilder::default()
    }

    pub fn with_jensen_shannon(mut self, value: f64) -> NullableCategoricalDriftContractBuilder {
        self.jensen_shannon = Some(value);
        self
    }

    pub fn with_population_stability_index(
        mut self,
        value: f64,
    ) -> NullableCategoricalDriftContractBuilder {
        self.population_stability_index = Some(value);
        self
    }

    pub fn with_wasserstein_distance(
        mut self,
        value: f64,
    ) -> NullableCategoricalDriftContractBuilder {
        self.wasserstein_distance = Some(value);
        self
    }

    pub fn with_kullback_leibler(mut self, value: f64) -> NullableCategoricalDriftContractBuilder {
        self.kullback_leibler = Some(value);
        self
    }

    pub fn with_chi_squared(mut self, value: f64) -> NullableCategoricalDriftContractBuilder {
        self.chi_squared = Some(value);
        self
    }

    pub fn with_hellinger(mut self, value: f64) -> NullableCategoricalDriftContractBuilder {
        self.hellinger = Some(value);
        self
    }

    pub fn with_g_test(mut self, value: f64) -> NullableCategoricalDriftContractBuilder {
        self.g_test = Some(value);
        self
    }

    pub fn with_null_percentage(mut self, value: f64) -> NullableCategoricalDriftContractBuilder {
        self.null_percentage = Some(value);
        self
    }

    pub fn build(self) -> NullableCategoricalDriftContract {
        NullableCategoricalDriftContract {
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
            chi_squared: self
                .chi_squared
                .unwrap_or(defaults::categorical::CHI_SQUARED),
            hellinger: self.hellinger.unwrap_or(defaults::common::HELLINGER),
            g_test: self.g_test.unwrap_or(defaults::categorical::G_TEST),
            null_percentage: self
                .null_percentage
                .unwrap_or(defaults::common::NULL_PERCENTAGE),
        }
    }
}
