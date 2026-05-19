use crate::core::drift_metrics::CategoricalDriftType;
use crate::drift::DriftComputation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CategoricalDriftContract {
    pub jensen_shannon: Option<f64>,
    pub population_stability_index: Option<f64>,
    pub wasserstein_distance: Option<f64>,
    pub kullback_leibler: Option<f64>,
    pub chi_squared: Option<f64>,
    pub hellinger: Option<f64>,
    pub g_test: Option<f64>,
}

impl CategoricalDriftContract {
    /// Returns `true` if every result whose metric has a configured threshold is at or below that
    /// threshold. Results for metrics with no threshold always pass.
    pub fn check(&self, results: &[DriftComputation<CategoricalDriftType>]) -> bool {
        results.iter().all(|r| {
            let threshold = match r.drift_type {
                CategoricalDriftType::JensenShannon => self.jensen_shannon,
                CategoricalDriftType::PopulationStabilityIndex => self.population_stability_index,
                CategoricalDriftType::WassersteinDistance => self.wasserstein_distance,
                CategoricalDriftType::KullbackLeibler => self.kullback_leibler,
                CategoricalDriftType::ChiSquared => self.chi_squared,
                CategoricalDriftType::Hellinger => self.hellinger,
                CategoricalDriftType::GTest => self.g_test,
            };
            threshold.map_or(true, |t| r.drift_magnitude <= t)
        })
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

    pub fn with_population_stability_index(mut self, value: f64) -> CategoricalDriftContractBuilder {
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
            jensen_shannon: self.jensen_shannon,
            population_stability_index: self.population_stability_index,
            wasserstein_distance: self.wasserstein_distance,
            kullback_leibler: self.kullback_leibler,
            chi_squared: self.chi_squared,
            hellinger: self.hellinger,
            g_test: self.g_test,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NullableCategoricalDriftContract {
    pub jensen_shannon: Option<f64>,
    pub population_stability_index: Option<f64>,
    pub wasserstein_distance: Option<f64>,
    pub kullback_leibler: Option<f64>,
    pub chi_squared: Option<f64>,
    pub hellinger: Option<f64>,
    pub g_test: Option<f64>,
    pub null_percentage: Option<f64>,
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
        null_percentage: f64,
        results: &[DriftComputation<CategoricalDriftType>],
    ) -> bool {
        if self
            .null_percentage
            .map_or(false, |t| null_percentage > t)
        {
            return false;
        }
        results.iter().all(|r| {
            let threshold = match r.drift_type {
                CategoricalDriftType::JensenShannon => self.jensen_shannon,
                CategoricalDriftType::PopulationStabilityIndex => self.population_stability_index,
                CategoricalDriftType::WassersteinDistance => self.wasserstein_distance,
                CategoricalDriftType::KullbackLeibler => self.kullback_leibler,
                CategoricalDriftType::ChiSquared => self.chi_squared,
                CategoricalDriftType::Hellinger => self.hellinger,
                CategoricalDriftType::GTest => self.g_test,
            };
            threshold.map_or(true, |t| r.drift_magnitude <= t)
        })
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
            jensen_shannon: self.jensen_shannon,
            population_stability_index: self.population_stability_index,
            wasserstein_distance: self.wasserstein_distance,
            kullback_leibler: self.kullback_leibler,
            chi_squared: self.chi_squared,
            hellinger: self.hellinger,
            g_test: self.g_test,
            null_percentage: self.null_percentage,
        }
    }
}
