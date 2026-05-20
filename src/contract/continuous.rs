use crate::core::drift_metrics::ContinuousDriftType;
use crate::drift::DriftComputation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ContinuousDriftContract {
    pub jensen_shannon: Option<f64>,
    pub population_stability_index: Option<f64>,
    pub wasserstein_distance: Option<f64>,
    pub kullback_leibler: Option<f64>,
    pub kolmogorov_smirnov: Option<f64>,
    pub hellinger: Option<f64>,
}

impl ContinuousDriftContract {
    /// Returns `true` if every result whose metric has a configured threshold is at or below that
    /// threshold. Results for metrics with no threshold always pass.
    pub fn check(&self, results: &[DriftComputation<ContinuousDriftType>]) -> bool {
        results.iter().all(|r| {
            let threshold = match r.drift_type {
                ContinuousDriftType::JensenShannon => self.jensen_shannon,
                ContinuousDriftType::PopulationStabilityIndex => self.population_stability_index,
                ContinuousDriftType::WassersteinDistance => self.wasserstein_distance,
                ContinuousDriftType::KullbackLeibler => self.kullback_leibler,
                ContinuousDriftType::KolmogorovSmirnov => self.kolmogorov_smirnov,
                ContinuousDriftType::Hellinger => self.hellinger,
            };
            threshold.map_or(true, |t| r.drift_magnitude <= t)
        })
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
            jensen_shannon: self.jensen_shannon,
            population_stability_index: self.population_stability_index,
            wasserstein_distance: self.wasserstein_distance,
            kullback_leibler: self.kullback_leibler,
            kolmogorov_smirnov: self.kolmogorov_smirnov,
            hellinger: self.hellinger,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NullableContinuousDriftContract {
    pub jensen_shannon: Option<f64>,
    pub population_stability_index: Option<f64>,
    pub wasserstein_distance: Option<f64>,
    pub kullback_leibler: Option<f64>,
    pub kolmogorov_smirnov: Option<f64>,
    pub hellinger: Option<f64>,
    pub null_percentage: Option<f64>,
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
        null_percentage: f64,
        results: &[DriftComputation<ContinuousDriftType>],
    ) -> bool {
        if self.null_percentage.map_or(false, |t| null_percentage > t) {
            return false;
        }
        results.iter().all(|r| {
            let threshold = match r.drift_type {
                ContinuousDriftType::JensenShannon => self.jensen_shannon,
                ContinuousDriftType::PopulationStabilityIndex => self.population_stability_index,
                ContinuousDriftType::WassersteinDistance => self.wasserstein_distance,
                ContinuousDriftType::KullbackLeibler => self.kullback_leibler,
                ContinuousDriftType::KolmogorovSmirnov => self.kolmogorov_smirnov,
                ContinuousDriftType::Hellinger => self.hellinger,
            };
            threshold.map_or(true, |t| r.drift_magnitude <= t)
        })
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
            jensen_shannon: self.jensen_shannon,
            population_stability_index: self.population_stability_index,
            wasserstein_distance: self.wasserstein_distance,
            kullback_leibler: self.kullback_leibler,
            kolmogorov_smirnov: self.kolmogorov_smirnov,
            hellinger: self.hellinger,
            null_percentage: self.null_percentage,
        }
    }
}
