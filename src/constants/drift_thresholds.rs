pub mod common {
    /// [0, 1]. Flag at 0.1 — distributions have meaningfully diverged.
    pub(crate) const JENSEN_SHANNON: f64 = 0.1;
    /// [0, ∞). Industry standard: < 0.1 no shift, 0.1–0.25 moderate, > 0.25 significant.
    pub(crate) const POPULATION_STABILITY_INDEX: f64 = 0.25;
    /// [0, 1]. Flag at 0.1 — 10% shift in mass between distributions.
    pub(crate) const WASSERSTEIN_DISTANCE: f64 = 0.1;
    /// [0, ∞). Flag at 0.2 — KL grows faster than bounded metrics for the same shift.
    pub(crate) const KULLBACK_LEIBLER: f64 = 0.2;
    /// [0, 1]. Flag at 0.1 — comparable sensitivity to JSD.
    pub(crate) const HELLINGER: f64 = 0.1;
    /// Flag when more than 10% of observed values are null.
    pub(crate) const NULL_PERCENTAGE: f64 = 0.10;
}

pub mod continuous {
    /// [0, 1]. Flag at 0.1 — max CDF gap indicating a meaningful location or shape shift.
    pub(crate) const KOLMOGOROV_SMIRNOV: f64 = 0.1;
}

pub mod categorical {
    /// [0, ∞). Chi-squared statistic scales with sample size and degrees of freedom.
    /// Default of 10.0 is a conservative action threshold that works across typical cardinalities.
    pub(crate) const CHI_SQUARED: f64 = 10.0;
    /// [0, ∞). Asymptotically chi-squared distributed; same default as CHI_SQUARED.
    pub(crate) const G_TEST: f64 = 10.0;
}
