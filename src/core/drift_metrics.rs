use crate::drift::{DriftComputation, DriftComputationMulti};
const STABILITY_EPS: f64 = 1e-12;
const HALF_CONSTANT: f64 = 0.5_f64;

pub trait DriftMeasurement {}

/// Dictates the supported continuous drift metrics in this crate. This enum is non-exhuastive as
/// support will be added later. All drift computations and containers build off of this set of
/// avialble measurements. Each metric has a different range of possible values:
///     JensenShannon: [0, 1]
///     PopulationStabilityIndex: [0, inf)
///     WassersteinDistance: [0, 1]
///     KullbackLeibler: [0, inf)
///     KolmogorovSmirnov: [0, 1]
///     Hellinger: [0, 1]
#[derive(Debug, PartialEq, Copy, Clone)]
#[non_exhaustive]
pub enum ContinuousDriftMeasurement {
    JensenShannon,
    PopulationStabilityIndex,
    WassersteinDistance,
    KullbackLeibler,
    KolmogorovSmirnov,
    Hellinger,
}

impl DriftMeasurement for ContinuousDriftMeasurement {}

impl TryFrom<&str> for ContinuousDriftMeasurement {
    type Error = crate::core::error::DriftError;
    fn try_from(val: &str) -> Result<Self, Self::Error> {
        match val {
            "JensenShannon" => Ok(Self::JensenShannon),
            "PopulationStabilityIndex" => Ok(Self::PopulationStabilityIndex),
            "WassersteinDistance" => Ok(Self::WassersteinDistance),
            "KullbackLeibler" => Ok(Self::KullbackLeibler),
            "KolmogorovSmirnov" => Ok(Self::KolmogorovSmirnov),
            "Hellinger" => Ok(Self::Hellinger),
            _ => Err(Self::Error::UnsupportedDriftType),
        }
    }
}

/// Dictates the supported categorical drift metrics in this crate. This enum is non-exhuastive as
/// support will be added later. All drift computations and containers build off of this set of
/// avialble measurements. Each metric has a different range of possible values:
///     JensenShannon: [0, 1]
///     PopulationStabilityIndex: [0, inf)
///     WassersteinDistance: [0, 1]
///     KullbackLeibler: [0, inf)
///     ChiSquared: [0, 1]
///     Hellinger: [0, 1]
///     GTest: [0, inf)
#[derive(Debug, PartialEq, Copy, Clone)]
#[non_exhaustive]
pub enum CategoricalDriftMeasurement {
    JensenShannon,
    PopulationStabilityIndex,
    WassersteinDistance,
    KullbackLeibler,
    ChiSquared,
    Hellinger,
    GTest,
}

impl DriftMeasurement for CategoricalDriftMeasurement {}

impl TryFrom<&str> for CategoricalDriftMeasurement {
    type Error = crate::core::error::DriftError;
    fn try_from(val: &str) -> Result<Self, Self::Error> {
        match val {
            "JensenShannon" => Ok(Self::JensenShannon),
            "PopulationStabilityIndex" => Ok(Self::PopulationStabilityIndex),
            "WassersteinDistance" => Ok(Self::WassersteinDistance),
            "KullbackLeibler" => Ok(Self::KullbackLeibler),
            "ChiSquared" => Ok(Self::ChiSquared),
            "Hellinger" => Ok(Self::Hellinger),
            "GTest" => Ok(Self::GTest),
            _ => Err(Self::Error::UnsupportedDriftType),
        }
    }
}

/// Trait that defines common behavior required to compute drift metrics.
/// When additional drift criteria is added, this may be built upon. Behavior should not be removed
/// for backward compatibility later.
pub(crate) trait DriftContainer {
    fn baseline_bins(&self) -> &[f64];

    fn runtime_bins(&self) -> &[f64];

    fn runtime_sample_size(&self) -> f64;

    fn baseline_sample_size(&self) -> f64;

    fn drift_components(&self) -> (&[f64], f64, &[f64], f64) {
        (
            self.baseline_bins(),
            self.baseline_sample_size(),
            self.runtime_bins(),
            self.runtime_sample_size(),
        )
    }
}

pub(crate) fn compute_drift_continuous<T: DriftContainer>(
    drift_container: &T,
    drift_type: ContinuousDriftMeasurement,
) -> DriftComputation<ContinuousDriftMeasurement> {
    let drift_magnitude = continuous_drift_computation(drift_container, drift_type);
    DriftComputation {
        drift_magnitude,
        drift_type,
    }
}

pub(crate) fn compute_drift_continuous_multi<T: DriftContainer>(
    drift_container: &T,
    drift_types: &[ContinuousDriftMeasurement],
) -> DriftComputationMulti<ContinuousDriftMeasurement> {
    DriftComputationMulti {
        drift: drift_types
            .iter()
            .map(|t| compute_drift_continuous(drift_container, *t))
            .collect(),
    }
}

// Central entry point into the continuous computation methods. DriftContainer trait provides all
// the methods required to acquire the components needed to compute.
fn continuous_drift_computation<T: DriftContainer>(
    drift_container: &T,
    drift_type: ContinuousDriftMeasurement,
) -> f64 {
    let (bl_bins, bl_pop_size, rt_bins, rt_pop_size) = drift_container.drift_components();

    match drift_type {
        ContinuousDriftMeasurement::JensenShannon => {
            compute_jensen_shannon_divergence_drift(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        ContinuousDriftMeasurement::PopulationStabilityIndex => {
            compute_psi(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        ContinuousDriftMeasurement::KullbackLeibler => {
            compute_kl_divergence_drift(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        ContinuousDriftMeasurement::WassersteinDistance => {
            continuous_wasserstein_distance(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        ContinuousDriftMeasurement::KolmogorovSmirnov => {
            kolmogorov_smirnov(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        ContinuousDriftMeasurement::Hellinger => {
            hellinger(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
    }
}

pub(crate) fn compute_drift_categorical<T: DriftContainer>(
    drift_container: &T,
    drift_type: CategoricalDriftMeasurement,
) -> DriftComputation<CategoricalDriftMeasurement> {
    let drift_magnitude = categorical_drift_computation(drift_container, drift_type);
    DriftComputation {
        drift_magnitude,
        drift_type,
    }
}

pub(crate) fn compute_drift_categorical_multi<T: DriftContainer>(
    drift_container: &T,
    drift_types: &[CategoricalDriftMeasurement],
) -> DriftComputationMulti<CategoricalDriftMeasurement> {
    DriftComputationMulti {
        drift: drift_types
            .iter()
            .map(|t| compute_drift_categorical(drift_container, *t))
            .collect(),
    }
}

fn categorical_drift_computation<T: DriftContainer>(
    drift_container: &T,
    drift_type: CategoricalDriftMeasurement,
) -> f64 {
    let (bl_bins, bl_pop_size, rt_bins, rt_pop_size) = drift_container.drift_components();

    match drift_type {
        CategoricalDriftMeasurement::JensenShannon => {
            compute_jensen_shannon_divergence_drift(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        CategoricalDriftMeasurement::PopulationStabilityIndex => {
            compute_psi(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        CategoricalDriftMeasurement::KullbackLeibler => {
            compute_kl_divergence_drift(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        CategoricalDriftMeasurement::WassersteinDistance => {
            categorical_wasserstein_distance(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        CategoricalDriftMeasurement::ChiSquared => {
            chi_squared(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        CategoricalDriftMeasurement::Hellinger => {
            hellinger(bl_bins, bl_pop_size, rt_bins, rt_pop_size)
        }
        CategoricalDriftMeasurement::GTest => g_test(bl_bins, bl_pop_size, rt_bins, rt_pop_size),
    }
}

// Numerically stable division.
#[inline]
fn numeric_stable_divide(dividened: f64, divisor: f64) -> f64 {
    (dividened / (divisor + STABILITY_EPS)).max(STABILITY_EPS)
}

// Derive the bin ratio, applying numerical stablity.
#[inline]
fn stable_apply_ratio(bin_c: f64, pop_size: f64) -> f64 {
    numeric_stable_divide(bin_c, pop_size)
}

/// Population Stability Index. Measures how much a distribution has shifted relative to a
/// baseline. Computed as the sum of element-wise `(p - q) * ln(p / q)` over all bins, where
/// `p` is the baseline proportion and `q` is the runtime proportion.
///
/// Unbounded and non-negative. Common interpretation thresholds: < 0.1 no significant shift,
/// 0.1–0.25 moderate shift, > 0.25 significant shift.
///
/// Expects unnormalized bin counts and population sizes for both groups.
#[inline]
pub(crate) fn compute_psi(
    baseline_hist: &[f64],
    bl_n: f64,
    runtime_bins: &[f64],
    rt_n: f64,
) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());

    baseline_hist
        .iter()
        .zip(runtime_bins.iter())
        .map(|(bl, rt)| {
            let b = stable_apply_ratio(*bl, bl_n);
            let r = stable_apply_ratio(*rt, rt_n);
            (b - r) * (b / r).ln()
        })
        .sum()
}

/// Kullback-Leibler divergence. Asymmetric measure of how much the runtime distribution
/// diverges from the baseline. Computed as `sum(p * ln(p / q))` where `p` is the baseline
/// proportion and `q` is the runtime proportion — i.e. KL(baseline || runtime).
///
/// Unbounded and non-negative. Not symmetric: KL(baseline || runtime) ≠ KL(runtime || baseline).
/// Sensitive to bins where the runtime has probability mass but the baseline does not.
///
/// Expects unnormalized bin counts and population sizes for both groups.
#[inline]
pub(crate) fn compute_kl_divergence_drift(
    baseline_hist: &[f64],
    bl_n: f64,
    runtime_bins: &[f64],
    rt_n: f64,
) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());

    baseline_hist
        .iter()
        .zip(runtime_bins.iter())
        .map(|(bl, rt)| {
            let dist_bl = stable_apply_ratio(*bl, bl_n);
            let dist_rt = stable_apply_ratio(*rt, rt_n);
            dist_bl * (dist_bl / dist_rt).max(STABILITY_EPS).ln()
        })
        .sum()
}

/// Jensen-Shannon divergence. Symmetric, smoothed variant of KL divergence. Computed as
/// `(KL(p || m) + KL(q || m)) / 2` where `m = (p + q) / 2` is the pointwise mixture, then
/// normalized by `ln(2)` to produce a result bounded in [0, 1].
///
/// Unlike KL, JSD is defined even when one distribution has zero mass in a bin. Bounded and
/// symmetric, making it well-suited for comparing distributions without a fixed reference direction.
///
/// Expects unnormalized bin counts and population sizes for both groups.
#[inline]
pub(crate) fn compute_jensen_shannon_divergence_drift(
    baseline_hist: &[f64],
    bl_n: f64,
    runtime_bins: &[f64],
    rt_n: f64,
) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let js: f64 = baseline_hist
        .iter()
        .zip(runtime_bins.iter())
        .map(|(bl, rt)| {
            let p = stable_apply_ratio(*bl, bl_n);
            let q = stable_apply_ratio(*rt, rt_n);
            let m = (p + q) * HALF_CONSTANT;

            (HALF_CONSTANT * p * (p / m).ln()) + (HALF_CONSTANT * q * (q / m).ln())
        })
        .sum();

    js / 2_f64.ln()
}

/// Wasserstein distance (Earth Mover's Distance) for continuous distributions. Computed as the
/// mean L1 distance between the two normalized bin proportions, skipping the two outermost
/// overflow bins which capture values outside the baseline range. The result is averaged over
/// the number of interior bins.
///
/// Unbounded in theory but practically small for similar distributions. The overflow bin
/// exclusion prevents out-of-range runtime values from dominating the statistic.
///
/// Expects unnormalized bin counts and population sizes for both groups.
#[inline]
pub(crate) fn continuous_wasserstein_distance(
    baseline_hist: &[f64],
    bl_n: f64,
    runtime_bins: &[f64],
    rt_n: f64,
) -> f64 {
    // Outer 2 bins are effectively overflow on the tails.
    // Skipping those quantile bins here.
    let n_bins = baseline_hist.len();
    let w_dist = wasserstein_inner(
        &baseline_hist[1..n_bins - 1],
        bl_n,
        &runtime_bins[1..n_bins - 1],
        rt_n,
    );

    w_dist / (baseline_hist.len() - 2).max(1_usize) as f64
}

// Abstracts out the inner implementation of wasserstein, irrespective of the step size used.
#[inline]
fn wasserstein_inner(baseline_hist: &[f64], bl_n: f64, runtime_bins: &[f64], rt_n: f64) -> f64 {
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    baseline_hist
        .iter()
        .zip(runtime_bins.iter())
        .map(|(bl, rt)| {
            let p = stable_apply_ratio(*bl, bl_n);
            let q = stable_apply_ratio(*rt, rt_n);

            (p - q).abs()
        })
        .sum()
}

/// Wasserstein distance for categorical distributions. For categorical data bins have unit width,
/// so this reduces to Total Variation Distance: `0.5 * sum |p_i - q_i|`. Bounded in [0, 1],
/// where 0 means identical distributions and 1 means completely disjoint support.
///
/// Expects unnormalized bin counts and population sizes for both groups.
#[inline]
pub(crate) fn categorical_wasserstein_distance(
    baseline_hist: &[f64],
    bl_n: f64,
    runtime_bins: &[f64],
    rt_n: f64,
) -> f64 {
    // bins are effectively unit width for categorical distributions
    // this effectively turns into total variation distance
    let w_dist = wasserstein_inner(&baseline_hist, bl_n, &runtime_bins, rt_n);

    w_dist * 0.5_f64
}

// Compute the group level expected, given that the runtime and baseline distributions will have
// different population sizes.
// Maps to total (group count * bin count) / total population.
// where
//  group count is the count of samples in the entire group (baseline or runtime)
//  bin count is the number of examples resolved to the current bin
//  total population is the number of examples in the entire population across the 2 groups
#[inline]
fn chi_sq_e(group_c: f64, bin_c: f64, total_pop: f64) -> f64 {
    numeric_stable_divide(group_c * bin_c, total_pop)
}

// Compute the chi squared value for a particular bin with the normalized and expected values.
#[inline]
fn chi_sq_bin_value(expected: f64, observed: f64, e_bl_norm: f64, e_rt_norm: f64) -> f64 {
    ((expected - e_bl_norm).powi(2) / e_bl_norm) + ((observed - e_rt_norm).powi(2) / e_rt_norm)
}

/// Pearson chi-squared test for homogeneity. Tests whether two samples are drawn from the same
/// distribution, correctly accounting for different population sizes between groups. Expected
/// counts are derived from the joint bin totals: `E[group][bin] = group_n * (bl[i] + rt[i]) / total_n`.
///
/// Unbounded and non-negative. Asymptotically chi-squared distributed with `k - 1` degrees of
/// freedom where `k` is the number of bins, which can be used to derive a p-value upstream.
///
/// Expects unnormalized bin counts and population sizes for both groups.
// assumes that bins contains unnormalized counts
#[inline]
fn chi_squared(
    baseline_hist: &[f64],
    n_bl_samples: f64,
    runtime_hist: &[f64],
    n_rt_samples: f64,
) -> f64 {
    debug_assert_eq!(baseline_hist.len(), runtime_hist.len());
    let total_samples = n_bl_samples + n_rt_samples;

    let chi_sq = baseline_hist
        .iter()
        .zip(runtime_hist.iter())
        .map(|(e, o)| {
            let bin_samples = e + o;
            let e_bl = chi_sq_e(n_bl_samples, bin_samples, total_samples);
            let e_rt = chi_sq_e(n_rt_samples, bin_samples, total_samples);
            chi_sq_bin_value(*e, *o, e_bl, e_rt)
        })
        .sum();

    return chi_sq;
}

/// Kolmogorov-Smirnov statistic. The maximum absolute difference between the two empirical CDFs,
/// where each CDF is built by cumulating normalized bin counts left to right. Both groups are
/// normalized independently so different population sizes are handled correctly.
///
/// Bounded in [0, 1]. Sensitive to location and shape shifts. Approximated from histogram bins
/// rather than raw samples — resolution is limited by bin count.
///
/// Expects unnormalized bin counts and population sizes for both groups.
#[inline]
fn kolmogorov_smirnov(baseline_hist: &[f64], bl_n: f64, runtime_hist: &[f64], rt_n: f64) -> f64 {
    debug_assert_eq!(baseline_hist.len(), runtime_hist.len());

    let mut ks = 0_f64;
    let mut rt_cum = 0_f64;
    let mut bl_cum = 0_f64;

    baseline_hist
        .iter()
        .zip(runtime_hist.iter())
        .for_each(|(bl, rt)| {
            bl_cum += bl;
            rt_cum += rt;
            let cdf_bl = stable_apply_ratio(bl_cum, bl_n);
            let cdf_rt = stable_apply_ratio(rt_cum, rt_n);
            ks = ks.max((cdf_bl - cdf_rt).abs());
        });
    ks
}

/// Hellinger distance. Geometric distance between two probability distributions computed as
/// `sqrt(0.5 * sum(sqrt(p_i) - sqrt(q_i))^2)`. Bounded in [0, 1], symmetric, and satisfies
/// the triangle inequality.
///
/// Does not require numerical stabilization for zero bins since `sqrt(0) = 0` is well-defined,
/// unlike divergence-based metrics that involve logarithms. Interpretation: 0 means identical
/// distributions, 1 means completely disjoint support.
///
/// Expects unnormalized bin counts and population sizes for both groups.
#[inline]
fn hellinger(baseline_hist: &[f64], bl_n: f64, runtime_hist: &[f64], rt_n: f64) -> f64 {
    debug_assert_eq!(baseline_hist.len(), runtime_hist.len());
    let h_base: f64 = baseline_hist
        .iter()
        .zip(runtime_hist.iter())
        .map(|(bl, rt)| {
            let p = stable_apply_ratio(*bl, bl_n).sqrt();
            let q = stable_apply_ratio(*rt, rt_n).sqrt();
            (p - q).powi(2)
        })
        .sum();
    (h_base * HALF_CONSTANT).sqrt()
}

#[inline]
fn g_test_elementwise(e: f64, norm_e: f64, o: f64, norm_o: f64) -> f64 {
    let e_resolved = e * (((e + STABILITY_EPS) / norm_e).max(STABILITY_EPS)).ln();
    let o_resolved = o * (((o + STABILITY_EPS) / norm_o).max(STABILITY_EPS)).ln();
    e_resolved + o_resolved
}

/// G-test (log-likelihood ratio test). Tests the same homogeneity null hypothesis as chi-squared
/// but uses `G = 2 * sum(O * ln(O / E))` rather than `(O - E)^2 / E`. More numerically stable
/// than chi-squared when expected counts are small or bins are sparse, since `O * ln(O / E)`
/// approaches zero gracefully as `O → 0`.
///
/// Uses the same expected value construction as chi-squared: `E[group][bin] = group_n * (bl[i] + rt[i]) / total_n`.
/// Unbounded and non-negative. Asymptotically chi-squared distributed with `k - 1` degrees of
/// freedom, so p-values can be derived the same way as for chi-squared.
///
/// Expects unnormalized bin counts and population sizes for both groups.
#[inline]
fn g_test(baseline_hist: &[f64], bl_n: f64, runtime_hist: &[f64], rt_n: f64) -> f64 {
    debug_assert_eq!(baseline_hist.len(), runtime_hist.len());

    let total_samples = bl_n + rt_n;

    let g_test_base: f64 = baseline_hist
        .iter()
        .zip(runtime_hist.iter())
        .map(|(e, o)| {
            let bin_samples = e + o;
            let e_normalized = chi_sq_e(bl_n, bin_samples, total_samples);
            let o_normalized = chi_sq_e(rt_n, bin_samples, total_samples);
            g_test_elementwise(*e, e_normalized, *o, o_normalized)
        })
        .sum();
    g_test_base * 2_f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1000 samples, evenly split across 2 bins
    fn identical() -> (&'static [f64], f64, &'static [f64], f64) {
        (&[500.0, 500.0], 1000.0, &[500.0, 500.0], 1000.0)
    }

    // All baseline mass in bin 0, all runtime mass in bin 1
    fn disjoint() -> (&'static [f64], f64, &'static [f64], f64) {
        (&[1000.0, 0.0], 1000.0, &[0.0, 1000.0], 1000.0)
    }

    // Moderate shift: 70/30 → 30/70
    fn shifted() -> (&'static [f64], f64, &'static [f64], f64) {
        (&[700.0, 300.0], 1000.0, &[300.0, 700.0], 1000.0)
    }

    // --- PSI ---

    #[test]
    fn psi_identical_is_zero() {
        let (bl, bl_n, rt, rt_n) = identical();
        assert!(compute_psi(bl, bl_n, rt, rt_n) < 1e-10);
    }

    #[test]
    fn psi_shifted_exceeds_significant_threshold() {
        // 80/20 → 20/80 flip, expected PSI ≈ 1.66, well above 0.25 threshold
        let bl = [800.0, 200.0];
        let rt = [200.0, 800.0];
        assert!(compute_psi(&bl, 1000.0, &rt, 1000.0) > 0.25);
    }

    #[test]
    fn psi_is_nonnegative() {
        let (bl, bl_n, rt, rt_n) = shifted();
        assert!(compute_psi(bl, bl_n, rt, rt_n) >= 0.0);
    }

    // --- KL divergence ---

    #[test]
    fn kl_identical_is_zero() {
        let (bl, bl_n, rt, rt_n) = identical();
        assert!(compute_kl_divergence_drift(bl, bl_n, rt, rt_n) < 1e-10);
    }

    #[test]
    fn kl_is_nonnegative() {
        let (bl, bl_n, rt, rt_n) = shifted();
        assert!(compute_kl_divergence_drift(bl, bl_n, rt, rt_n) >= 0.0);
    }

    // --- Jensen-Shannon divergence ---

    #[test]
    fn jsd_identical_is_zero() {
        let (bl, bl_n, rt, rt_n) = identical();
        assert!(compute_jensen_shannon_divergence_drift(bl, bl_n, rt, rt_n) < 1e-10);
    }

    #[test]
    fn jsd_is_bounded_zero_to_one() {
        let (bl, bl_n, rt, rt_n) = shifted();
        let result = compute_jensen_shannon_divergence_drift(bl, bl_n, rt, rt_n);
        assert!(result >= 0.0 && result <= 1.0, "JSD out of [0,1]: {result}");
    }

    #[test]
    fn jsd_disjoint_is_near_one() {
        let (bl, bl_n, rt, rt_n) = disjoint();
        let result = compute_jensen_shannon_divergence_drift(bl, bl_n, rt, rt_n);
        assert!(
            result > 0.99,
            "JSD of disjoint distributions should be ~1, got {result}"
        );
    }

    #[test]
    fn jsd_is_symmetric() {
        let (bl, bl_n, rt, rt_n) = shifted();
        let forward = compute_jensen_shannon_divergence_drift(bl, bl_n, rt, rt_n);
        let backward = compute_jensen_shannon_divergence_drift(rt, rt_n, bl, bl_n);
        assert!(
            (forward - backward).abs() < 1e-10,
            "JSD should be symmetric"
        );
    }

    // --- Hellinger distance ---

    #[test]
    fn hellinger_identical_is_zero() {
        let (bl, bl_n, rt, rt_n) = identical();
        assert!(hellinger(bl, bl_n, rt, rt_n) < 1e-10);
    }

    #[test]
    fn hellinger_disjoint_is_one() {
        let (bl, bl_n, rt, rt_n) = disjoint();
        let result = hellinger(bl, bl_n, rt, rt_n);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "Hellinger of disjoint should be 1, got {result}"
        );
    }

    #[test]
    fn hellinger_is_bounded_zero_to_one() {
        let (bl, bl_n, rt, rt_n) = shifted();
        let result = hellinger(bl, bl_n, rt, rt_n);
        assert!(
            result >= 0.0 && result <= 1.0,
            "Hellinger out of [0,1]: {result}"
        );
    }

    #[test]
    fn hellinger_is_symmetric() {
        let (bl, bl_n, rt, rt_n) = shifted();
        let forward = hellinger(bl, bl_n, rt, rt_n);
        let backward = hellinger(rt, rt_n, bl, bl_n);
        assert!(
            (forward - backward).abs() < 1e-10,
            "Hellinger should be symmetric"
        );
    }

    // --- Kolmogorov-Smirnov ---

    #[test]
    fn ks_identical_is_zero() {
        let (bl, bl_n, rt, rt_n) = identical();
        assert!(kolmogorov_smirnov(bl, bl_n, rt, rt_n) < 1e-10);
    }

    #[test]
    fn ks_disjoint_is_one() {
        let (bl, bl_n, rt, rt_n) = disjoint();
        let result = kolmogorov_smirnov(bl, bl_n, rt, rt_n);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "KS of disjoint should be 1, got {result}"
        );
    }

    #[test]
    fn ks_is_bounded_zero_to_one() {
        let (bl, bl_n, rt, rt_n) = shifted();
        let result = kolmogorov_smirnov(bl, bl_n, rt, rt_n);
        assert!(result >= 0.0 && result <= 1.0, "KS out of [0,1]: {result}");
    }

    // --- Wasserstein (continuous) ---

    #[test]
    fn continuous_wasserstein_identical_is_zero() {
        // needs at least 3 bins so tail exclusion doesn't collapse to empty slice
        let bl = [0.0, 500.0, 500.0, 0.0];
        let rt = [0.0, 500.0, 500.0, 0.0];
        assert!(continuous_wasserstein_distance(&bl, 1000.0, &rt, 1000.0) < 1e-10);
    }

    #[test]
    fn continuous_wasserstein_shifted_is_nonzero() {
        let bl = [0.0, 800.0, 200.0, 0.0];
        let rt = [0.0, 200.0, 800.0, 0.0];
        assert!(continuous_wasserstein_distance(&bl, 1000.0, &rt, 1000.0) > 0.0);
    }

    // --- Wasserstein (categorical / TVD) ---

    #[test]
    fn categorical_wasserstein_identical_is_zero() {
        let (bl, bl_n, rt, rt_n) = identical();
        assert!(categorical_wasserstein_distance(bl, bl_n, rt, rt_n) < 1e-10);
    }

    #[test]
    fn categorical_wasserstein_disjoint_is_one() {
        let (bl, bl_n, rt, rt_n) = disjoint();
        let result = categorical_wasserstein_distance(bl, bl_n, rt, rt_n);
        // TVD of fully disjoint distributions = 1 (up to numerical eps)
        assert!(
            (result - 1.0).abs() < 1e-9,
            "TVD of disjoint should be ~1, got {result}"
        );
    }

    #[test]
    fn categorical_wasserstein_is_bounded_zero_to_one() {
        let (bl, bl_n, rt, rt_n) = shifted();
        let result = categorical_wasserstein_distance(bl, bl_n, rt, rt_n);
        assert!(result >= 0.0 && result <= 1.0, "TVD out of [0,1]: {result}");
    }

    // --- Chi-squared ---

    #[test]
    fn chi_squared_identical_is_zero() {
        let (bl, bl_n, rt, rt_n) = identical();
        assert!(chi_squared(bl, bl_n, rt, rt_n) < 1e-10);
    }

    #[test]
    fn chi_squared_is_nonnegative() {
        let (bl, bl_n, rt, rt_n) = shifted();
        assert!(chi_squared(bl, bl_n, rt, rt_n) >= 0.0);
    }

    #[test]
    fn chi_squared_large_shift_is_elevated() {
        // 90/10 → 10/90 flip
        let bl = [900.0, 100.0];
        let rt = [100.0, 900.0];
        // expected ≈ 1280 for this setup
        assert!(chi_squared(&bl, 1000.0, &rt, 1000.0) > 100.0);
    }

    // --- G-test ---

    #[test]
    fn g_test_identical_is_near_zero() {
        let (bl, bl_n, rt, rt_n) = identical();
        assert!(g_test(bl, bl_n, rt, rt_n).abs() < 1e-6);
    }

    #[test]
    fn g_test_is_nonnegative() {
        let (bl, bl_n, rt, rt_n) = shifted();
        assert!(g_test(bl, bl_n, rt, rt_n) >= 0.0);
    }

    #[test]
    fn g_test_large_shift_is_elevated() {
        let bl = [900.0, 100.0];
        let rt = [100.0, 900.0];
        assert!(g_test(&bl, 1000.0, &rt, 1000.0) > 100.0);
    }
}
