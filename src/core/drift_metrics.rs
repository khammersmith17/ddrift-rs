use crate::drift_types::DriftComputation;
const STABILITY_EPS: f64 = 1e-12;

// Enum to be used in DriftContainer trait when drift metric computation args differ between the
// Continuous and Categorical DriftContainers.
#[derive(Debug, PartialEq)]
pub(crate) enum DriftContainerType {
    Continuous,
    Categorical,
}

//TODO: list out continuous and categorical metrics supported in different enums.
//Then also have a single enum

#[derive(Debug, PartialEq, Copy, Clone)]
#[non_exhaustive]
pub enum DataDriftType {
    JensenShannon,
    PopulationStabilityIndex,
    WassersteinDistance,
    KullbackLeibler,
}

impl TryFrom<&str> for DataDriftType {
    type Error = crate::core::error::DriftError;
    fn try_from(val: &str) -> Result<Self, Self::Error> {
        match val {
            "JensenShannon" => Ok(Self::JensenShannon),
            "PopulationStabilityIndex" => Ok(Self::PopulationStabilityIndex),
            "WassersteinDistance" => Ok(Self::WassersteinDistance),
            "KullbackLeibler" => Ok(Self::KullbackLeibler),
            _ => Err(Self::Error::UnsupportedDriftType),
        }
    }
}

/// Trait that defines common behavior required to compute drift metrics.
/// When additional drift criteria is added, this may be built upon. Behavior should not be removed
/// for backward compatibility later.
pub(crate) trait DriftContainer {
    fn get_baseline_hist(&self) -> &[f64];

    fn get_runtime_bins(&self) -> &[f64];

    fn num_examples(&self) -> f64;

    fn container_type(&self) -> DriftContainerType;
}

pub(crate) fn streaming_drift_single<T: DriftContainer>(
    drift_container: &T,
    drift_type: DataDriftType,
) -> DriftComputation {
    let drift_magnitude = global_compute_drift(drift_container, drift_type);
    DriftComputation {
        drift_magnitude,
        drift_type,
    }
}

pub(crate) fn streaming_drift_multi<T: DriftContainer>(
    drift_container: &T,
    drift_types: &[DataDriftType],
) -> Vec<DriftComputation> {
    drift_types
        .iter()
        .map(|t| streaming_drift_single(drift_container, *t))
        .collect()
}

/// Global drift computation that allows a shared interface between all drift types.
/// Additional drift criteria added later should all dispatch drift through this method.
pub(crate) fn global_compute_drift<T: DriftContainer>(
    drift_container: &T,
    drift_type: DataDriftType,
) -> f64 {
    match drift_type {
        DataDriftType::PopulationStabilityIndex => compute_psi(
            drift_container.get_baseline_hist(),
            drift_container.get_runtime_bins(),
            drift_container.num_examples(),
        ),
        DataDriftType::KullbackLeibler => compute_kl_divergence_drift(
            drift_container.get_baseline_hist(),
            drift_container.get_runtime_bins(),
            drift_container.num_examples(),
        ),
        DataDriftType::JensenShannon => compute_jensen_shannon_divergence_drift(
            drift_container.get_baseline_hist(),
            drift_container.get_runtime_bins(),
            drift_container.num_examples(),
        ),
        DataDriftType::WassersteinDistance => match drift_container.container_type() {
            DriftContainerType::Continuous => continuous_wasserstein_distance(
                drift_container.get_baseline_hist(),
                drift_container.get_runtime_bins(),
                drift_container.num_examples(),
            ),
            DriftContainerType::Categorical => categorical_wasserstein_distance(
                drift_container.get_baseline_hist(),
                drift_container.get_runtime_bins(),
                drift_container.num_examples(),
            ),
        },
    }
}

//define traits to use the continuous and discrete drift bins, where getting the implementation of
//a particular metric is declared via trait methods. This will only implement the logic on the
//class when the user it in scope. Traits will be implemented where the bin types are implemented

// Compute psi on runtime and baseline bins. Element wise distance for each bucket with a
// sum reduction.
#[inline]
pub(crate) fn compute_psi(baseline_hist: &[f64], runtime_bins: &[f64], n: f64) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());

    baseline_hist
        .iter()
        .zip(runtime_bins.iter())
        .map(|(bl, rt)| {
            let b = (bl + STABILITY_EPS).max(STABILITY_EPS);
            let r = ((rt + STABILITY_EPS) / n).max(STABILITY_EPS);
            (b - r) * (b / r).ln()
        })
        .sum()
}

#[inline]
pub(crate) fn compute_kl_divergence_drift(
    baseline_hist: &[f64],
    runtime_bins: &[f64],
    n: f64,
) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());

    baseline_hist
        .iter()
        .zip(runtime_bins.iter())
        .map(|(bl, rt)| {
            let dist_rt = (*rt + STABILITY_EPS) / n;
            let dist_bl = (*bl + STABILITY_EPS).max(STABILITY_EPS);
            dist_bl * (dist_bl / dist_rt).max(STABILITY_EPS).ln()
        })
        .sum()
}

#[inline]
pub(crate) fn compute_jensen_shannon_divergence_drift(
    baseline_hist: &[f64],
    runtime_bins: &[f64],
    n: f64,
) -> f64 {
    // validate that rt and baseline bins are of same length
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());

    let mut js = 0_f64;
    let half_fac = 0.5_f64;

    for (bl, rt) in baseline_hist.iter().zip(runtime_bins.iter()) {
        let p = (bl + STABILITY_EPS).max(STABILITY_EPS);
        let q = ((rt / n) + STABILITY_EPS).max(STABILITY_EPS);
        let m = (p + q) * half_fac;

        js += (half_fac * p * (p / m).ln()) + (half_fac * q * (q / m).ln());
    }

    js / 2_f64.ln()
}

#[inline]
pub(crate) fn continuous_wasserstein_distance(
    baseline_hist: &[f64],
    runtime_bins: &[f64],
    n: f64,
) -> f64 {
    // Outer 2 bins are effectively overflow on the tails.
    // Skipping those quantiole bins here.
    let n_bins = baseline_hist.len();
    let w_dist = wasserstein_inner(
        &baseline_hist[1..n_bins - 1],
        &runtime_bins[1..n_bins - 1],
        n,
    );

    w_dist / (baseline_hist.len() - 2).max(1_usize) as f64
}

#[inline]
fn wasserstein_inner(baseline_hist: &[f64], runtime_bins: &[f64], n: f64) -> f64 {
    debug_assert_eq!(runtime_bins.len(), baseline_hist.len());
    let mut w_dist = 0_f64;

    for (bl, rt) in baseline_hist.iter().zip(runtime_bins.iter()) {
        let p = (bl + STABILITY_EPS).max(STABILITY_EPS);
        let q = ((rt / n) + STABILITY_EPS).max(STABILITY_EPS);

        w_dist += (p - q).abs();
    }
    w_dist
}

#[inline]
pub(crate) fn categorical_wasserstein_distance(
    baseline_hist: &[f64],
    runtime_bins: &[f64],
    n: f64,
) -> f64 {
    // bins are effectively unit width for categorical distributions
    // this effectively turns into total variation distance
    let w_dist = wasserstein_inner(&baseline_hist, &runtime_bins, n);

    w_dist * 0.5_f64
}

// assumes that bins contains unnormalized counts
#[inline]
fn chi_square(
    baseline_hist: &[f64],
    n_bl_samples: f64,
    runtime_hist: &[f64],
    n_rt_samples: f64,
) -> (f64, f64) {
    debug_assert_eq!(baseline_hist.len(), runtime_hist.len());
    let total_samples = n_bl_samples + n_rt_samples;

    let mut e_bl = 0_f64;
    let mut e_rt = 0_f64;
    let mut chi_sq = 0_f64;

    baseline_hist
        .iter()
        .zip(runtime_hist.iter())
        .for_each(|(e, o)| {
            let bin_samples = e + o;
            e_bl += (n_bl_samples * bin_samples) / total_samples;
            e_rt += (n_rt_samples * bin_samples) / total_samples;
            chi_sq += ((e - e_bl).powi(2) / e_bl).max(0_f64);
            chi_sq += ((o - e_rt).powi(2) / e_rt).max(0_f64);
        });

    return (chi_sq, (baseline_hist.len() - 1) as f64);
}
