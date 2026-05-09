use num_traits::Float;
use serde::{Deserialize, Serialize};

#[non_exhaustive]
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy, Deserialize, Serialize)]
pub enum QuantileType {
    #[default]
    FreedmanDiaconis,
    Scott,
    Sturges,
}

impl TryFrom<&str> for QuantileType {
    type Error = crate::core::error::DriftError;
    fn try_from(val: &str) -> Result<Self, Self::Error> {
        match val {
            "FreedmanDiaconis" => Ok(Self::FreedmanDiaconis),
            "Scott" => Ok(Self::Scott),
            "Sturges" => Ok(Self::Sturges),
            _ => Err(Self::Error::UnsupportedDriftType),
        }
    }
}

impl QuantileType {
    pub fn compute_num_bins<T: Float>(&self, baseline_distribution: &[T]) -> usize {
        match self {
            QuantileType::FreedmanDiaconis => freedman_diaconis(baseline_distribution),
            QuantileType::Scott => scott(baseline_distribution),
            QuantileType::Sturges => sturges(baseline_distribution),
        }
    }
}

const SCOTT_CONSTANT: f32 = 3.49;
pub(crate) const MIN_BIN_CLAMP: usize = 3_usize;

/// Compute the optimal number of bins using Scott's method.
/// Assumes data is sorted. Also assumes that dataset is nonempty.
fn scott<T: Float>(dataset: &[T]) -> usize {
    let n = dataset.len();
    let n_t = T::from(n).unwrap();
    let mean = dataset.iter().copied().fold(T::zero(), |acc, v| acc + v) / n_t;
    let deviation_term = dataset
        .iter()
        .copied()
        .fold(T::zero(), |acc, v| acc + (v - mean).powi(2));

    // use the sample standard deviation
    let std_dev = (T::one() / (n_t - T::one()) * deviation_term).sqrt();
    let bin_width =
        T::from(SCOTT_CONSTANT).unwrap() * std_dev * n_t.powf(T::from(-1_f32 / 3_f32).unwrap());

    let max = dataset[n - 1];
    let min = dataset[0];

    if max == min {
        return MIN_BIN_CLAMP;
    }

    (((max - min) / bin_width)
        .ceil()
        .to_usize()
        .unwrap_or(MIN_BIN_CLAMP))
    .max(MIN_BIN_CLAMP)
}

/// Compute the optimal number of bins using Sturges method.
fn sturges<T: Float>(dataset: &[T]) -> usize {
    ((dataset.len() as f64).log2().floor() as usize + 1_usize).max(MIN_BIN_CLAMP)
}

/// Compute the optimal number of bins using Freedman Diaconis method.
/// Assumes data is sorted.
fn freedman_diaconis<T: Float>(sorted_data: &[T]) -> usize {
    let n = sorted_data.len();
    let n_t = T::from(n).unwrap();
    if n == 1 {
        return 1;
    }

    let p75 = sorted_data[((T::from(0.75_f32).unwrap() * n_t)
        .floor()
        .to_usize()
        .unwrap_or(0))
    .min(sorted_data.len() - 1)];
    let p25 = sorted_data[(T::from(0.25_f32).unwrap() * n_t)
        .floor()
        .to_usize()
        .unwrap_or(0)];
    let iqr = p75 - p25;

    if iqr == T::zero() {
        return MIN_BIN_CLAMP;
    }

    let width = T::from(2_f32).unwrap() * iqr * n_t.powf(T::from(-1_f32 / 3_f32).unwrap());
    let max = sorted_data[n - 1];
    let min = sorted_data[0];
    // clamp to [3, ...)
    (((max - min) / width)
        .ceil()
        .to_usize()
        .unwrap_or(MIN_BIN_CLAMP))
    .max(MIN_BIN_CLAMP)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn quantile_type_try_from_valid() {
        assert!(matches!(
            QuantileType::try_from("FreedmanDiaconis").unwrap(),
            QuantileType::FreedmanDiaconis
        ));
        assert!(matches!(
            QuantileType::try_from("Scott").unwrap(),
            QuantileType::Scott
        ));
        assert!(matches!(
            QuantileType::try_from("Sturges").unwrap(),
            QuantileType::Sturges
        ));
    }

    #[test]
    fn quantile_type_try_from_unknown_returns_err() {
        assert!(QuantileType::try_from("unknown").is_err());
        assert!(QuantileType::try_from("").is_err());
        assert!(QuantileType::try_from("freedmandiaconis").is_err()); // case sensitive
    }

    #[test]
    fn sturges_known_value() {
        // n=8: log2(8).floor() + 1 = 3 + 1 = 4
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        assert_eq!(sturges(&data), 4);
    }

    #[test]
    fn sturges_minimum_clamp() {
        // n=2: log2(2).floor() + 1 = 2, clamped to MIN_BIN_CLAMP=3
        let data = vec![0.0_f64, 1.0];
        assert_eq!(sturges(&data), MIN_BIN_CLAMP);
    }

    #[test]
    fn scott_constant_data_returns_clamp() {
        let data = vec![5.0_f64; 20];
        assert_eq!(scott(&data), MIN_BIN_CLAMP);
    }

    #[test]
    fn scott_known_value() {
        // n=50, uniform [0..49]
        // σ = sqrt(sum((i-24.5)^2) / 49) ≈ 14.577
        // bin_width = 3.49 * 14.577 * 50^(-1/3) ≈ 13.81
        // ceil(49 / 13.81) = 4
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert_eq!(scott(&data), 4);
    }

    #[test]
    fn scott_result_at_least_min_clamp() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert!(scott(&data) >= MIN_BIN_CLAMP);
    }

    #[test]
    fn freedman_diaconis_single_element_returns_one() {
        assert_eq!(freedman_diaconis(&[42.0_f64]), 1);
    }

    #[test]
    fn freedman_diaconis_zero_iqr_returns_clamp() {
        // more than half the values are the same → IQR = 0
        let data = vec![1.0_f64, 1.0, 1.0, 1.0, 2.0];
        assert_eq!(freedman_diaconis(&data), MIN_BIN_CLAMP);
    }

    #[test]
    fn freedman_diaconis_known_value() {
        // n=100, uniform [0..99]
        // p25=25, p75=75, iqr=50
        // width = 2 * 50 * 100^(-1/3) ≈ 21.54
        // ceil(99 / 21.54) = 5
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(freedman_diaconis(&data), 5);
    }

    #[test]
    fn freedman_diaconis_result_at_least_min_clamp() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(freedman_diaconis(&data) >= MIN_BIN_CLAMP);
    }
}
