//! Probability distribution functions

use std::convert::{TryFrom, TryInto};
use std::error::Error;
use std::f64::consts;
use std::fmt;
use rand::prelude::*;
use rand_distr::StandardNormal;
use crate::geometry::ThreeVector;
use crate::pwmci::Interpolant;
use crate::special_functions::*;

/// Represents the distribution of the particles' spatial coordinates.
#[derive(Copy, Clone, Debug)]
pub enum SpatialDistribution {
    /// Gaussian (normal) distribution with std dev `sigma`,
    /// optionally truncated at `max`, in one dimension.
    Normal { sigma: f64, max: Option<f64>, dim: i32 },
    /// A hard-sphere distribution, `0 <= r < max`, in the given
    /// number of dimensions
    Disk { max: f64, dim: i32 },
}

impl fmt::Display for SpatialDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpatialDistribution::Normal { sigma, max, dim } => {
                let trunc = if max.is_some() {"trunc "} else {""};
                write!(f, "SpatialDistribution ({}D): {}normal, std. dev. = {:.3e}", dim, trunc, sigma)
            },
            SpatialDistribution::Disk { max, dim } => {
                write!(f, "SpatialDistribution ({}D): uniform, bounds = ±{:.3e}", dim, 0.5 * max)
            }
        }
    }
}

impl SpatialDistribution {
    /// Supply with a uniformly distributed random variate x ~ U(0, 1).
    pub fn sample(&self, rng: f64) -> f64 {
        match self {
            Self::Normal { sigma, dim, max } => {
                match dim {
                    1 => {
                        let multiplier = if let Some(max) = max {
                            (max / (consts::SQRT_2 * sigma)).erf()
                        } else {
                            1.0
                        };
                        consts::SQRT_2 * sigma * ((2.0 * rng - 1.0) * multiplier).inv_erf()
                    },
                    2 => {
                        let multiplier = if let Some(max) = max {
                            let arg = -max * max / (2.0 * sigma * sigma);
                            1.0 - arg.exp()
                        } else {
                            1.0
                        };
                        let u = multiplier * rng;
                        sigma * (-2.0 * (1.0 - u).ln()).sqrt()
                    },
                    _ => unreachable!(),
                }
            },

            Self::Disk { max, dim } => {
                match dim {
                    1 => max * (2.0 * rng - 1.0),
                    2 => max * rng.sqrt(),
                    _ => unreachable!(),
                }
            },
        }
    }

    pub fn normal(sigma: f64, max: Option<f64>, dim: i32) -> Self {
        Self::Normal { sigma, max, dim }
    }

    pub fn uniform(max: f64, dim: i32) -> Self {
        Self::Disk { max, dim }
    }

    pub fn std_dev(&self) -> f64 {
        match self {
            SpatialDistribution::Normal { sigma, max, dim: _ } => {
                if let Some(max) = max {
                    let arg = max / (consts::SQRT_2 * sigma);
                    let var = sigma * (sigma - (2.0 / consts::PI).sqrt() * max * (-arg * arg).exp() / arg.erf());
                    var.sqrt()
                } else {
                    *sigma
                }
            },
            SpatialDistribution::Disk { max, dim: _ } => max / 3_f64.sqrt(),
        }
    }
}

pub enum DistributionError {
    NonNumerical,
    Positivity,
    Length,
    InterpolationOrder,
}

impl fmt::Debug for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DistributionError::NonNumerical => write!(f, "the distribution contains non-numerical values"),
            DistributionError::Positivity => write!(f, "the distribution contains negative values"),
            DistributionError::Length => write!(f, "the number of points in distribution < 2"),
            DistributionError::InterpolationOrder => write!(f, "the interpolation order is not one or two"),
        }
    }
}

impl fmt::Display for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for DistributionError {}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum InterpolationOrder {
    Linear = 1,
    Quadratic = 2,
}

impl TryFrom<i32> for InterpolationOrder {
    type Error = DistributionError;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(InterpolationOrder::Linear),
            2 => Ok(InterpolationOrder::Quadratic),
            _ => Err(DistributionError::InterpolationOrder)
        }
    }
}

/// Represents the distribution of Lorentz factors (i.e. energy divided by
/// the mass) in a particle beam
#[derive(Clone, Debug)]
pub enum GammaDistribution {
    /// Normally distributed
    Normal {
        mu: f64,
        sigma: f64,
        rho: f64,
    },
    /// Spectrum arising from incoherent bremsstrahlung
    Brem {
        min: f64,
        max: f64,
    },
    /// Arbitrary function
    Custom {
        vals: Vec<f64>,
        /// Guaranteed to satisfy 0 <= cdf <= 1.
        cdf: Vec<[f64; 2]>,
        min: f64,
        max: f64,
        step: f64,
        rho: f64,
        order: InterpolationOrder,
    }
}

impl GammaDistribution {
    pub fn normal(mean_gamma: f64, sigma: f64) -> Self {
        Self::Normal { mu: mean_gamma, sigma, rho: 0.0 }
    }

    pub fn from_brem_source(min_gamma: f64, max_gamma: f64) -> Self {
        Self::Brem { min: min_gamma, max: max_gamma }
    }

    pub fn custom(vals: Vec<f64>, min: f64, max: f64, step: f64, order: i32) -> Result<Self, DistributionError> {
        let order: InterpolationOrder = order.try_into()?;

        if vals.iter().any(|v| !v.is_finite()) {
            Err(DistributionError::NonNumerical)
        } else if vals.iter().any(|v| *v < 0.0) {
            Err(DistributionError::Positivity)
        } else if vals.len() < 2 {
            Err(DistributionError::Length)
        } else {
            let mut cdf: Vec<[f64; 2]> = vec![];
            let mut gamma = min;
            let mut rt = 0.0;
            cdf.push([gamma, rt]);

            for y in vals.windows(2) {
                match order {
                    InterpolationOrder::Linear => {
                        let mid = rt + 0.125 * (3.0 * y[0] + y[1]);
                        cdf.push([gamma + 0.5 * step, mid]);
                        gamma += step;
                        rt += 0.5 * (y[0] + y[1]);
                        cdf.push([gamma, rt]);
                    },
                    InterpolationOrder::Quadratic => {
                        gamma += step;
                        rt += 0.5 * (y[0] + y[1]);
                        cdf.push([gamma, rt]);
                    },
                }
            }

            for entry in cdf.iter_mut() {
                entry[1] /= rt;
            }

            Ok(Self::Custom { vals, cdf, min, max, step, rho: 0.0, order })
        }
    }

    /// Sets the value of the correlation coefficient
    pub fn with_correlation_coeff(self, rho: f64) -> Self {
        match self {
            GammaDistribution::Normal { mu, sigma, rho: _ } => {
                GammaDistribution::Normal { mu, sigma, rho }
            },
            GammaDistribution::Custom { vals, cdf, min, max, step, rho: _, order } => {
                GammaDistribution::Custom { vals, cdf, min, max, step, rho, order }
            }
            _ => self
        }
    }

    pub fn gamma(&self) -> f64 {
        match self {
            Self::Normal { mu, sigma: _, rho: _ } => *mu,
            Self::Brem { min: _, max } => *max,
            Self::Custom { vals, cdf: _, min, max: _, step, rho: _, order: _ } => {
                let moments = vals.iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let g = min + (i as f64) * step;
                        ThreeVector::new(*f, g * f, g * g * f)
                    })
                    .fold(
                        ThreeVector::new(0.0, 0.0, 0.0),
                        |acc, x| acc + x
                    );

                moments[1] / moments[0]
            },
        }
    }

    #[cfg(feature = "hdf5-output")]
    pub fn std_dev(&self) -> f64 {
        match self {
            Self::Normal { mu: _, sigma, rho: _ } => *sigma,
            // approximation that works for min / max > 0.2
            Self::Brem { min, max } => 0.5 * (max - min) / 3_f64.sqrt(),
            Self::Custom { vals, cdf: _, min, max: _, step, rho: _ , order: _} => {
                let moments = vals.iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let g = min + (i as f64) * step;
                        ThreeVector::new(*f, g * f, g * g * f)
                    })
                    .fold(
                        ThreeVector::new(0.0, 0.0, 0.0),
                        |acc, x| acc + x
                    );

                (moments[2] / moments[0] - (moments[1] / moments[0]).powi(2)).sqrt()
            },
        }
    }

    #[cfg(feature = "hdf5-output")]
    pub fn min_gamma(&self) -> f64 {
        match self {
            Self::Normal { mu, sigma, rho: _ } => mu - 3.0 * sigma,
            Self::Brem { min, max: _ } => *min,
            Self::Custom { vals: _, cdf: _, min, max: _, step: _, rho: _, order: _ } => *min,
        }
    }

    /// Returns a tuple of the sampled gamma and z
    pub fn sample<R: Rng>(&self, z_dstr: &SpatialDistribution, rng: &mut R) -> (f64, f64) {
        match self {
            Self::Normal { mu, sigma, rho } => {
                loop {
                    // for correlated gamma and z
                    let n0 = rng.sample::<f64,_>(StandardNormal);
                    let n1 = rng.sample::<f64,_>(StandardNormal);
                    let n2 = rho * n0 + (1.0 - rho * rho).sqrt() * n1;

                    let dz = {
                        // NORTA method: transform from N(0,1) to U(0,1)
                        let u = 0.5 * (1.0 + (n0 / consts::SQRT_2).erf());
                        z_dstr.sample(u)
                    };

                    let gamma = mu + sigma * n2;

                    if gamma > 1.0 {
                        break (gamma, dz);
                    }
                }
            },

            Self::Brem { min, max } => {
                let x_min = min / max;
                let y_max = 4.0 / (3.0 * x_min) - 4.0 / 3.0 + x_min;
                let x = loop {
                    let x = x_min + (1.0 - x_min) * rng.gen::<f64>();
                    let u = rng.gen::<f64>();
                    let y = 4.0 / (3.0 * x) - 4.0 / 3.0 + x;
                    if u <= y / y_max {
                        break x;
                    }
                };

                let dz = z_dstr.sample(rng.gen());
                (x * max, dz)
            },

            Self::Custom { vals: _, cdf, min: _, max: _, step: _, rho, order} => {
                // Correlated variables
                let n0 = rng.sample::<f64,_>(StandardNormal);
                let n1 = {
                    let n1 = rng.sample::<f64,_>(StandardNormal);
                    rho * n0 + (1.0 - rho * rho).sqrt() * n1
                };

                // NORTA method: transform from N(0,1) to U(0,1)
                let u1 = 0.5 * (1.0 + (n1 / consts::SQRT_2).erf());
                let u1 = u1.clamp(0.0, 1.0);

                let gamma = match order {
                    InterpolationOrder::Linear => {
                        let index = cdf.iter().step_by(2).position(|[_, y]| u1 < *y).unwrap();
                        let index = 2 * index;
                        let [x0, c0] = cdf[index-2];
                        let [_, c1] = cdf[index-1];
                        let [x2, c2] = cdf[index];
                        let d = x2 - x0; // x1 is midpoint
                        // cdf = alpha * gamma^2 + beta * gamma + charl
                        let alpha = 2.0 * (c0 - 2.0 * c1 + c2) / (d * d);
                        let bravo = -(3.0 * c0 - 4.0 * c1 + c2) / d - 4.0 * (c0 - 2.0 * c1 + c2) * x0 / (d * d);
                        let charl = c0 + (3.0 * c0 - 4.0 * c1 + c2) * x0 / d + 2.0 * (c0 - 2.0 * c1 + c2) * x0 * x0 / (d * d);
                        // solve u1 = cdf
                        let charl = charl - u1;
                        let det = bravo * bravo - 4.0 * alpha * charl;
                        let gamma = if alpha != 0.0 && det >= 0.0 {
                            (-bravo + det.sqrt()) / (2.0 * alpha)
                        } else {
                            // fall back to linear
                            x0 + (u1 - c0) * (x2 - x0) / (c2 - c0)
                        };
                        gamma
                    },
                    InterpolationOrder::Quadratic => {
                        Interpolant::new(cdf).invert(u1).unwrap()
                    },
                };

                let dz = {
                    // NORTA method: transform from N(0,1) to U(0,1)
                    let u = 0.5 * (1.0 + (n0 / consts::SQRT_2).erf());
                    z_dstr.sample(u)
                };

                (gamma, dz)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_xoshiro::*;
    use super::*;

    #[test]
    fn correlated_dstr() {
        let beam_len = 5.0e-6;

        let z_dstrs = [
            SpatialDistribution::normal(beam_len, None, 1),
            SpatialDistribution::normal(beam_len, Some(beam_len), 1),
            SpatialDistribution::uniform(beam_len, 1),
        ];

        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for z_dstr in z_dstrs.iter() {
            let target_sigma = 100.0;
            let target_rho = 0.5;
            let e_dstr = GammaDistribution::normal(1000.0, target_sigma)
                .with_correlation_coeff(target_rho);

            let mut avg_z = 0.0;
            let mut avg_g = 0.0;
            let mut avg_z_sqd = 0.0;
            let mut avg_g_sqd = 0.0;
            let mut avg_zg = 0.0;
            let n = 1_000_000;
            let w = 1.0 / (n as f64);

            for _i in 0..n {
                let (g, z) = e_dstr.sample(z_dstr, &mut rng);
                avg_z += w * z;
                avg_g += w * g;
                avg_z_sqd += w * z * z;
                avg_g_sqd += w * g * g;
                avg_zg += w * z * g;
            }

            let var_z = avg_z_sqd - avg_z * avg_z;
            let var_g = avg_g_sqd - avg_g * avg_g;
            let rho = (avg_zg - avg_z * avg_g) / (var_z * var_g).sqrt();

            println!("z [{}] vs gamma [{:?}]:", z_dstr, e_dstr);

            let target_rms_z = z_dstr.std_dev();
            let err = (target_rms_z - var_z.sqrt()).abs() / target_rms_z;
            println!("\tgot std. dev. z = {:.3e} [{:.3e}], err = {:.3}%", var_z.sqrt(), target_rms_z, 100.0 * err);
            assert!(err < 1.0e-2);

            let err = (target_sigma - var_g.sqrt()).abs() / target_sigma;
            println!("\tgot std. dev. γ = {:.3e} [{:.3e}], err = {:.3}%", var_g.sqrt(), target_sigma, 100.0 * err);
            assert!(err < 1.0e-2);

            let err = (target_rho - rho).abs() / target_rho;
            println!("\tgot rho = {:.3e} [{:.3e}], err = {:.3}%", rho, target_rho, 100.0 * err);
            assert!(err < 5.0e-2);
        }
    }
}