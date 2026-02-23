//! Probability distribution functions

use std::convert::{TryFrom, TryInto};
use std::error::Error;
use std::f64::consts;
use std::fmt;
use rand::prelude::*;
use rand_distr::StandardNormal;
use crate::geometry::ThreeVector;
use crate::pwmci::Interpolant;

#[derive(Copy, Clone)]
pub(super) enum RadialDistribution {
    Normal {
        sigma_x: f64,
        sigma_y: f64,
    },
    TruncNormal {
        sigma_x: f64,
        sigma_y: f64,
        x_max: f64,
        y_max: f64,
    },
    Uniform {
        r_max: f64,
    },
}

impl RadialDistribution {
    pub fn sample<R: Rng>(&self, rng: &mut R) -> (f64, f64) {
        match self {
            Self::Normal { sigma_x, sigma_y } => {(
                sigma_x * rng.sample::<f64,_>(StandardNormal),
                sigma_y * rng.sample::<f64,_>(StandardNormal),
            )},

            Self::TruncNormal {sigma_x, sigma_y, x_max, y_max} => {
                loop {
                    let x = sigma_x * rng.sample::<f64,_>(StandardNormal);
                    let y = sigma_y * rng.sample::<f64,_>(StandardNormal);
                    if x * x / (x_max * x_max) + y * y / (y_max * y_max) <= 1.0 {
                        return (x, y);
                    }
                }
            },

            Self::Uniform {r_max} => {
                let r = r_max * rng.gen::<f64>().sqrt();
                let theta = 2.0 * consts::PI * rng.gen::<f64>();
                (r * theta.cos(), r * theta.sin())
            },
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

#[derive(Copy, Clone)]
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
#[derive(Clone)]
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

    pub fn sample<R: Rng>(&self, sigma_z: f64, rng: &mut R) -> (f64, f64) {
        match self {
            Self::Normal { mu, sigma, rho } => {
                loop {
                    // for correlated gamma and z
                    let n0 = rng.sample::<f64,_>(StandardNormal);
                    let n1 = rng.sample::<f64,_>(StandardNormal);
                    let n2 = rho * n0 + (1.0 - rho * rho).sqrt() * n1;

                    let dz = sigma_z * n0;
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
                let dz = sigma_z * rng.sample::<f64,_>(StandardNormal);
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
                let u1 = {
                    let x = n1 / consts::SQRT_2;
                    let erf_x = (167.0 * x / 148.0 + 11.0 * x.powi(3) / 109.0).tanh();
                    0.5 * (1.0 + erf_x)
                };

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

                let dz = sigma_z * n0;

                (gamma, dz)
            },
        }
    }
}