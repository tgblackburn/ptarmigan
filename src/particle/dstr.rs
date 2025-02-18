//! Probability distribution functions

use std::f64::consts;
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
        cdf: Vec<[f64; 2]>,
        min: f64,
        max: f64,
        step: f64,
        rho: f64,
    }
}

impl GammaDistribution {
    pub fn normal(mean_gamma: f64, sigma: f64) -> Self {
        Self::Normal { mu: mean_gamma, sigma, rho: 0.0 }
    }

    pub fn from_brem_source(min_gamma: f64, max_gamma: f64) -> Self {
        Self::Brem { min: min_gamma, max: max_gamma }
    }

    pub fn custom(vals: Vec<f64>, min: f64, max: f64, step: f64) -> Option<Self> {
        if vals.iter().any(|v| !v.is_finite()) {
            None
        } else {
            let total: f64 = vals.iter().sum();

            let mut cdf: Vec<[f64; 2]> = vec![];
            cdf.push([min, 0.0]);
            let mut gamma = min;
            let mut rt = 0.0;
            for v in vals.iter() {
                gamma += step;
                rt += v;
                cdf.push([gamma, rt / total]);
            }

            Some(Self::Custom { vals, cdf, min, max, step, rho: 0.0 })
        }
    }

    pub fn gamma(&self) -> f64 {
        match self {
            Self::Normal { mu, sigma: _, rho: _ } => *mu,
            Self::Brem { min: _, max } => *max,
            Self::Custom { vals, cdf: _, min, max: _, step, rho: _ } => {
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
            Self::Custom { vals, cdf: _, min, max: _, step, rho: _ } => {
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
            Self::Custom { vals: _, cdf: _, min, max: _, step: _, rho: _ } => *min,
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

            Self::Custom { vals: _, cdf, min: _, max: _, step: _, rho} => {
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

                let dz = sigma_z * n0;
                let gamma = Interpolant::new(cdf).invert(u1).unwrap();

                (gamma, dz)
            },
        }
    }
}