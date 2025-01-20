//! Probability distribution functions

use std::f64::consts;
use rand::prelude::*;
use rand_distr::StandardNormal;

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

#[derive(Copy, Clone)]
pub(super) enum GammaDistribution<'a> {
    Normal {
        mu: f64,
        sigma: f64,
        rho: f64,
    },
    Brem {
        min: f64,
        max: f64,
    },
    #[allow(unused)]
    Analytical {
        func: &'a dyn Fn(f64) -> f64,
        min: f64,
        max: f64,
    }
}

impl<'a> GammaDistribution<'a> {
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

            Self::Analytical { func, min, max } => {
                let gamma = loop {
                    let x = min + (max - min) * rng.gen::<f64>();
                    let u = rng.gen::<f64>();
                    let y = func(x);
                    if u <= y {
                        break x;
                    }
                };
                let dz = sigma_z * rng.sample::<f64,_>(StandardNormal);
                (gamma, dz)
            },
        }
    }
}