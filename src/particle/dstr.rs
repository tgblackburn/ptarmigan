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