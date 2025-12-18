//! Handling systematic uncertainties

use rand::prelude::*;

#[derive(Copy, Clone)]
pub enum Uncertainty {
    #[allow(unused)]
    Fixed { value: f64 },
    Between { min: f64, max: f64 },
    None,
}

impl Uncertainty {
    pub fn range(&self) -> Option<f64> {
        match self {
            Uncertainty::Fixed { .. } => None,
            Uncertainty::Between { min, max } => Some(0.5 * (max - min)),
            Uncertainty::None => None,
        }
    }

    pub fn is_some(&self) -> bool {
        match self {
            Uncertainty::Fixed { .. } => true,
            Uncertainty::Between { .. } => true,
            Uncertainty::None => false,
        }
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            Uncertainty::Fixed { value } => *value,
            Uncertainty::Between { min, max } => {
                if rng.gen::<bool>() {
                    0.5 * (min + max)
                } else if rng.gen::<bool>() {
                    *min
                } else {
                    *max
                }
            },
            Uncertainty::None => 0.0,
        }
    }
}

impl Default for Uncertainty {
    fn default() -> Self {
        Uncertainty::Between { min: -1.0, max: 1.0 }
    }
}