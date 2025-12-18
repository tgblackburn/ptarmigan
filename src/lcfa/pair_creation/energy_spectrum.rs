use crate::quadrature::{GL_NODES, GL_WEIGHTS};

pub struct Spectrum {
    chi: f64,
    sv1: f64,
}

impl Spectrum {
    pub fn new(chi: f64, sv1: f64) -> Self {
        Self { chi, sv1 }
    }

    /// Proportional to the probability spectrum dW/ds for a photon
    /// with quantum parameter `chi` and Stokes parameter `sv1`.
    ///
    /// The rate is obtained by integrating [spectrum] over s and multiplying
    /// by ɑ m^2 / (√3 π ω).
    pub fn value(&self, s: f64) -> f64 {
        GL_NODES.iter()
            .zip(GL_WEIGHTS.iter())
            .map(|(t, w)| {
                let xi = 2.0 / (3.0 * self.chi * s * (1.0 - s));
                let prefactor = (-xi * t.cosh() + t).exp();
                w * prefactor * ((1.0 / (s * (1.0 - s)) - self.sv1 - 2.0) * (2.0 * t / 3.0).cosh() + (t / 3.0).cosh() / t.cosh())
            })
            .sum()
    }

    /// Returns the maximum value of [spectrum] for a polarized photon,
    /// padded by a small safety margin.
    pub fn ceiling(&self) -> f64 {
        let chi_switch = ((1.5 - self.sv1) / 3_f64.sqrt()).exp();

        let max = if self.chi < chi_switch {
            self.value(0.5)
        } else if self.chi > 100.0 {
            self.value(4.0 / (3.0 * self.chi))
        } else {
            let m = -0.94866 + 0.170159 * self.sv1;
            let s = 0.5 * chi_switch.powf(-m) * self.chi.powf(m);
            self.value(s)
        };

        1.05 * max
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_xoshiro::*;
    use super::*;

    #[test]
    fn pair_spectrum_ceiling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _i in 0..100 {
            let chi = (1_f64.ln() + (100_f64.ln() - 1_f64.ln()) * rng.gen::<f64>()).exp();
            let sv1 = -1.0 + 2.0 * rng.gen::<f64>();
            let spectrum = Spectrum::new(chi, sv1);

            let target: f64 = (0..10_000)
                .map(|i| 0.5 * (i as f64) / 10000.0)
                .map(|s| spectrum.value(s))
                .reduce(f64::max)
                .unwrap();

            let result = spectrum.ceiling();

            let err = (target - result) / target;

            println!(
                "chi = {:>9.3e}, ξ_1 = {:>6.3} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                chi, sv1, target, result, 100.0 * err,
            );

            assert!(err < 0.0);
        }
    }
}