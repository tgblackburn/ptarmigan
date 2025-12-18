use std::f64::consts;
use rand::prelude::*;
use crate::geometry::StokesVector;

pub struct Spectrum {
    s: f64,
    chi: f64,
    sv: StokesVector,
}

impl Spectrum {
    pub fn new(s: f64, chi: f64, sv: StokesVector) -> Self {
        Self { s, chi, sv }
    }

    /// Proportional to the angularly resolved spectrum d^2 W/(ds dy),
    /// where z = [2ɣ^2(1 - β cosθ)]^(3/2) = 1 + 4 chi y^2.
    /// The domain of interest is 0 < y < 1.
    pub fn polar(&self, y: f64) -> f64 {
        // The spectrum is given by
        // dW/(ds dy) = y [1 + z^(2/3) (s/(1-s) + (1-s)/s - sv1)] K_{1/3}(xi z)
        // where z = 1 + 4 chi y^2, xi = 2 / [3 chi s (1-s)]
        // In principle, 1 < z < infty, but dominated by 1 < z < 1 + 4 chi
        use crate::special_functions::*;
        let xi = 2.0 / (3.0 * self.chi * self.s * (1.0 - self.s));
        let prefactor = self.s / (1.0 - self.s) + (1.0 - self.s) / self.s - self.sv[1];
        let z = 1.0 + 4.0 * self.chi * y * y;
        y * (1.0 + prefactor * z.powf(2.0 / 3.0)) * (xi * z).bessel_K_1_3().unwrap_or(0.0)
    }

    /// Returns the maximum value of [polar] for a polarized photon,
    /// padded by a small safety margin.
    pub fn polar_ceiling(&self) -> f64 {
        // y that maximises the spectrum, assuming s = 1/2:
        let y_peak = {
            let y_min = 0.216;
            let y_max = 0.259;
            y_min + (y_max - y_min) * (1.0 - (8.3 / self.chi).powf(2.0/3.0).tanh())
        };

        // and for general s:
        let y = y_peak * (1.0 - (1.0 - 2.0 * self.s).powi(2)).sqrt();

        1.05 * self.polar(y)
    }

    pub fn sample_azimuthal<R: Rng>(&self, z: f64, rng: &mut R) -> f64 {
        let arg = 2.0 * z / (3.0 * self.chi * self.s * (1.0 - self.s));
        // ratio of K_{2/3}(arg) / K_{1/3}(arg)
        let k_ratio = if arg < 1.0e-4 {
            0.6368498843179743 / arg.cbrt()
        } else {
            1.0 + 1.0 / (1.4624087952220928 * arg.cbrt() + 1.023821552056939 * arg.sqrt() + 6.0 * arg)
        };
        let a = 1.0 + z.powf(2.0/3.0) * (self.s.powi(2) + (1.0 - self.s).powi(2)) / (self.s * (1.0 - self.s));
        let b = 1.0;
        let c = z.powf(2.0/3.0);
        let d = z.powf(2.0/3.0) - 1.0;
        let e = z.powf(1.0/3.0) * (z.powf(2.0/3.0) - 1.0) * (self.s.powi(2) + (1.0 - self.s).powi(2)) * k_ratio / (self.s * (1.0 - self.s));

        fn azimuthal_spectrum(phi: f64, a: f64, b: f64, c: f64, d: f64, e: f64, sv: StokesVector) -> f64 {
            a + b * ((2.0 * phi).cos() * (1.0 - c) - c) * sv[1] - d * (2.0 * phi).sin() * sv[2] + e * phi.sin() * sv[3]
        }

        let max = (0..32)
            .map(|i| azimuthal_spectrum(2.0 * consts::PI * (i as f64) / 32.0, a, b, c, d, e, self.sv))
            .reduce(f64::max)
            .map(|y| 1.1 * y)
            .unwrap();

        loop {
            let phi = 2.0 * consts::PI * rng.gen::<f64>();
            let u = rng.gen::<f64>();
            let f = azimuthal_spectrum(phi, a, b, c, d, e, self.sv);
            if u <= f / max {
                break phi;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_xoshiro::*;
    use super::*;

    #[test]
    fn pair_angular_spectrum_ceiling() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _i in 0..1000 {
            let chi = (0.1_f64.ln() + (100_f64.ln() - 1_f64.ln()) * rng.gen::<f64>()).exp();
            let sv1 = -1.0 + 2.0 * rng.gen::<f64>();
            let s = 0.5 * rng.gen::<f64>();

            let spectrum = Spectrum::new(s, chi, [1.0, sv1, 0.0, 0.0].into());

            let target: f64 = (0..100)
                .map(|i| 0.5 * (i as f64) / 100.0) // search in 0 < y < 0.5
                .map(|y| spectrum.polar(y))
                .reduce(f64::max)
                .unwrap();

            let result = spectrum.polar_ceiling();

            let err = (target - result) / target;

            println!(
                "chi = {:>9.3e}, ξ_1 = {:>6.3}, s = {:.3} => max = {:>9.3e}, predicted = {:>9.3e}, err = {:.2}%",
                chi, sv1, s, target, result, 100.0 * err,
            );

            assert!(err < 0.0 || target < 1.0e-200);
        }
    }

}