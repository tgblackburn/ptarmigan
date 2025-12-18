//! Evaluate and sample from the angularly integrated energy spectrum

use std::f64::consts;
use rand::prelude::*;
use crate::quadrature;

pub struct Spectrum {
    chi: f64,
}

impl Spectrum {
    pub fn at(chi: f64) -> Self {
        Self { chi }
    }

    /// Returns the differential spectrum, dW/df, at given chi.
    /// Multiply by alpha / (sqrt(3) pi gamma t_c) to get the rate
    /// per unit f per unit lab time.
    pub fn value(&self, f: f64) -> f64 {
        let xi = 2.0 * f / (3.0 * self.chi * (1.0 - f));
        let p = 1.0 - f + 1.0 / (1.0 - f);
        if xi < 0.01 {
            let xi_2_3 = xi.powf(2.0 / 3.0);
            p * (1.07476412077 / xi_2_3 - 0.5 * 2.53143828844 * xi_2_3) - consts::PI / 3_f64.sqrt() + 2.53143828844 * xi_2_3
        } else if xi > 100.0 {
            let a = (-xi).exp() * (consts::FRAC_PI_2 / xi).sqrt();
            a * (p * (1.0 + 7.0 / (72.0 * xi)) - 1.0 + 41.0 / (72.0 * xi))
        } else {
            // Express p K_{2/3}(xi) - ∫ K_{1/3}(t) dt as a single integral
            let x_max = 3.0 / xi.cbrt();
            quadrature::GAUSS_16_NODES.iter()
                .zip(quadrature::GAUSS_16_WEIGHTS.iter()) // -1 < t < 1
                .map(|(t, w)| {
                    let x = 0.5 * x_max * (1.0 + t);
                    let a = (3.0 + 2.0 * x * x) / (1.0 + x * x / 3.0).sqrt();
                    let b = (1.0 + 4.0 * x * x / 3.0) * (1.0 + x * x / 3.0).sqrt();
                    let fx = (p * a / 3_f64.sqrt() - 3_f64.sqrt() / b) * (-xi * b).exp();
                    0.5 * x_max * w * fx
                })
                .sum()
        }
    }

    /// Returns a value for f, u = f/(1-f) and the number of attempts required
    #[allow(unused)]
    pub fn sample<R: Rng>(&self, rng: &mut R) -> (f64, f64, i32) {
        let mut n = 0;
        // If f ~ dW/df, y = 3f^(1/3) ~ f^(2/3) dW/df, which cancels
        // the divergence at f -> 0

        if self.chi < 1.0 {
            // maximum is at f = y = 0
            let max = 1.35411793943 * (3.0 * self.chi).powf(2.0 / 3.0);
            // power is maximised here
            let y0 = 3.0 * (0.429 * self.chi).cbrt();
            loop {
                n += 1;
                // Generate y from a better proposal distribution
                let y_below_y0 = rng.gen::<f64>() < 2.0 * y0 / (y0 + 3.0);

                let (y, z1) = if y_below_y0 {
                    (y0 * rng.gen::<f64>(), max * rng.gen::<f64>())
                } else {
                    let r: f64 = rng.gen();
                    let y = 3.0 + (y0 - 3.0) * (1.0 - r).sqrt();
                    (y, max * (y - 3.0) / (y0 - 3.0) * rng.gen::<f64>())
                };

                // Test against target distribution
                let f = (y / 3.0).powi(3);
                let z2 = (y / 3.0).powi(2) * self.value(f);

                if z1 < z2 {
                    break (f, f / (1.0 - f), n);
                }
            }
        } else if self.chi < 30.6702 {
            let max = 1.35411793943 * (3.0 * self.chi).powf(2.0 / 3.0);
            loop {
                n += 1;
                let y = 3.0 * rng.gen::<f64>();
                let f = (y / 3.0).powi(3);
                let z1 = max * rng.gen::<f64>();
                let z2 = (y / 3.0).powi(2) * self.value(f);
                if z1 < z2 {
                    break (f, f / (1.0 - f), n);
                }
            }
        } else {
            // peaked at f = 1 - 4/(3 chi)
            let f_peak = 1.0 - 4.0 / (3.0 * self.chi);
            let upper_max = f_peak.powf(2.0 / 3.0) * self.value(f_peak);
            let lower_max = 1.35411793943 * (3.0 * self.chi).powf(2.0 / 3.0);
            let y0 = 2.8; // spectrum > lower_max if y0 < y < 3

            loop {
                n += 1;

                let y_below_y0 = rng.gen::<f64>() < 1.0 / (1.0 + (upper_max * (3.0 - y0)) / (lower_max * y0));

                let (y, z1) = if y_below_y0 {
                    (y0 * rng.gen::<f64>(), lower_max * rng.gen::<f64>())
                } else {
                    (y0 + (3.0 - y0) * rng.gen::<f64>(), upper_max * rng.gen::<f64>())
                };

                let f = (y / 3.0).powi(3);
                let z2 = (y / 3.0).powi(2) * self.value(f);

                if z1 < z2 {
                    break (f, f / (1.0 - f), n);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalised_spectrum_test() {
        let data = [
            (1.0e-3, 0.02, 18.93015363110),
            (0.1,    0.02, 0.0187963720759),
            (0.7,    0.02, 6.26040872624e-35),
            (0.9,    0.02, 3.391893114131443e-131),
            (1.0e-3, 0.1, 58.82978028823),
            (0.1,    0.1, 1.047213583232),
            (0.7,    0.1, 1.499650511606e-7),
            (0.99,   0.1, 1.1211851106259e-286),
            (1.0e-3, 1.0, 279.66643712355),
            (0.1,    1.0, 10.45117481790),
            (0.7,    1.0, 0.641365473143),
            (0.99,   1.0, 3.3205461849919e-28),
            (1.0e-3, 40.0, 3290.39874403077),
            (0.1,    40.0, 141.51871069391),
            (0.7,    40.0, 31.99151529936),
            (0.99,   40.0, 19.492333426454),
            (1.0e-3, 200.0, 9624.67407484370),
            (0.1,    200.0, 417.29328709328),
            (0.7,    200.0, 97.58287038101),
            (0.99,   200.0, 181.345501003872),
            (1.0e-3, 3000.0, 58548.42995994882),
            (0.1,    3000.0, 2547.28052138260),
            (0.7,    3000.0, 603.20104280620),
            (0.99,   3000.0, 1357.94983969650),
        ];

        for (f, chi, target) in data.iter() {
            let spectrum = Spectrum::at(*chi);
            let value = spectrum.value(*f);
            let error = (target - value).abs() / target;
            let xi = 2.0 * f / (3.0 * chi *  (1.0 - f));
            println!(
                "f = {:.2e}, chi = {:.2e}, xi = {:.3e}: dW/df = {:.6e}, err = {:.2e}",
                f, chi, xi, value, error
            );
            assert!(error < 1.0e-3);
        }
    }

    #[test]
    fn rejection_sampling() {
        let chi = 0.1;
        let spectrum = Spectrum::at(chi);
        let mut rng = thread_rng();
        let mut hgram = vec![0.0; 200];
        let mut count = 0;
        let total = 100_000;
        for _ in 0..total {
            let (f, _, n) = spectrum.sample(&mut rng);
            count += n;
            let bin = (200.0 * f) as usize;
            if bin < 200 {
                hgram[bin] += 1.0;
            }
        }
        let eff = (total as f64) / (count as f64);
        println!("Efficiency at chi = {}: {:.2}%", chi, 100.0 * eff);
        assert!(eff > 0.5);
        // for elem in hgram.iter() {
        //     print!("{:.6e},", 200.0 * elem / (total as f64));
        // }
        // println!();
    }
}