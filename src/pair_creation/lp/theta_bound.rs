use std::f64::consts;
use num_complex::Complex64;

#[derive(Clone, Debug)]
pub struct ThetaBound {
    s: [f64; 16],
    f: [f64; 16],
}

impl ThetaBound {
    pub fn for_harmonic(n: i32, a: f64, eta: f64) -> Self {
        let mut s = [0.0; 16];
        let mut f = [0.0; 16];

        let s_min = 0.5;
        let s_max = 0.5 + (0.25 - (1.0 + 0.5 * a * a) / (2.0 * (n as f64) * eta)).sqrt();

        for i in 0..16 {
            let z = (consts::PI * (i as f64) / 15.0).cos();
            s[i] = s_min + 0.5 * (z + 1.0) * (s_max - s_min);

            // Coordinates in (x,y) space where integration over theta begins
            let x = {
                let r_n = (2.0 * (n as f64) * eta * s[i] * (1.0 - s[i]) - (1.0 + 0.5 * a * a)).sqrt();
                a * r_n / (eta * s[i] * (1.0 - s[i]))
            };
        
            let y = a * a / (8.0 * eta * s[i] * (1.0 - s[i]));

            // At fixed y, J(n,x,y) is maximised at
            let x_crit = if y > (n as f64) / 6.0 {
                4.0 * y.sqrt() * ((n as f64) - 2.0 * y).sqrt()
            } else {
                (n as f64) + 2.0 * y
            };

            // with value approx J_n(n) ~ 0.443/n^(1/3), here log-scaled
            let ln_j_crit = 0.443f64.ln() - (n as f64).ln() / 3.0;

            // The value of J(n,0,y) is approximately
            let ln_j_bdy = if n % 2 == 0 {
                Self::ln_double_bessel_x_zero(n, y)
            } else {
                Self::ln_double_bessel_x_zero(n + 1, y)
            };
            let ln_j_bdy = ln_j_bdy.min(ln_j_crit);

            // Exponential fit between x = 0 and x = x_crit, looking for the
            // x which is 100x smaller than at the starting point
            let cos_theta = 1.0 - x_crit * 0.01f64.ln() / (x * (ln_j_bdy - ln_j_crit));
            let cos_theta = cos_theta.max(-1.0);

            f[i] = cos_theta.acos();
            if f[i].is_nan() {
                f[i] = consts::PI;
            }
        }

        let mut f_min = f[0];
        for i in 1..16 {
            if f[i] < f_min {
                f_min = f[i];
            } else {
                f[i] = f_min;
            }
        }

        Self {s, f}
    }

    /// Approximate value for the double Bessel function J(n,0,y)
    /// along x = 0, using a saddle point approximation for small y
    /// and a Taylor series for y near n/2.
    fn ln_double_bessel_x_zero(n: i32, y: f64) -> f64 {
        let n = n as f64;
        let z = (n / (4.0 * y) - 0.5).sqrt();
        let theta = Complex64::new(
            consts::FRAC_PI_2,
            ((1.0 + z * z).sqrt() - z).ln()
        );
        let f = Complex64::i() * (-n * theta - y * (2.0 * theta).sin());
        let f2 = Complex64::i() * 4.0 * y * (2.0 * theta).sin();
        let e = 0.5 * (consts::PI - f2.arg());
        let phase = f + Complex64::i() * e;
        (2.0 / (consts::PI * f2.norm())).sqrt().ln() + phase.re + Complex64::new(phase.im, 0.0).cos().ln().re
    }

    pub fn at(&self, s: f64) -> f64 {
        let mut val = self.f[15];

        for i in 1..16 {
            // s[i] is stored backwards, decreasing from s_max
            if s > self.s[i] {
                let weight = (s - self.s[i-1]) / (self.s[i] - self.s[i-1]);
                val = weight * self.f[i] + (1.0 - weight) * self.f[i-1];
                break;
            }
        }

        val.min(consts::FRAC_PI_2)
    }
}