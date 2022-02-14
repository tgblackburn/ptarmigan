//! Implementation of the generalized Bessel function of two
//! arguments, J(n, x, y).

use std::convert::TryInto;
use std::f64::consts;
use num_complex::Complex;

const DELTA: i32 = 100;

#[derive(Copy, Clone)]
struct Element {
    xi: [f64; 3],
    lambda: [f64; 2],
    f: f64,
    g: f64,
    h: f64,
}

/// Holds sufficient memory to store intermediate results
/// generated when evaluating a double Bessel function of
/// order `n`.
/// The workspace can be reused for many calls at
/// the same `n`.
pub struct DoubleBessel {
    n: i32,
    k_min: i32,
    x_max: f64,
    y_max: f64,
    tbl: Vec<Element>,
}

impl DoubleBessel {
    /// Allocates a workspace that is used to store intermediate
    /// results generated during evaluation of the double Bessel
    /// function. A workspace can be reused for many calls at
    /// the same index `n`, provided that the requested `x` and `y`
    /// satisfy `0 <= x <= x_max` and `0 <= y <= y_max`.
    #[allow(unused)]
    pub fn at_index(n: i32, x_max: f64, y_max: f64) -> Self {
        // We need to store elements (i.e. values of J_n)
        // for n in k_min < n_minus < n < n_plus < k_max,
        // where the difference between k_min and n_minus
        // is given by the safety margin DELTA.
        //
        // For fixed x, y, we have
        //  n_minus = -2.0 * y - x
        //  n_plus = 2.0 * y + x * x / (16.0 * y) if 8.0 * y > x,
        //           -2.0 * y + x, otherwise
        // We need n_minus_bound and n_plus_bound such that,
        // for all 0 <= x <= x_max, 0 <= y <= y_max,
        //   n_minus > n_minus_bound
        //   n_plus < n_plus_bound
        let n_minus_bound = (-2.0 * y_max - x_max).floor() as i32;
        let n_plus_bound = if x_max > 4.0 * (2.0 - consts::SQRT_2) * y_max {
            x_max.ceil() as i32
        } else {
            (2.0 * y_max + x_max * x_max / (16.0 * y_max)).ceil() as i32
        };
        let size: usize = (n_plus_bound - n_minus_bound + 2 * DELTA)
            .try_into()
            .unwrap();
        let default = Element {
            xi: [1.0; 3],
            lambda: [1.0; 2],
            f: 0.0,
            g: 0.0,
            h: 0.0,
        };
        DoubleBessel {
            n,
            k_min: n_minus_bound - DELTA,
            x_max: x_max.abs(),
            y_max: y_max.abs(),
            tbl: vec![default; size],
        }
    }

    /// Returns the index this workspace has been prepared for.
    #[allow(unused)]
    pub fn n(&self) -> i32 {
        self.n
    }

    /// Returns the value of the double Bessel function, `J(n, x, y)`,
    /// for integer `n` and real, positive `x`, `y`.
    ///
    /// The function is defined as
    ///   `(1/π) ∫_0^π cos[-n t + x cos(t) - y sin(t)] dt`
    /// or equivalently by
    ///   `Σ_{k=-∞}^∞ J(n+2k, x) J(n, y)`
    /// where `J(n,x)` is the usual Bessel function.
    ///
    /// The index `n` is specified by the workspace `self`,
    /// which can be reused for many calculations at the same `n`:
    ///
    /// # Examples
    ///
    /// ```
    /// let n = 10;
    /// let mut dj = DoubleBessel::at_index(n, 10.0, 5.0);
    /// let res1 = dj.at(0.0, 5.0);
    /// let res2 = dj.at(2.0, 3.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the `x` and `y` requested are
    /// outside the bounds specified when constructing the
    /// workspace `dj`.
    #[allow(unused)]
    pub fn at(&mut self, x: f64, y: f64) -> f64 {
        self.around(x, y)[2]
    }

    /// Returns the value of the double Bessel function, `J(n, x, y)`,
    /// for integers `n-2`, `n-1`, `n`, `n+1` and `n+2` for
    /// real, positive `x`, `y`.
    ///
    /// The function is defined as
    ///   `(1/π) ∫_0^π cos[-n t + x cos(t) - y sin(t)] dt`
    /// or equivalently by
    ///   `Σ_{k=-∞}^∞ J(n+2k, x) J(n, y)`
    /// where `J(n,x)` is the usual Bessel function.
    ///
    /// The index `n` is specified by the workspace `self`,
    /// which can be reused for many calculations at the same `n`:
    ///
    /// # Examples
    ///
    /// ```
    /// let n = 10;
    /// let mut dj = DoubleBessel::at_index(n, 10.0, 5.0);
    /// let res1 = dj.around(0.0, 5.0);
    /// let res2 = dj.around(2.0, 3.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the `x` and `y` requested are
    /// outside the bounds specified when constructing the
    /// workspace `dj`.
    pub fn around(&mut self, x: f64, y: f64) -> [f64; 5] {
        let n = self.n;
        assert!(x <= self.x_max && y <= self.y_max);

        let n_minus = (-2.0 * y - x).floor() as i32;
        let n_plus = if 8.0 * y > x {
            (2.0 * y + x * x / (16.0 * y)).ceil() as i32
        } else {
            (-2.0 * y + x).ceil() as i32
        };

        let tbl = &mut self.tbl;
        let len = tbl.len();
        let k_min = self.k_min;

        // forward recurrence for
        // xi_0[n] = -x - 4 y^2 / xi_2[n-1]
        // xi_1[n] = 2 (n - 1) - 2 y xi_0[n-1]
        // xi_2[n] = -x - 2 y xi_1[n-1] / xi_2[n-1]
        // and
        // lambda_0[n] = xi_0[n] - 2 y xi_2[n]
        // lambda_1[n] = xi_1[n] - lambda_0[n-1] xi_2[n] / lambda_1[n-1]
        for i in 1..len {
            let k = (i as i32) + k_min; // the real index of the sum
            tbl[i].xi[0] = -x - 4.0 * y * y / tbl[i - 1].xi[2];
            tbl[i].xi[1] = 2.0 * ((k - 1) as f64) - 2.0 * y * tbl[i - 1].xi[0] / tbl[i - 1].xi[2];
            tbl[i].xi[2] = -x - 2.0 * y * tbl[i - 1].xi[1] / tbl[i - 1].xi[2];
            tbl[i].lambda[0] = tbl[i].xi[0] - 2.0 * y * tbl[i].xi[2] / tbl[i - 1].lambda[1];
            tbl[i].lambda[1] =
                tbl[i].xi[1] - tbl[i - 1].lambda[0] * tbl[i].xi[2] / tbl[i - 1].lambda[1];
        }

        let initial = if n < 2000 {
            1.0e-100 // for n = 10, 100, 1000
        } else {
            1.0e-300 // better for n = 3000, 10_000
        };

        // backward recurrence
        tbl[len - 1].f = 0.0;
        tbl[len - 2].f = 1.0 * initial;
        tbl[len - 3].f = 2.0 * initial;
        tbl[len - 1].g = 0.0;
        tbl[len - 2].g = 1.0 / initial;
        tbl[len - 3].g = -(2.0 * y * tbl[len - 1].g + tbl[len - 2].lambda[0] * tbl[len - 2].g)
            / tbl[len - 2].lambda[1];

        // match f and g
        let k_match = (n_plus + n_minus) / 2;
        // k_min -> 0, k_max -> len
        let i_match = (k_match - k_min) as usize;
        assert!(i_match > 0);
        // tbl may contain values from a previous run, so overwrite
        tbl[i_match].f = 0.0;
        tbl[i_match].g = 0.0;

        for i in (0..(len - 3)).rev() {
            tbl[i].f = -(2.0 * y * tbl[i + 3].f
                + tbl[i + 2].xi[0] * tbl[i + 2].f
                + tbl[i + 2].xi[1] * tbl[i + 1].f)
                / tbl[i + 2].xi[2];
            tbl[i].g = -(2.0 * y * tbl[i + 2].g + tbl[i + 1].lambda[0] * tbl[i + 1].g)
                / tbl[i + 1].lambda[1];
            // if tbl[i].f.is_finite() {
            //     debug_println!("[{}]: f = {:.3e}, g = {:.3e}", i, tbl[i].f, tbl[i].g);
            // }
            if tbl[i].f > 1.0 && tbl[i_match].g.abs() < 1.0e200 {
                for j in i..len {
                    tbl[j].f *= 1.0e-100;
                    tbl[j].g *= 1.0e100;
                }
            }
        }

        let f_match = tbl[i_match].f;
        let g_match = tbl[i_match].g;
        let mut sum_linear = 0.0;
        let mut root_sum_square: f64 = 0.0;

        for i in 0..i_match {
            tbl[i].h = tbl[i].g;
            sum_linear += tbl[i].h;
            root_sum_square = root_sum_square.hypot(tbl[i].h);
        }

        for i in i_match..len {
            tbl[i].h = g_match * tbl[i].f / f_match;
            sum_linear += tbl[i].h;
            root_sum_square = root_sum_square.hypot(tbl[i].h);
        }

        // get target index
        let i_target = (n - k_min) as usize;

        let mut j = [0.0; 5];

        for (i, t) in tbl[i_target-2..=i_target+2].iter().enumerate() {
            let val = (t.h / sum_linear).signum() * t.h.abs() / root_sum_square;
            if val.is_finite() {
                j[i] = val;
            }
        }

        j
    }

    /// Returns the value of the double Bessel function, `J(n, x, y)`,
    /// for integer `n` and real, positive `x`, `y`, calculated
    /// using a saddle-point approximation valid for large order.
    /// The index `n` is specified by the workspace `self`.
    #[allow(unused)]
    fn at_asymptotic(&self, x: f64, y: f64) -> f64 {
        self.around_asymptotic(x, y)[2]
    }

    /// Returns the values of the double Bessel function, `J(n, x, y)`,
    /// for integer `n-2`, `n-1`, `n`, `n+1`, `n+2` and real,
    /// positive `x`, `y`, calculated using a saddle-point approximation
    /// valid for large order.
    /// The central index `n` is specified by the workspace `self`.
    fn around_asymptotic(&self, x: f64, y: f64) -> [f64; 5] {
        [
            DoubleBessel::j_spa(self.n - 2, x, y),
            DoubleBessel::j_spa(self.n - 1, x, y),
            DoubleBessel::j_spa(self.n,     x, y),
            DoubleBessel::j_spa(self.n + 1, x, y),
            DoubleBessel::j_spa(self.n + 2, x, y),
        ]
    }

    fn j_spa(n: i32, x: f64, y: f64) -> f64 {
        let n = n as f64;
        let j = if n <= -2.0 * y + x {
            // one real, two complex saddle points
            let cos_theta = if y > 0.0 {
                x / (8.0 * y) - (x * x / (64.0 * y * y) + 0.5 - n / (4.0 * y)).sqrt()
            } else {
                n / x
            };

            let theta = cos_theta.acos();
            let f = x * theta.sin() - y * (2.0 * theta).sin();
            let d2f = -x * theta.sin() + 4.0 * y * (2.0 * theta).sin();
            let phi = consts::FRAC_PI_4 * d2f.signum();

            (f - n * theta + phi).cos() * (2.0 / (consts::PI * d2f.abs())).sqrt()
        } else if 8.0 * y < x || n > 2.0 * y + x * x / (16.0 * y) {
            // two complex saddle points
            let discriminant = x * x / (64.0 * y * y) + 0.5 - n / (4.0 * y);
            // if dis = 0, then d2f vanishes.

            let cos_theta = if discriminant.is_nan() {
                // y = 0
                Complex::new(n / x, 0.0)
            } else if discriminant < 0.0 {
                Complex::new(x / (8.0 * y), (-discriminant).sqrt())
            } else {
                // take root with smallest imaginary part
                Complex::new(x / (8.0 * y) - discriminant.sqrt(), 0.0)
            };

            let theta = cos_theta.acos();
            let f = Complex::<f64>::i() * (-n * theta + x * theta.sin() - y * (2.0 * theta).sin());
            let d2f = Complex::<f64>::i() * (-x * theta.sin() + 4.0 * y * (2.0 * theta).sin());
            let phi = Complex::new(0.0, (consts::PI - d2f.arg()) / 2.0);
            let phase = f + phi;

            let term = if f.im == 0.0 && d2f.im == 0.0 {
                // arg d2f = 0.0, phi = i pi / 2, phase = i e(f)
                //Complex::<f64>::i() * f.re.exp() * (2.0 / (consts::PI * d2f.norm())).sqrt()
                let val = 0.5 * (-f.re).exp() * (2.0 / (consts::PI * d2f.norm())).sqrt();
                Complex::new(val, 0.0)
            } else {
                phase.exp() * (2.0 / (consts::PI * d2f.norm())).sqrt()
            };

            if discriminant != 0.0 {
                term.re
            } else {
                0.0
            }
        } else {
            // two real saddle points
            let cos_theta = [
                x / (8.0 * y) - (x * x / (64.0 * y * y) + 0.5 - n / (4.0 * y)).sqrt(),
                x / (8.0 * y) + (x * x / (64.0 * y * y) + 0.5 - n / (4.0 * y)).sqrt(),
            ];

            let theta = [
                cos_theta[0].acos(),
                cos_theta[1].acos(),
            ];

            let f = [
                x * theta[0].sin() - y * (2.0 * theta[0]).sin(),
                x * theta[1].sin() - y * (2.0 * theta[1]).sin(),
            ];

            let d2f = [
                -x * theta[0].sin() + 4.0 * y * (2.0 * theta[0]).sin(),
                -x * theta[1].sin() + 4.0 * y * (2.0 * theta[1]).sin(),
            ];

            let phi = [
                consts::FRAC_PI_4 * d2f[0].signum(),
                consts::FRAC_PI_4 * d2f[1].signum(),
            ];

            (f[0] - n * theta[0] + phi[0]).cos() * (2.0 / (consts::PI * d2f[0].abs())).sqrt()
                + (f[1] - n * theta[1] + phi[1]).cos() * (2.0 / (consts::PI * d2f[1].abs())).sqrt()
        };

        j
    }

    /// Returns the value of the double Bessel function, `J(n, x, y)`,
    /// for integers `n-2`, `n-1`, `n`, `n+1` and `n+2` for
    /// real, positive `x`, `y`, switching between recurrence and
    /// saddle-point methods as appropriate.
    ///
    /// The function is defined as
    ///   `(1/π) ∫_0^π cos[-n t + x cos(t) - y sin(t)] dt`
    /// or equivalently by
    ///   `Σ_{k=-∞}^∞ J(n+2k, x) J(n, y)`
    /// where `J(n,x)` is the usual Bessel function.
    ///
    /// The index `n` is specified by the workspace `self`,
    /// which can be reused for many calculations at the same `n`:
    ///
    /// # Examples
    ///
    /// ```
    /// let n = 10;
    /// let mut dj = DoubleBessel::at_index(n, 10.0, 5.0);
    /// let res1 = dj.evaluate(0.0, 5.0);
    /// let res2 = dj.evaluate(2.0, 3.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the `x` and `y` requested are
    /// outside the bounds specified when constructing the
    /// workspace `dj`.
    #[allow(unused)]
    pub fn evaluate(&mut self, x: f64, y: f64) -> [f64; 5] {
        if self.use_saddle_point(x, y) {
            self.around_asymptotic(x, y)
        } else {
            self.around(x, y)
        }
    }

    /// Returns true if the relative error of the saddle point
    /// approximation is less than 1% for given x and y.
    fn use_saddle_point(&self, x: f64, y: f64) -> bool {
        if self.n <= 10 {
            false
        } else {
            let n_eff = self.n as f64;
            let n_eff = n_eff * (1.0 - (10.0 / n_eff).sqrt());
            let x_crit = if y > n_eff / 6.0 {
                4.0 * (y * (n_eff - 2.0 * y)).sqrt()
            } else {
                n_eff + 2.0 * y
            };
            x < x_crit
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    const MAX_ERROR: f64 = 1.0e-9;

    #[test]
    fn j_10() {
        let n = 10;
        let x_max = (n as f64) * consts::SQRT_2;
        let y_max = (n as f64) * 0.5;

        let pts: [(f64, f64, f64); 9] = [
            (0.0 * x_max, 0.0 * y_max, 0.0),
            (0.5 * x_max, 0.0 * y_max, 0.025390202714102875284),
            (1.0 * x_max, 0.0 * y_max, 0.059874153886298191146),
            (0.0 * x_max, 0.5 * y_max, -0.019501625134503219887),
            (0.5 * x_max, 0.5 * y_max, 0.027510433345552824743),
            (1.0 * x_max, 0.5 * y_max, 0.38215542188913600712),
            (0.0 * x_max, 1.0 * y_max, -0.26114054612017009006),
            (0.5 * x_max, 1.0 * y_max, -0.23047030587105184156),
            (1.0 * x_max, 1.0 * y_max, -0.25467130786338730740),
        ];

        let mut j = DoubleBessel::at_index(n, x_max, y_max);

        for (x, y, target) in pts.iter() {
            let result = j.at(*x, *y);
            let error = if target.abs() != 0.0 {
                ((result - target) / target).abs()
            } else {
                result.abs()
            };
            println!(
                "\tJ({}, {:.3e}, {:.3e}) = {:.3e}, error = {:.3e}",
                n, x, y, result, error
            );
            assert!(error < MAX_ERROR);
        }
    }

    #[test]
    fn j_100() {
        let n = 100;
        let x_max = (n as f64) * consts::SQRT_2;
        let y_max = (n as f64) * 0.5;

        let pts: [(f64, f64, f64); 15] = [
            (0.000 * x_max, 0.0 * y_max, 0.0),
            (0.500 * x_max, 0.0 * y_max, 1.2767655707448450836e-9),
            (0.750 * x_max, 0.0 * y_max, 0.11348399670876930488),
            (0.875 * x_max, 0.0 * y_max, -0.088848909903940246084),
            (1.000 * x_max, 0.0 * y_max, -0.019823793313669992518),
            (0.0000 * x_max, 0.5 * y_max, 9.7561594280301528247e-12),
            (0.5000 * x_max, 0.5 * y_max, -1.1516847439470076527e-8),
            (0.8750 * x_max, 0.5 * y_max, 0.0015245492806493982369),
            (0.9375 * x_max, 0.5 * y_max, -0.024581413776829650737),
            (1.0000 * x_max, 0.5 * y_max, -0.18453656424118167504),
            (0.00 * x_max, 1.0 * y_max, 0.12140902189761506382),
            (0.25 * x_max, 1.0 * y_max, -0.11027612091221220768),
            (0.50 * x_max, 1.0 * y_max, 0.12395439679758243884),
            (0.75 * x_max, 1.0 * y_max, 0.035901855604319056478),
            (1.00 * x_max, 1.0 * y_max, -0.11789385413375555804),
        ];

        let mut j = DoubleBessel::at_index(n, x_max, y_max);

        for (x, y, target) in pts.iter() {
            let result = j.at(*x, *y);
            let error = if target.abs() != 0.0 {
                ((result - target) / target).abs()
            } else {
                result.abs()
            };
            println!(
                "\tJ({}, {:.3e}, {:.3e}) = {:.3e}, error = {:.3e}",
                n, x, y, result, error
            );
            assert!(error < MAX_ERROR);
        }
    }

    #[test]
    fn j_1000() {
        let n = 1000;
        let x_max = (n as f64) * consts::SQRT_2;
        let y_max = (n as f64) * 0.5;

        let pts: [(f64, f64, f64); 12] = [
            //( 800.0,   0.0,  5.7306149153241744571e-43),
            ( 900.0,   0.0,  5.0841100850412997894e-16),
            (1000.0,   0.0,  4.4730672947964040881e-2),
            (1200.0,   0.0,  3.5826674378828883711e-3),
            (1400.0,   0.0, -2.3607454432146489015e-2),
            ( 900.0, 250.0, -2.9901552551414229197e-52),
            (1000.0, 250.0,  2.9185939076571946059e-42),
            (1100.0, 250.0,  1.9079405288254989534e-31),
            (1200.0, 250.0, -1.8162913281548281903e-21),
            (   0.0, 500.0,  5.6357003281836941079e-2),
            ( 250.0, 500.0, -6.5000743052577121220e-3),
            ( 500.0, 500.0, -9.0439582759926842120e-3),
            (1000.0, 500.0,  5.6390291047140394025e-3),
        ];

        let mut j = DoubleBessel::at_index(n, x_max, y_max);

        for (x, y, target) in pts.iter() {
            let result = j.at(*x, *y);
            let error = if target.abs() != 0.0 {
                ((result - target) / target).abs()
            } else {
                result.abs()
            };
            println!(
                "\tJ({}, {:.3e}, {:.3e}) = {:.3e}, error = {:.3e}",
                n, x, y, result, error
            );
            assert!(error < MAX_ERROR);
        }
    }

    #[test]
    fn j_3000() {
        let n = 3000;
        let x_max = (n as f64) * consts::SQRT_2;
        let y_max = (n as f64) * 0.5;

        let pts: [(f64, f64, f64); 7] = [
            (2800.0,    0.0,  1.8840754382138119402e-24),
            (3000.0,    0.0,  3.1014547810126577726e-2),
            (3500.0,    0.0, -1.7454715995955427297e-2),
            (4000.0,    0.0,  1.1253611410673444442e-2),
            (3000.0, 1500.0,  3.1854710128360870423e-3),
            (3500.0, 1500.0,  9.5390642676650740986e-3),
            (4000.0, 1500.0, -8.2208784394371069136e-3),
        ];

        let mut j = DoubleBessel::at_index(n, x_max, y_max);

        for (x, y, target) in pts.iter() {
            let result = j.at(*x, *y);
            let error = if target.abs() != 0.0 {
                ((result - target) / target).abs()
            } else {
                result.abs()
            };
            println!(
                "\tJ({}, {:.3e}, {:.3e}) = {:.3e}, error = {:.3e}",
                n, x, y, result, error
            );
            assert!(error < MAX_ERROR);
        }
    }

    #[test]
    fn j_10_000() {
        let n = 10_000;
        let x_max = (n as f64) * consts::SQRT_2;
        let y_max = (n as f64) * 0.5;

        let pts: [(f64, f64, f64); 7] = [
            (1.0e4, 0.0,    2.0762165277200784504e-2),
            (1.2e4, 0.0,   -9.1522201827134628227e-3),
            (1.4e4, 2.5e3, -1.8302613412305081307e-6),
            (4.0e3, 5.0e3, -2.5197488647084015072e-2),
            (8.0e3, 5.0e3, -2.2166895777931910330e-3),
            (1.0e4, 5.0e3, -1.3363371280336776726e-2),
            (1.2e4, 5.0e3,  1.7163084844210792828e-4),
        ];

        let mut j = DoubleBessel::at_index(n, x_max, y_max);

        for (x, y, target) in pts.iter() {
            let result = j.at(*x, *y);
            let error = if target.abs() != 0.0 {
                ((result - target) / target).abs()
            } else {
                result.abs()
            };
            println!(
                "\tJ({}, {:.3e}, {:.3e}) = {:.3e}, error = {:.3e}",
                n, x, y, result, error
            );
            assert!(error < MAX_ERROR);
        }
    }

    #[test]
    fn saddle_point_accuracy() {
        use rand::prelude::*;
        use rand_xoshiro::*;

        let n = 100;
        let mut j = DoubleBessel::at_index(n, (n as f64) * consts::SQRT_2, 0.5 * (n as f64));
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        for _i in 0..100 {
            let x = (n as f64) * consts::SQRT_2 * rng.gen::<f64>();
            let y = 0.5 * (n as f64) * rng.gen::<f64>();

            // skip points outside the [x(s), y(s)] curve
            // if y > (n as f64) / 6.0 && x > 4.0 * y.sqrt() * ((n as f64) - 2.0 * y).sqrt() {
            //     continue;
            // } else if y <= (n as f64) / 6.0 && x > (n as f64) + 2.0 * y {
            //     continue;
            // }
            if !j.use_saddle_point(x, y) {
                continue;
            }

            let target = j.at(x, y);
            let value = j.at_asymptotic(x, y);
            let error = if target != 0.0 {
                let error = ((target - value) / target).abs();
                println!("{:.6e} {:.6e} {:.6e} {:.6e} {:.6e}", x, y, target, value, error);
                error
            } else {
                0.0
            };

            assert!(target.abs() < 1.0e-9 || error < 1.0e-2);
        }
    }
}
