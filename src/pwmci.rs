//! Piecewise monotonic cubic interpolation

/// A piecewise, monotonic cubic interpolant (PWMCI) of a 1D function
pub struct Interpolant<'a> {
    tbl: &'a [[f64; 2]],
    ext: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Location {
    Above,
    Within,
    Below,
}

#[derive(Copy, Clone, Debug)]
struct Root {
    x: f64,
    f: f64,
}

impl<'a> Interpolant<'a> {
    const MAX_RECURSION: usize = 32;
    const MIN_PRECISION: f64 = 1.0e-6;

    /// Interpolates the supplied data points.
    /// Assumes that the data are strictly increasing!
    pub fn new(table: &'a [[f64; 2]]) -> Self {
        Self {
            tbl: table,
            ext: false,
        }
    }

    /// Specifies whether the data should be linearly extrapolated
    /// beyond the supplied domain.
    #[allow(unused)]
    pub fn extrapolate(mut self, status: bool) -> Self {
        self.ext = status;
        self
    }


    /// Returns the value of the interpolating function at the point
    /// `x` (wrapped in `Some`), or `None` if the specified point is
    /// outside the domain and extrapolation has not been selected.
    pub fn evaluate(&self, x: f64) -> Option<f64> {
        let len = self.tbl.len();

        // Find the i for which table[i-1][0] < x <= table[i][0]
        let (i, loc) = self.tbl.iter()
            .enumerate()
            .rev()
            .find(|(_, [t, _])| {
                x > *t
            })
            .map_or(
                // None means x <= table[0][0]
                (1, Location::Below),
                // Otherwise, x might be larger than table[len-1][0]
                |(i, _)| if i == len - 1 {
                    (len - 1, Location::Above)
                } else {
                    (i + 1, Location::Within)
                }
            );

        match loc {
            Location::Within => {
                let fit_pars = FitParameters::construct(i, self.tbl);
                Some(fit_pars.evaluate(x))
            },
            Location::Above | Location::Below => {
                if self.ext {
                    // Linear extrapolation
                    let [x1, f1] = self.tbl[i-1];
                    let [x2, f2] = self.tbl[i];
                    let f = f1 + (f2 - f1) * (x - x1) / (x2 - x1);
                    Some(f)
                } else {
                    None
                }
            }
        }
    }

    /// Is min < x < max satisfied?
    fn is_between(x: f64, min: f64, max: f64) -> bool {
        if min < max {
            (x > min) && (x < max)
        } else if max < min {
            (x > max) && (x < min)
        } else {
            true
        }
    }

    /// Returns the point `x` at which the interpolating function has
    /// value `f` (wrapped in `Some`), or `None` if the supplied point `f`
    /// is outside the range of the supplied points and extrapolation has
    /// not been selected.
    pub fn invert(&self, f: f64) -> Option<f64> {
        let len = self.tbl.len();

        // Find the i for which table[i-1][1] < f <= table[i][1]
        let (i, loc) = self.tbl.iter()
            .enumerate()
            .rev()
            .find(|(_, [_, y])| {
                f > *y
            })
            .map_or(
                // None means f <= table[0][1]
                (1, Location::Below),
                // Otherwise, f might be larger than table[len-1][1]
                |(i, _)| if i == len - 1 {
                    (len - 1, Location::Above)
                } else {
                    (i + 1, Location::Within)
                }
            );

        let fit_pars = match loc {
            // Get the tangent slopes at x_{i-1} and x_i, corrected to
            // guarantee monotonicity.
            Location::Within => FitParameters::construct(i, self.tbl),
            // Otherwise, return early
            Location::Above | Location::Below => {
                if self.ext {
                    // Linear extrapolation
                    let [x1, f1] = self.tbl[i-1];
                    let [x2, f2] = self.tbl[i];
                    let x = x1 + (f - f1) * (x2 - x1) / (f2 - f1);
                    return Some(x);
                } else {
                    return None;
                }
            }
        };

        // Now that we have the tangents at x_{i-1} and x_i, we can
        // use cubic Hermite interpolation to evaluate f(x).
        //
        // To solve f(x) == f, we use inverse quadratic interpolation.
        // We know that the root must lie between table[i-1][0] and
        // table[i][0].
        //
        // Initialising the loop requires 3 values. ‘root[0].x’
        // is always the best current estimate of the root;
        // ‘root[1].x’ the previous best estimate, and ‘root[2].x’
        // the one before that. */

        let mut root = [Root{x: 0.0, f: 0.0}; 4];

        root[0].x = self.tbl[i-1][0];
        root[0].f = self.tbl[i-1][1] - f;
        root[1].x = self.tbl[i  ][0];
        root[1].f = self.tbl[i  ][1] - f;

        if root[1].f.abs() < root[0].f.abs() {
            // swap root[0] and root[1]
            let tmp = root[0];
            root[0] = root[1];
            root[1] = tmp;
        }

        root[2] = root[1];

        let mut prev_bisect = false;

        for _i in 0..Interpolant::MAX_RECURSION {
            // First try inverse quadratic interpolation,
            // provided all roots are distinct.
            let s = if root[0].f != root[2].f && root[1].f != root[2].f {
                //println!("Attempting inverse quadratic interpolation...");
                let r = root[0].f / root[2].f;
                let s = root[0].f / root[1].f;
                let t = root[1].f / root[2].f;
                let p = s * (t * (r-t) * (root[2].x - root[0].x) - (1.0-r) * (root[0].x - root[1].x));
                let q = (t-1.0) * (r-1.0) * (s-1.0);
                root[0].x + p / q
            } else {
                // use secant method
                //println!("Attempting linear interpolation...");
                root[0].x - (root[0].f * (root[0].x - root[1].x) / (root[0].f - root[1].f))
            };

            // Verify if our 's' is acceptable
            let s = if !Self::is_between(s, 0.25 * (3.0 * root[1].x + root[0].x), root[0].x)
                || (prev_bisect && (s - root[0].x).abs() >= 0.5 * (root[0].x - root[2].x).abs())
                || (!prev_bisect && (s - root[0].x).abs() >= 0.5 * (root[2].x - root[3].x).abs())
                {
                //println!("Failed, using bisection.");
                prev_bisect = true;
                0.5 * (root[0].x + root[1].x)
            } else {
                //println!("Successful!");
                prev_bisect = false;
                s
            };

            // reorder roots
            root[3] = root[2];
            root[2] = root[0];
            root[0].x = s;
            root[0].f = fit_pars.evaluate(s) - f;

            if root[1].f * root[0].f >= 0.0 {
                root[1] = root[0];
                root[0] = root[2];
            }

            if root[1].f.abs() < root[0].f.abs() {
                let tmp = root[0];
                root[0] = root[1];
                root[1] = tmp;
            }

            let error = (root[1].x - root[0].x).abs() / root[0].x.abs();

            //println!("Root estd to be {:e} (1 pm {:e}).", root[0].x, 0.5 * error);

            if error < Interpolant::MIN_PRECISION || root[0].f == 0.0 {
                break;
            }
        }

        Some(root[0].x)
    }
}

#[derive(Copy,Clone,Debug)]
struct FitParameters {
    x: [f64; 2],
    f: [f64; 2],
    m: [f64; 2],
}

impl FitParameters {
    fn construct(i: usize, table: &[[f64; 2]]) -> FitParameters {
        let len = table.len();

        // Slopes of the secant lines between x_{i-2}, x_{i-1}, x_i and x_{i+1}
        let secant = if i == 1 {
            [
                (table[i  ][1] - table[i-1][1]) / (table[i  ][0] - table[i-1][0]),
                (table[i  ][1] - table[i-1][1]) / (table[i  ][0] - table[i-1][0]),
                (table[i+1][1] - table[i  ][1]) / (table[i+1][0] - table[i  ][0]),
            ]
        } else if i == len - 1 {
            [
                (table[i-1][1] - table[i-2][1]) / (table[i-1][0] - table[i-2][0]),
                (table[i  ][1] - table[i-1][1]) / (table[i  ][0] - table[i-1][0]),
                (table[i  ][1] - table[i-1][1]) / (table[i  ][0] - table[i-1][0]),
            ]
        } else {
            // all accesses are within bounds
            assert!(i > 0 && i < len - 1);
            [
                (table[i-1][1] - table[i-2][1]) / (table[i-1][0] - table[i-2][0]),
                (table[i  ][1] - table[i-1][1]) / (table[i  ][0] - table[i-1][0]),
                (table[i+1][1] - table[i  ][1]) / (table[i+1][0] - table[i  ][0]),
            ]  
        };

        // Tangent slopes at x_{i-1} and x_i are calculated using the average
        // of the secant slopes UNLESS they have different signs (i.e. that point
        // is a local maximum) or one is zero (the curve is flat).
        let mut tangent = [0.0; 2];
        tangent[0] = if secant[0] * secant[1] > 0.0 {
            0.5 * (secant[0] + secant[1])
        } else {
            0.0
        };
        tangent[1] = if secant[1] * secant[2] > 0.0 {
            0.5 * (secant[1] + secant[2])
        } else {
            0.0
        };

        // Correct tangent slopes to ensure monotonicity
        if secant[1] != 0.0 {
            let alpha = tangent[0] / secant[1];
            if alpha > 3.0 {tangent[0] = 3.0 * secant[1];}
        }
        if secant[2] != 0.0 {
            let beta = tangent[1] / secant[2];
            if beta > 3.0 {tangent[1] = 3.0 * secant[2];}
        }

        FitParameters {
            x: [table[i-1][0], table[i][0]],
            f: [table[i-1][1], table[i][1]],
            m: tangent
        }
    }

    fn evaluate(&self, x: f64) -> f64 {
        let t = (x - self.x[0]) / (self.x[1] - self.x[0]);
        let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
        let h10 = t * (1.0 - t).powi(2);
        let h01 = t.powi(2) * (3.0 - 2.0 * t);
        let h11 = t.powi(2) * (t - 1.0);
        self.f[0] * h00 + self.f[1] * h01 + (self.x[1] - self.x[0]) * (self.m[0] * h10 + self.m[1] * h11)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_x_sqd() {
        let mut table: Vec<[f64; 2]> = Vec::new();
        for i in 0..20 {
            let x = (i as f64) / 20.0;
            table.push([x, x.powi(2)]);
        }

        let y = 0.73;
        let x = Interpolant::new(&table).invert(y).unwrap();
        let err = (x - y.sqrt()).abs();

        println!("got {:e}, expected {:e}, error = {:e}", x, y.sqrt(), err);
        assert!(err < 1.0e-4);
    }

    #[test]
    fn invert_tanh() {
        let mut table: Vec<[f64; 2]> = Vec::new();
        for i in 0..40 {
            let x = 5.0 * (i as f64) / 40.0;
            table.push([x, x.tanh()]);
        }

        let y = 0.22;
        let x = Interpolant::new(&table).invert(y).unwrap();
        let err = (x - y.atanh()).abs();

        println!("got {:e}, expected {:e}, error = {:e}", x, y.atanh(), err);
        assert!(err < 1.0e-4);
    }

    #[test]
    fn invert_shifted_tanh() {
        let mut table: Vec<[f64; 2]> = Vec::new();
        for i in 0..20 {
            let x = 5.0 * (i as f64) / 20.0;
            table.push([x, 1.0 + (x - 2.0).tanh()]);
        }

        let y = 1.24;
        let x = Interpolant::new(&table).invert(y).unwrap();
        let target = 2.0 - (1.0 - y).atanh();
        let err = (x - target).abs();

        println!("got {:e}, expected {:e}, error = {:e}", x, target, err);
        assert!(err < 1.0e-4);
    }

    #[test]
    fn extrapolation() {
        let mut table: Vec<[f64; 2]> = Vec::new();
        for i in 0..20 {
            let x = (i as f64) / 20.0;
            table.push([x, x.tanh()]);
        }

        let interp = Interpolant::new(&table).extrapolate(true);

        let x = -1.0;
        let y = interp.evaluate(x).unwrap();
        let target = -1.0; // x.tanh();
        let err = (y - target).abs();

        println!("at x = {:e}, got {:e}, expected {:e}, error = {:e}", x, y, target, err);
        assert!(err < 1.0e-3);

        let y = 0.9;
        let x = interp.invert(y).unwrap();
        let target = (-1.0 - y + 1_f64.tanh() + 1_f64.tanh().powi(2)) / (1_f64.tanh().powi(2) - 1.0); // y.atanh();
        let err = (x - target).abs();

        println!("got {:e}, expected {:e}, error = {:e}", x, target, err);
        assert!(err < 0.1);

        let x = -1.0;
        let y = interp.extrapolate(false).evaluate(x);
        assert!(y.is_none());
    }
}