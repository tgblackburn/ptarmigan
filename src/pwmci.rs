//! Piecewise monotonic cubic interpolation

const RECURSION_LIMIT: usize = 32;
const PRECISION_LIMIT: f64 = 1.0e-6;

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

#[derive(Copy,Clone,Debug)]
struct Root {
    x: f64,
    f: f64,
}

fn is_between(x: f64, min: f64, max: f64) -> bool {
    if min < max {
        (x > min) && (x < max)
    } else if max < min {
        (x > max) && (x < min)
    } else {
        true
    }
}

pub fn evaluate(x: f64, table: &[[f64; 2]]) -> Option<f64> {
    // Find the i for which table[i-1][1] < f <= table[i][1]

    let mut i = 0;
    while i < table.len() && x > table[i][0] {
        i += 1;
    }

    if i >= table.len() - 1 && x > table.last().unwrap()[0] {
        return None;
    }

    let fit_pars = FitParameters::construct(i, table);

    Some(fit_pars.evaluate(x))
}

pub fn invert(f: f64, table: &[[f64; 2]]) -> Option<(f64, usize)> {
    // Find the i for which table[i-1][1] < f <= table[i][1]

    let mut i = 0;
    while i < table.len() && f > table[i][1] {
        i += 1;
    }

    if i >= table.len() - 1 && f > table.last().unwrap()[1] {
        return None;
    }

    //println!("Root lies between {} and {}", table[i-1][0], table[i][0]);

    // Get the tangent slopes at x_{i-1} and x_i, corrected to
    // guarantee monotonicity.
    let fit_pars = FitParameters::construct(i, table);

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

    root[0].x = table[i-1][0];
    root[0].f = table[i-1][1] - f;
    root[1].x = table[i  ][0];
    root[1].f = table[i  ][1] - f;

    if root[1].f.abs() < root[0].f.abs() {
        // swap root[0] and root[1]
        let tmp = root[0];
        root[0] = root[1];
        root[1] = tmp;
    }

    root[2] = root[1];

    let mut prev_bisect = false;

    let mut counter = 0;
    for _i in 0..RECURSION_LIMIT {
        counter +=1;

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
        let s = if !is_between(s, 0.25 * (3.0 * root[1].x + root[0].x), root[0].x)
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

        if error < PRECISION_LIMIT || root[0].f == 0.0 {
            break;
        }
    }

    /*
    if i == RECURSION_LIMIT - 1 {
        eprintln!("Recusion limit of {} exceeded, estimated error = {}.", RECURSION_LIMIT, error);
    }
    */

    Some((root[0].x, counter))
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
        let (x, nevals) = invert(y, &table).unwrap();
        let err = (x - y.sqrt()).abs();

        println!("got {:e}, expected {:e}, error = {:e}, nevals = {}", x, y.sqrt(), err, nevals);
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
        let (x, nevals) = invert(y, &table).unwrap();
        let err = (x - y.atanh()).abs();

        println!("got {:e}, expected {:e}, error = {:e}, nevals = {}", x, y.atanh(), err, nevals);
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
        let (x, nevals) = invert(y, &table).unwrap();
        let target = 2.0 - (1.0 - y).atanh();
        let err = (x - target).abs();

        println!("got {:e}, expected {:e}, error = {:e}, nevals = {}", x, target, err, nevals);
        assert!(err < 1.0e-4);
    }
}