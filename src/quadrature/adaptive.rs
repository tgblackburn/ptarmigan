use num_complex::Complex64;

struct Region {
    integral: Complex64,
    error: Complex64,
    x: [f64; 2],
    y: [f64; 2],
    cache: [[Complex64; 3]; 3],
}

impl Region {
    const CLENSHAW_CURTIS_DATA: [(f64, f64, f64); 5] = [
        (0.0,                1.0 / 30.0, 1.0 / 6.0),
        (0.1464466094067262, 4.0 / 15.0, 0.0),
        (0.5,                2.0 / 5.0,  2.0 / 3.0),
        (0.8535533905932738, 4.0 / 15.0, 0.0),
        (1.0,                1.0 / 30.0, 1.0 / 6.0),
    ];

    const CLENSHAW_CURTIS_EXTENDED_NODES: [f64; 9] = [
        0.0,
        0.1464466094067262,
        0.5,
        0.8535533905932738,
        1.0,
        1.1464466094067262,
        1.5,
        1.8535533905932738,
        2.0,
    ];

    fn custom_partition<F, R>(&self, f: &mut F, region_function: &mut R, requested_x_split: Option<f64>, requested_y_split: Option<f64>) -> (Self, Self, Self, Self, i32)
    where F: FnMut(f64, f64) -> Complex64, R: FnMut(f64, f64) -> bool {
        let mut evals = 0;
        let [x0, x1] = self.x;
        let [y0, y1] = self.y;
        let x_mid = requested_x_split.unwrap_or(0.5 * (x0 + x1));
        let y_mid = requested_y_split.unwrap_or(0.5 * (y0 + y1));
        let mut z: [[Complex64; 9]; 9] = Default::default();

        // build 9x9 array of function values, 81 - 9 = 72 evals
        for (i, ty) in Region::CLENSHAW_CURTIS_EXTENDED_NODES.iter().enumerate() {
            let y = if i < 4 {
                y0 + ty * (y_mid - y0)
            } else {
                y_mid + (ty - 1.0) * (y1 - y_mid)
            };
            for (j, tx) in Region::CLENSHAW_CURTIS_EXTENDED_NODES.iter().enumerate() {
                let x = if j < 4 {
                    x0 + tx * (x_mid - x0)
                } else {
                    x_mid + (tx - 1.0) * (x1 - x_mid)
                };
                z[i][j] = if i % 8 == 0 && j % 8 == 0 {
                    // corners, always fine
                    self.cache[i/4][j/4]
                } else if i % 4 == 0 && j % 4 == 0 {
                    // fine if split is 50-50
                    if requested_x_split.is_none() && requested_y_split.is_none() {
                        self.cache[i/4][j/4]
                    } else {
                        if region_function(x, y) { evals +=1; f(x, y) } else { Complex64::new(0.0, 0.0) }
                    }
                } else {
                    if region_function(x, y) { evals +=1; f(x, y) } else { Complex64::new(0.0, 0.0) }
                };
            }
        }

        let mut result = [Complex64::new(0.0, 0.0); 4];
        let mut error = [Complex64::new(0.0, 0.0); 4];

        for r in 0..2 {
            for c in 0..2 {
                for i in 0..5 {
                    let (_, wy, ey) = Region::CLENSHAW_CURTIS_DATA[i];
                    for j in 0..5 {
                        let (_, wx, ex) = Region::CLENSHAW_CURTIS_DATA[j];
                        let val = z[i+4*r][j+4*c];
                        // how has domain been split?
                        let dy = if r < 1 {
                            y_mid - y0
                        } else {
                            y1 - y_mid
                        };
                        let dx = if c < 1 {
                            x_mid - x0
                        } else {
                            x1 - x_mid
                        };
                        result[2*r+c] += wx * wy * dx * dy * val;
                        error[2*r+c] += ex * ey * dx * dy * val;
                    }
                }
            }
        }

        let bl = Region {
            integral: result[0],
            error: result[0] - error[0],
            x: [x0, x_mid],
            y: [y0, y_mid],
            cache: [
                [z[0][0], z[0][2], z[0][4]],
                [z[2][0], z[2][2], z[2][4]],
                [z[4][0], z[4][2], z[4][4]],
            ]
        };

        let br = Region {
            integral: result[1],
            error: result[1] - error[1],
            x: [x_mid, x1],
            y: [y0, y_mid],
            cache: [
                [z[0][4], z[0][6], z[0][8]],
                [z[2][4], z[2][6], z[2][8]],
                [z[4][4], z[4][6], z[4][8]],
            ]
        };

        let tl = Region {
            integral: result[2],
            error: result[2] - error[2],
            x: [x0, x_mid],
            y: [y_mid, y1],
            cache: [
                [z[4][0], z[4][2], z[4][4]],
                [z[6][0], z[6][2], z[6][4]],
                [z[8][0], z[8][2], z[8][4]],
            ]
        };

        let tr = Region {
            integral: result[3],
            error: result[3] - error[3],
            x: [x_mid, x1],
            y: [y_mid, y1],
            cache: [
                [z[4][4], z[4][6], z[4][8]],
                [z[6][4], z[6][6], z[6][8]],
                [z[8][4], z[8][6], z[8][8]],
            ]
        };

        (bl, br, tl, tr, evals)
    }

    const CLENSHAW_CURTIS_9_DATA: [(f64, f64, f64); 9] = [
        (0.0000000000000000, 0.0079365079365079, 1.0 / 30.0),
        (0.0380602337443566, 0.0731093246080091, 0.0),
        (0.1464466094067262, 0.1396825396825397, 4.0 / 15.0),
        (0.3086582838174551, 0.1808589293602449, 0.0),
        (0.5000000000000000, 0.1968253968253968, 2.0 / 5.0),
        (0.6913417161825449, 0.1808589293602449, 0.0),
        (0.8535533905932738, 0.1396825396825397, 4.0 / 15.0),
        (0.9619397662556434, 0.0731093246080091, 0.0),
        (1.0000000000000000, 0.0079365079365079, 1.0 / 30.0),
    ];

    const CLENSHAW_CURTIS_9_EXTENDED_NODES: [f64; 17] = [
        0.0000000000000000,
        0.0380602337443566,
        0.1464466094067262,
        0.3086582838174551,
        0.5000000000000000,
        0.6913417161825449,
        0.8535533905932738,
        0.9619397662556434,
        1.0000000000000000,
        1.0380602337443566,
        1.1464466094067262,
        1.3086582838174551,
        1.5000000000000000,
        1.6913417161825449,
        1.8535533905932738,
        1.9619397662556434,
        2.0000000000000000,
    ];

    fn partition<F, R>(&self, f: &mut F, region_function: &mut R) -> (Self, Self, Self, Self, i32)
    where F: FnMut(f64, f64) -> Complex64, R: FnMut(f64, f64) -> bool {
        let mut evals = 0;
        let [x0, x1] = self.x;
        let [y0, y1] = self.y;
        let x_mid = 0.5 * (x0 + x1);
        let y_mid = 0.5 * (y0 + y1);
        let mut z: [[Complex64; 17]; 17] = Default::default();

        // build 9x9 array of function values
        for (i, ty) in Region::CLENSHAW_CURTIS_9_EXTENDED_NODES.iter().enumerate() {
            let y = y0 + 0.5 * ty * (y1 - y0);
            for (j, tx) in Region::CLENSHAW_CURTIS_9_EXTENDED_NODES.iter().enumerate() {
                let x = x0 + 0.5 * tx * (x1 - x0);
                z[i][j] = if i % 8 == 0 && j % 8 == 0 {
                    self.cache[i/8][j/8]
                } else if region_function(x, y) {
                    evals +=1;
                    f(x, y)
                } else {
                    Complex64::new(0.0, 0.0)
                };
            }
        }

        let mut result = [Complex64::new(0.0, 0.0); 4];
        let mut error = [Complex64::new(0.0, 0.0); 4];

        for r in 0..2 {
            for c in 0..2 {
                for i in 0..9 {
                    let (_, wy, ey) = Region::CLENSHAW_CURTIS_9_DATA[i];
                    for j in 0..9 {
                        let (_, wx, ex) = Region::CLENSHAW_CURTIS_9_DATA[j];
                        let val = z[i+8*r][j+8*c];
                        let dy = 0.5 * (y1 - y0);
                        let dx = 0.5 * (x1 - x0);
                        result[2*r+c] += wx * wy * dx * dy * val;
                        error[2*r+c] += ex * ey * dx * dy * val;
                    }
                }
            }
        }

        let bl = Region {
            integral: result[0],
            error: result[0] - error[0],
            x: [x0, x_mid],
            y: [y0, y_mid],
            cache: [
                [z[0][0], z[0][4], z[0][8]],
                [z[4][0], z[4][4], z[4][8]],
                [z[8][0], z[8][4], z[8][8]],
            ]
        };

        let br = Region {
            integral: result[1],
            error: result[1] - error[1],
            x: [x_mid, x1],
            y: [y0, y_mid],
            cache: [
                [z[0][8], z[0][12], z[0][16]],
                [z[4][8], z[4][12], z[4][16]],
                [z[8][8], z[8][12], z[8][16]],
            ]
        };

        let tl = Region {
            integral: result[2],
            error: result[2] - error[2],
            x: [x0, x_mid],
            y: [y_mid, y1],
            cache: [
                [z[8][0], z[8][4], z[8][8]],
                [z[12][0], z[12][4], z[12][8]],
                [z[16][0], z[16][4], z[16][8]],
            ]
        };

        let tr = Region {
            integral: result[3],
            error: result[3] - error[3],
            x: [x_mid, x1],
            y: [y_mid, y1],
            cache: [
                [z[8][8], z[8][12], z[8][16]],
                [z[12][8], z[12][12], z[12][16]],
                [z[16][8], z[16][12], z[16][16]],
            ]
        };

        (bl, br, tl, tr, evals)
    }

    fn new<F, R>(f: &mut F, region_function: &mut R, x0: f64, x1: f64, y0: f64, y1: f64) -> (Self, i32)
    where F: FnMut(f64, f64) -> Complex64, R: FnMut(f64, f64) -> bool {
        let mut evals = 0;
        let mut result = Complex64::new(0.0, 0.0);
        let mut error = Complex64::new(0.0, 0.0);
        let mut cache: [[Complex64; 3]; 3] = Default::default();

        // need to evaluate f(x, y) 5x5 = 25 times, of which 9 can be reused later

        for (i, (t1, w1, e1)) in Region::CLENSHAW_CURTIS_DATA.iter().enumerate() {
            let y = y0 + t1 * (y1 - y0);
            for (j, (t2, w2, e2)) in Region::CLENSHAW_CURTIS_DATA.iter().enumerate() {
                let x = x0 + t2 * (x1 - x0);
                let z = if region_function(x, y) { evals +=1; f(x, y)} else { Complex64::new(0.0, 0.0) };
                result += w1 * w2 * (y1 - y0) * (x1 - x0) * z;
                error += e1 * e2 * (y1 - y0) * (x1 - x0) * z;

                // store function for future use
                if i % 2 == 0 && j % 2 == 0 {
                    cache[i/2][j/2] = z;
                }
            }
        }

        (Self {
            integral: result,
            error: result - error,
            x: [x0, x1],
            y: [y0, y1],
            cache,
        }, evals)
    }
}

/// Integrates a complex function of two variables `f(x, y)` over a rectangular domain
/// `x0 < x < x1` and `y0 < y < y1`, returning an answer that should be accurate to
/// within `tol`.
///
/// The adaptive integrator stops when either the tolerance condition is met or the
/// number of subdivisions exceeds `max_recursion`.
pub fn integrate_2d<F, R>(mut f: F, x0: f64, x_div: Option<f64>, x1: f64, y0: f64, y_div: Option<f64>, y1: f64, mut region_function: R, tolerance: f64, max_recursion: i32) -> (Complex64, i32)
where F: FnMut(f64, f64) -> Complex64, R: FnMut(f64, f64) -> bool {
    let mut integral = Complex64::new(0.0, 0.0);
    let mut error: Complex64;
    let mut regions: Vec<Region> = Vec::with_capacity(16);

    let (region, mut count) = Region::new(&mut f, &mut region_function, x0, x1, y0, y1);

    if x_div.is_none() && y_div.is_none() {
        regions.push(region);
    } else {
        let (bl, br, tl, tr, evals) = region.custom_partition(&mut f, &mut region_function, x_div, y_div);
        count += evals;
        regions.push(bl);
        regions.push(br);
        regions.push(tl);
        regions.push(tr);
    }

    for _i in 0..max_recursion {
        (integral, error) = regions.iter().fold(
            (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)),
            |acc, e| (acc.0 + e.integral, acc.1 + e.error)
        );

        // println!(
        //     "{}: {:>3} regions, integral = {:9.3e}, frac error = {:+.3e}, {:+.3e}",
        //     i, regions.len(), integral, error.re / integral.re, error.im / integral.im,
        // );
        // print!(" {:.6e} {:.6e}", integral.re, integral.im);

        if error.re.abs() <= tolerance * integral.re.abs() && error.im.abs() <= tolerance * integral.im.abs() {
            break;
        }

        // grab the region with the largest error
        let region = regions.pop().unwrap();

        // partition into four sub regions
        let (bl, br, tl, tr, evals) = region.partition(&mut f, &mut region_function);

        count += evals;

        regions.push(bl);
        regions.push(br);
        regions.push(tl);
        regions.push(tr);

        // prep for next partition
        regions.sort_unstable_by(|a, b| {
            let a = a.error.norm_sqr();
            let b = b.error.norm_sqr();
            a.partial_cmp(&b).unwrap()
        });
    }

    // for region in &regions {
    //     println!(
    //         "{:.6} {:.6} {:.6} {:.6}",
    //         region.x[0], region.x[1], region.y[0], region.y[1]
    //     );
    // }
    // println!();
    (integral, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_integration() {
        let tol = 1.0e-6;
        let (result, count) = integrate_2d(
            |x, y| Complex64::new(1.0 / (1.0 + x * x * x + 16.0 * y * y), 0.0),
            0.0, Some(1.0), 4.0, 0.0, Some(1.0), 4.0,
            |_, _| true, tol, 32,
        );
        let target = 0.64778624241275644456;
        let error = (target - result.re).abs() / target;
        println!("result = {:.6e}, target = {:.6e} [{} evals], err = {:.3e}", result, target, count, error);
        assert!(error < tol);
    }
}