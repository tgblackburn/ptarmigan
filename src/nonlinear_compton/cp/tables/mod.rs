//! Tables for the total rate and CDF as a function of harmonic order

use crate::pwmci;

mod total;
mod cdf;

pub const LN_A_MIN: f64 = total::MIN[0];
pub const LN_ETA_MIN: f64 = total::MIN[1];

pub fn contains(a: f64, eta: f64) -> bool {
    a.ln() >= total::MIN[0] && a < 20.0 && eta.ln() >= total::MIN[1] && eta < 2.0
}

#[allow(unused_parens)]
pub fn interpolate(a: f64, eta: f64) -> f64 {
    use total::*;

    let ia = ((a.ln() - MIN[0]) / STEP[0]) as usize;
    let ie = ((eta.ln() - MIN[1]) / STEP[1]) as usize;

    assert!(ia < N_COLS - 1);
    assert!(ie < N_ROWS - 1);
    
    let da = (a.ln() - MIN[0]) / STEP[0] - (ia as f64);
    let de = (eta.ln() - MIN[1]) / STEP[1] - (ie as f64);
    let f = (
        (1.0 - da) * (1.0 - de) * TABLE[ie][ia]
        + da * (1.0 - de) * TABLE[ie][ia+1]
        + (1.0 - da) * de * TABLE[ie+1][ia]
        + da * de * TABLE[ie+1][ia+1]
    );

    f.exp()
}

fn rescale(frac: f64, table: &[[f64; 2]; 32]) -> (f64, [[f64; 2]; 31]) {
    let mut output = [[0.0; 2]; 31];
    for i in 0..31 {
        output[i][0] = table[i][0].ln();
        output[i][1] = (-1.0 * (1.0 - table[i][1]).ln()).ln();
    }
    let frac2 = (-1.0 * (1.0 - frac).ln()).ln();
    (frac2, output)
}

pub fn invert(a: f64, eta: f64, frac: f64) -> i32 {
    use cdf::*;

    if a.ln() <= MIN[0] {
        // first harmonic only
       1
    } else if eta.ln() <= MIN[1] {
        // cdf(n) is independent of eta as eta -> 0
        let ix = ((a.ln() - MIN[0]) / STEP[0]) as usize;
        let dx = (a.ln() - MIN[0]) / STEP[0] - (ix as f64);

        let index = [ix, ix + 1];
        let weight = [1.0 - dx, dx];

        let n: f64 = index.iter()
            .zip(weight.iter())
            .map(|(i, w)| {
                let table = &TABLE[*i];
                let n = if frac <= table[0][1] {
                    0.9
                } else {
                    let (rs_frac, rs_table) = rescale(frac, table);
                    pwmci::Interpolant::new(&rs_table)
                        .extrapolate(true)
                        .invert(rs_frac)
                        .map(f64::exp)
                        .unwrap()
                };
                n * w
            })
            .sum();

        n.ceil() as i32
    } else {
        let ix = ((a.ln() - MIN[0]) / STEP[0]) as usize;
        let iy = ((eta.ln() - MIN[1]) / STEP[1]) as usize;

        let dx = (a.ln() - MIN[0]) / STEP[0] - (ix as f64);
        let dy = (eta.ln() - MIN[1]) / STEP[1] - (iy as f64);

        let index = [
            N_COLS * iy + ix,
            N_COLS * iy + ix + 1,
            N_COLS * (iy + 1) + ix,
            N_COLS * (iy + 1) + (ix + 1),
        ];

        let weight = [
            (1.0 - dx) * (1.0 - dy),
            dx * (1.0 - dy),
            (1.0 - dx) * dy,
            dx * dy,
        ];

        let n_alt: f64 = index.iter()
            .zip(weight.iter())
            .map(|(i, w)| {
                let table = &TABLE[*i];
                let n = if frac <= table[0][1] {
                    0.9
                } else {
                    let (rs_frac, rs_table) = rescale(frac, table);
                    pwmci::Interpolant::new(&rs_table)
                        .extrapolate(true)
                        .invert(rs_frac)
                        .map(f64::exp)
                        .unwrap()
                };
                n * w
            })
            .sum();

        n_alt.ceil() as i32
    }
}
