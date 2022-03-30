//! Quantum synchrotron emission, e -> e + gamma, in a background field

use std::f64::consts;
use crate::constants::*;
use crate::geometry::{ThreeVector, FourVector};
use crate::pwmci;
use crate::special_functions::Airy;

mod tables;

/// Returns the quantum synchrotron rate, per unit time (in seconds)
pub fn rate(chi: f64, gamma: f64) -> f64 {
    let h = if chi < 0.01 {
        5.0 * consts::FRAC_PI_3 * (1.0 - 8.0 * chi / (5.0 * 3.0f64.sqrt()))
    } else if chi >= 100.0 {
        let cbrt_chi = chi.cbrt();
        let mut h = -1019.4661473121777 + 1786.716527650374 * cbrt_chi * cbrt_chi;
        h = 1750.6263395722715 + cbrt_chi * cbrt_chi * h;
        h = -2260.1819695887225 + cbrt_chi * h;
        h = 0.00296527643253334 * h / (chi * chi);
        h
    } else {
        let index = (chi.ln() - tables::LN_H_CHI_TABLE[0][0]) / tables::DELTA_LN_CHI;
        let weight = index.fract(); // of upper entry
        let index = index.floor() as usize;
        assert!(index < tables::LN_H_CHI_TABLE.len() - 1);
        let ln_h = (1.0 - weight) * tables::LN_H_CHI_TABLE[index][1] + weight * tables::LN_H_CHI_TABLE[index+1][1];
        ln_h.exp()
    };

    3.0f64.sqrt() * ALPHA_FINE * chi * h / (2.0 * consts::PI * gamma * COMPTON_TIME)
}

/// Returns the classical synchrotron rate, per unit time (in seconds)
#[allow(unused)]
pub fn classical_rate(chi: f64, gamma: f64) -> f64 {
    let h = 5.0 * consts::FRAC_PI_3;
    3.0f64.sqrt() * ALPHA_FINE * chi * h / (2.0 * consts::PI * gamma * COMPTON_TIME)
}

fn from_linear_cdf_table(global_zero: f64, local_zero: f64, rand: f64, cdf: &tables::CDF) -> f64 {
    // need to ensure y > local_zero
    // table runs from global_zero < y < infinity, where global_zero < local_zero

    // First, find r_zero = cdf(local_zero)
    let r_zero = if local_zero < cdf.table[0][0] {
        cdf.coeff * (local_zero - global_zero).powf(cdf.power)
    } else {
        let tmp = pwmci::evaluate(local_zero, &cdf.table);

        // none if local_zero is larger than the last entry in the table
        if tmp.is_none() {
            //println!("global_zero = {:e}, local_zero = {:e}, table upper = {:?}", global_zero, local_zero, cdf.table[30]);
            return local_zero; // cdf.table[30][0]; // jump out of function
        }

        tmp.unwrap()
    };

    // Rescale r so that it lies between r_zero and 1.
    let r = r_zero + (1.0 - r_zero) * rand;

    // Now we can solve r = cdf(y), guaranteeding local_zero < y
    let y = if r <= cdf.table[0][1] {
        let tmp = (r.ln() - cdf.coeff.ln()) / cdf.power; // ln(y-global_zero)
        tmp.exp() + global_zero
    } else if let Some(result) = pwmci::invert(r, &cdf.table) {
        result.0
    } else {
        local_zero // cdf.table[30][0]
    };

    assert!(y >= local_zero);
    y
}

/// Samples the quantum synchrotron spectrum of an electron with
/// quantum parameter `chi` and Lorentz factor `gamma`.
/// 
/// Returns a triple of the photon energy, in units of mc^2,
/// and the polar and azimuthal angles of emission, in the range
/// [0,pi] and [0,2pi] respectively.
///
/// If the polar angle would be larger than pi (which is possible
/// at very low energy), then None is returned instead.
pub fn sample(chi: f64, gamma: f64, rand1: f64, rand2: f64, rand3: f64) -> (f64, Option<f64>, f64) {
    use tables::{LN_CHI_MIN, LN_CHI_STEP, QUANTUM_CDF};
    use tables::{LN_DELTA_MIN, LN_DELTA_STEP, Y_CDF, Y_INFINITE_DELTA_CDF};

    // index of closest tabulated chi
    let index = (chi.ln() - LN_CHI_MIN) / LN_CHI_STEP;
    let weight = index.fract(); // of upper table
    let index: usize = index.floor() as usize;
    //println!("index = {}, weight = {}", index, weight);

    if chi.ln() <= LN_CHI_MIN {
        let (omega_mc2, theta, cphi) = classical_sample(chi, gamma, rand1, rand2, rand3);
        // omega = u gamma m classically, but u/(1+u) gamma m in QED
        (omega_mc2 * gamma / (gamma + omega_mc2), theta, cphi)
    } else if index >= QUANTUM_CDF.len() - 1 {
        unimplemented!();
    } else {
        // First sample u from r_1 = cdf(u; chi)
        let lower = &QUANTUM_CDF[index];
        //println!("lrand = {:e}, bounds = {:e}, {:e}", rand1.ln(), lower.table[0][1], lower.table[30][1]);
        let ln_u_lower = if rand1.ln() <= lower.table[0][1] {
            (rand1.ln() - lower.coeff.ln()) / lower.power
        } else if let Some(result) = pwmci::invert(rand1.ln(), &lower.table) {
            result.0
        } else {
            lower.table[30][0] // clip to last tabulated ln_u
        };

        let upper = &QUANTUM_CDF[index+1];
        let ln_u_upper = if rand1.ln() <= upper.table[0][1] {
            (rand1.ln() - upper.coeff.ln()) / upper.power
        } else if let Some(result) = pwmci::invert(rand1.ln(), &upper.table) {
            result.0
        } else {
            upper.table[30][0]
        };

        let u = ((1.0 - weight) * ln_u_lower + weight * ln_u_upper).exp();
        //println!("rand1 = {}, u = {}", rand1, u);

        // Now get the angle, sampling r_2 = cdf(z|u; chi)
        let beta = 2.0 * u / (3.0 * chi);
        let delta = (1.0 + (1.0 + u).powi(2)) * beta.powf(-2.0/3.0) / (1.0 + u);

        let index = (delta.ln() - LN_DELTA_MIN) / LN_DELTA_STEP;
        let weight = index.fract(); // of upper table
        let index: usize = index.floor() as usize;

        // Now, delta_min = 0.1, which is guaranteed to cover ALL u for chi > 0.01.
        // So it's impossible for index to be 'negative'.
        // But it is possible for delta > 100.0, off the end of the table.
        // Handle this specially

        let y = if index >= Y_CDF.len() - 1 {
            from_linear_cdf_table(0.0, beta, rand2, &Y_INFINITE_DELTA_CDF)
        } else {
            let y_lower = from_linear_cdf_table(delta.powf(-1.5), beta, rand2, &Y_CDF[index]);
            let y_upper = from_linear_cdf_table(delta.powf(-1.5), beta, rand2, &Y_CDF[index+1]);
            //assert!(y_lower >= beta);
            //assert!(y_upper >= beta);
            (1.0 - weight) * y_lower + weight * y_upper
        };

        let z = (y / beta).max(1.0);
        //if z < 1.0 {
        //    println!("got f = {:e}, beta = {:e}, delta = {:e}, returning y = {:e}, z = {:e}", u / (1.0 + u), beta, delta, y, z);
        //}
        //assert!(y >= beta);
        //assert!(z >= 1.0);
        let cos_theta = (gamma - z.powf(2.0/3.0) / (2.0 * gamma)) / (gamma.powi(2) - 1.0).sqrt();
        let theta = if cos_theta >= 1.0 {
            Some(0.0)
        } else if cos_theta >= -1.0 {
            Some(cos_theta.acos())
        } else {
            None
        };
        //let theta = cos_theta.min(1.0f64).max(-1.0f64).acos(); // if cos_theta is NaN, it's replaced with 1.0 by the first min

        (gamma * u / (1.0 + u), theta, 2.0 * consts::PI * rand3)
    }
}

/// Returns the Stokes vector of the photon with four-momentum `k` (normalized to the
/// electron mass), assuming that it was emitted by an electron with quantum parameter `chi`,
/// Lorentz factor `gamma`, velocity `v` and instantaneous acceleration `w`.
///
/// The basis is defined with respect to a vector in the `x`-`z` plane that is perpendicular
/// to the photon three-momentum.
pub fn stokes_parameters(k: FourVector, chi: f64, gamma: f64, v: ThreeVector, w: ThreeVector) -> FourVector {
    // belt and braces
    let v = v.normalize();
    let w = w.normalize();
    let n = ThreeVector::from(k).normalize();

    // u = omega / (e - omega)
    let u = k[0] / (gamma - k[0]);

    // angle b/t k and plane (v, w)
    let beta = {
        let q = v.cross(w).normalize();
        (n * q).asin()
    };
    // println!("gamma beta = {:.3e}", gamma * beta);

    let mu = (beta * beta + 1.0 / (gamma * gamma)).sqrt();
    let eta = u * (gamma * mu).powi(3) / (3.0 * chi);

    // K(1/3, eta) = pi (2 sqrt(3) / x)^(1/3) Ai[ (1.5 x)^(2/3) ]
    let k1_3 = consts::PI * (2.0 * 3.0f64.sqrt() / eta).cbrt() * (1.5 * eta).powf(2.0 / 3.0).ai().unwrap_or(0.0);

    // K(2/3, eta) = -pi / 3^(1/6) (2 / x)^(2/3) Ai'[ (1.5 x)^(2/3) ]
    let k2_3 = -consts::PI * 3.0f64.powf(-1.0/6.0) * (2.0 / eta).powf(2.0 / 3.0) * (1.5 * eta).powf(2.0 / 3.0).ai_prime().unwrap_or(0.0);

    // println!("eta = {:.3e}, K(1/3, eta) = {:.3e}, K(2/3, eta) = {:.3e}", eta, k1_3, k2_3);

    let g = 0.5 * (u * mu).powi(2) * (k1_3 * k1_3 + k2_3 * k2_3) / (1.0 + u) + (mu * k2_3).powi(2) + (beta * k1_3).powi(2);

    let xi = ThreeVector::new(
        ((mu * k2_3).powi(2) - (beta * k1_3).powi(2)) / g,
        0.0,
        2.0 * (1.0 + 0.5 * u * u / (1.0 + u)) * beta * mu * k1_3 * k2_3 / g,
    );

    // println!("before rotation: |xi| = {:.3e}, xi = [{:.3e} {:.3e} {:.3e}]", xi.norm_sqr().sqrt(), xi[0], xi[1], xi[2]);

    // xi is defined w.r.t. the basis [w - (n.w) n, n, w x n], whereas
    // we want [e, n, e x n], where e is in the x-z plane.
    // phi rotates w - (n.w)n down to the x-z plane - we don't care which
    // direction in the x-z plane, because this is fixed by k
    let phi = ((w - (n * w) * n) * ThreeVector::new(0.0, 1.0, 0.0)).asin();
    // println!("rotating w by {:.3e}", phi);

    // which rotates the Stokes parameters through 2 phi
    let xi = ThreeVector::new(
        (2.0 * phi).cos() * xi[0] + (2.0 * phi).sin() * xi[1],
        -(2.0 * phi).sin() * xi[0] + (2.0 * phi).cos() * xi[1],
        xi[2],
    );

    // println!("after rotation: |xi| = {:.3e}, xi = [{:.3e} {:.3e} {:.3e}]", xi.norm_sqr().sqrt(), xi[0], xi[1], xi[2]);
    [1.0, xi[0], xi[1], xi[2]].into()
}

/// Samples the classical synchrotron spectrum of an electron with
/// quantum parameter `chi` and Lorentz factor `gamma`.
/// 
/// Returns a triple of the photon energy, in units of mc^2,
/// and the polar and azimuthal angles of emission, in the range
/// [0,pi] and [0,2pi] respectively.
/// 
/// As there is no hbar-dependent cutoff, the energy of the photon
/// can exceed that of the electron.
pub fn classical_sample(chi: f64, gamma: f64, rand1: f64, rand2: f64, rand3: f64) -> (f64, Option<f64>, f64) {
    // First determine z:
    // z^(1/3) = (2 + 4 cos(delta/3)) / (5 (1-r)) where 0 <= r < 1
    // and cos(delta) = (-9 + 50r - 25r^2) / 16
    let delta = ((-9.0 + 50.0 * rand2 - 25.0 * rand2.powi(2)) / 16.0).acos();
    let z = ((2.0 + 4.0 * (delta/3.0).cos()) / (5.0 * (1.0 - rand2))).powi(3);

    // now invert cdf(u|z) = (3/pi) \int_0^x t K_{1/3}(t) dt,
    // which is tabulated, to obtain x = 2 u z / (3 chi)
    // for x < 0.01, cdf(u|z) =

    let ln_rand = rand1.ln();
    let x = if ln_rand < tables::CLASSICAL_SPECTRUM_TABLE[0][1] {
        1.020377255 * rand1.powf(0.6)
    } else {
        //println!("Inverting ln(rand = {}) = {}", rand1, ln_rand);
        let (ln_x, _) = pwmci::invert(ln_rand, &tables::CLASSICAL_SPECTRUM_TABLE)
            .unwrap_or( (tables::CLASSICAL_SPECTRUM_TABLE.last().unwrap()[0],1) );
        ln_x.exp()
    };

    let u = 3.0 * chi * x / (2.0 * z);
    let omega_mc2 = u * gamma;

    let cos_theta = (gamma - z.powf(2.0/3.0) / (2.0 * gamma)) / (gamma.powi(2) - 1.0).sqrt();
    let theta = if cos_theta >= 1.0 {
        Some(0.0)
    } else if cos_theta >= -1.0 {
        Some(cos_theta.acos())
    } else {
        None
    };

    (omega_mc2, theta, 2.0 * consts::PI * rand3)
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_xoshiro::*;
    use crate::{Particle, Species};
    use super::*;

    #[test]
    fn rate_0_026() {
        let value = rate(0.026, 1000.0);
        let target = 2.07935e14;
        println!("rate(chi = 0.026, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }

    #[test]
    fn rate_3_5() {
        let value = rate(3.5, 1000.0);
        let target = 1.58485e16;
        println!("rate(chi = 3.5, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }

    #[test]
    fn rate_9_98() {
        let value = rate(9.98, 1000.0);
        let target = 3.45844e16;
        println!("rate(chi = 9.98, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }

    #[test]
    fn rate_12_4() {
        let value = rate(12.4, 1000.0);
        let target = 4.04647e16;
        println!("rate(chi = 12.4, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }

    #[test]
    fn rate_403() {
        let value = rate(403.0, 1000.0);
        let target = 4.46834e17;
        println!("rate(chi = 403, gamma = 1000) = {:e}, target = {:e}, error = {:e}", value, target, ((value - target) / target).abs() );
        assert!( ((value - target) / target).abs() < 1.0e-3 );
    }

    #[test]
    fn stokes_vector() {
        let (chi, gamma) = (1.0, 1000.0);
        let w: ThreeVector = [1.0, 0.0, 0.0].into();
        let long: ThreeVector = [0.0, 0.0, 1.0].into();
        let perp: ThreeVector = [1.0, 0.0, 0.0].into();
        let mut rng = Xoshiro256StarStar::seed_from_u64(0);

        let (omega_mc2, _, _) = sample(chi, gamma, 0.98, rng.gen(), rng.gen());
        println!("Sampling at omega/(m gamma) = {:.3e}...", omega_mc2 / gamma);

        // integrating over all angles is expected to yield a Stokes vector
        // [1.0, (W_11 - W_22) / (W_11 + W_22), 0, 0]
        let sv: FourVector = (0..10_000)
            .map(|_| {
                // sample at fixed energy
                let (omega_mc2, theta, cphi) = sample(chi, gamma, 0.8, rng.gen(), rng.gen());
                let theta = theta.unwrap();
                //println!("omega/(m gamma) = {:.2e}, gamma theta = {:.3e}, phi = {:.3e}", omega_mc2 / gamma, gamma * theta, cphi);

                // photon four-momentum
                let perp = perp.rotate_around(long, cphi);
                let k: ThreeVector = omega_mc2 * (theta.cos() * long + theta.sin() * perp);
                let k = FourVector::lightlike(k[0], k[1], k[2]);

                stokes_parameters(k, chi, gamma, long, w) / 10_000_f64
            })
            .fold([0.0; 4].into(), |a, b| a + b);

        println!("Summed Stokes vector = [{:.3e} {:.3e} {:.3e} {:.3e}]", sv[0], sv[1], sv[2], sv[3]);
        assert!(sv[2].abs() < 1.0e-3 && sv[3].abs() < 1.0e-3);

        let photon = Particle::create(Species::Photon, [0.0; 4].into())
            .with_normalized_momentum([omega_mc2, 0.0, 0.0, omega_mc2].into())
            .with_polarization(Some(sv));

        let pol_x = photon.polarization_along_x();
        let pol_y = photon.polarization_along_y();
        println!("Projected onto x = {:.3e}, y = {:.3e}, total = {:.3e}", pol_x, pol_y, pol_x + pol_y);
        assert!(pol_x + pol_y == 1.0);
    }

    // #[test]
    // fn classical_spectrum() {
    //     use rand::prelude::*;
    //     use rand_xoshiro::*;
    //     use std::fs::File;
    //     use std::io::Write;

    //     let chi = 0.01;
    //     let gamma = 1000.0;
    //     let mut rng = Xoshiro256StarStar::seed_from_u64(0);
    //     let mut results: Vec<(f64, f64)> = Vec::new();

    //     // 2_000_000 for something more stringent
    //     for _i in 0..100_000 {
    //         let (omega_mc2, theta, _) = classical_sample(chi, gamma, rng.gen(), rng.gen(), rng.gen());
    //         results.push((omega_mc2 / gamma, gamma * theta));
    //     }

    //     let mut file = File::create("ClassicalSpectrumTest.dat").unwrap();
    //     for result in &results {
    //         writeln!(file, "{} {}", result.0, result.1).unwrap();
    //     }
    // }

    // #[test]
    // fn quantum_spectrum() {
    //     use rand::prelude::*;
    //     use rand_xoshiro::*;
    //     use std::fs::File;
    //     use std::io::Write;
    //     use crate::particle::hgram::*;

    //     let chi = 0.07;
    //     let gamma = 1000.0;
    //     let gamma_theta_max = 10.0;
    //     let mut rng = Xoshiro256StarStar::seed_from_u64(0);
    //     let mut results: Vec<(f64, f64)> = Vec::new();

    //     // 4_000_000 for something more stringent
    //     for _i in 0..100_000 {
    //         let (omega_mc2, theta, _) = sample(chi, gamma, rng.gen(), rng.gen(), rng.gen());
    //         results.push((omega_mc2 / gamma, gamma * theta));
    //     }

    //     let mut file = File::create("QuantumSpectrumTest.dat").unwrap();
    //     for result in &results {
    //         writeln!(file, "{} {}", result.0, result.1).unwrap();
    //     }

    //     let universe = mpi::initialize().unwrap();
    //     let world = universe.world();

    //     let first = Box::new(|t: &(f64, f64)| t.0.ln()) as Box<dyn Fn(&(f64,f64)) -> f64>;
    //     let second = Box::new(|t: &(f64, f64)| t.1.ln()) as Box<dyn Fn(&(f64,f64)) -> f64>;
    //     let hgram = Histogram::generate_2d(&world, &results, [&first, &second], &|_t| 1.0, ["omega/mc^2", "gamma theta"], ["1", "1"], [BinSpec::Automatic, BinSpec::Automatic], HeightSpec::ProbabilityDensity).unwrap();
    //     hgram.write_fits("!QuantumSpectrumTest_LogScaled.fits").unwrap();

    //     //let subset: Vec<(f64, f64)> = results.into_iter().filter(|t| t.1 < 10.0).collect();
    //     let subset: Vec<(f64, f64)> = results
    //         .iter()
    //         .map(|t|
    //             if t.1 > gamma_theta_max {
    //                 (t.0, std::f64::NAN)
    //             } else {
    //                 *t
    //             })
    //         .collect();

    //     let first = Box::new(|t: &(f64, f64)| t.0) as Box<dyn Fn(&(f64,f64)) -> f64>;
    //     let second = Box::new(|t: &(f64, f64)| t.1) as Box<dyn Fn(&(f64,f64)) -> f64>;
    //     let hgram = Histogram::generate_2d(&world, &subset, [&first, &second], &|_t| 1.0, ["omega/mc^2", "gamma theta"], ["1", "1"], [BinSpec::Automatic, BinSpec::Automatic], HeightSpec::ProbabilityDensity).unwrap();
    //     hgram.write_fits("!QuantumSpectrumTest.fits").unwrap();
    // }
}