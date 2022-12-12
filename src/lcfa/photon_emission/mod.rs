//! Quantum synchrotron emission, e -> e + gamma, in a background field

use std::f64::consts;
use crate::constants::*;
use crate::geometry::{ThreeVector, FourVector, StokesVector};
use crate::pwmci;
use crate::special_functions::Airy;

mod tables;
pub mod classical;

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

fn from_linear_cdf_table(global_zero: f64, local_zero: f64, rand: f64, cdf: &tables::CDF) -> f64 {
    // need to ensure y > local_zero
    // table runs from global_zero < y < infinity, where global_zero < local_zero

    // First, find r_zero = cdf(local_zero)
    let r_zero = if local_zero < cdf.table[0][0] {
        cdf.coeff * (local_zero - global_zero).powf(cdf.power)
    } else {
        let tmp = pwmci::Interpolant::new(&cdf.table).evaluate(local_zero);

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
    } else if let Some(result) = pwmci::Interpolant::new(&cdf.table).invert(r) {
        result
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
        let (omega_mc2, theta, cphi) = classical::sample(chi, gamma, rand1, rand2, rand3);
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
        } else if let Some(result) = pwmci::Interpolant::new(&lower.table).invert(rand1.ln()) {
            result
        } else {
            lower.table[30][0] // clip to last tabulated ln_u
        };

        let upper = &QUANTUM_CDF[index+1];
        let ln_u_upper = if rand1.ln() <= upper.table[0][1] {
            (rand1.ln() - upper.coeff.ln()) / upper.power
        } else if let Some(result) = pwmci::Interpolant::new(&upper.table).invert(rand1.ln()) {
            result
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
pub fn stokes_parameters(k: FourVector, chi: f64, gamma: f64, v: ThreeVector, w: ThreeVector) -> StokesVector {
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

    let k1_3 = eta.bessel_K_1_3().unwrap_or(0.0);
    let k2_3 = eta.bessel_K_2_3().unwrap_or(0.0);

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

        let rand1 = 0.98;
        let (omega_mc2, _, _) = sample(chi, gamma, rand1, rng.gen(), rng.gen());
        println!("Sampling at omega/(m gamma) = {:.3e}...", omega_mc2 / gamma);

        // integrating over all angles is expected to yield a Stokes vector
        // [1.0, (W_11 - W_22) / (W_11 + W_22), 0, 0]
        let sv: StokesVector = (0..10_000)
            .map(|_| {
                // sample at fixed energy
                let (omega_mc2, theta, cphi) = sample(chi, gamma, rand1, rng.gen(), rng.gen());
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

        // Check whether sv[1] = (W_11 - W_22) / (W_11 + W_22)
        let u = omega_mc2 / (gamma - omega_mc2);

        // K(2/3, xi)
        let k2_3: f64 = GL_NODES.iter()
            .zip(GL_WEIGHTS.iter())
            .map(|(t, w)| {
                let xi = 2.0 * u / (3.0 * chi);
                w * (-xi * t.cosh() + t).exp() * (2.0 * t / 3.0).cosh()
            })
            .sum();

        // int_xi^infty K(5/3, y) dy
        let int_k5_3: f64 = GL_NODES.iter()
            .zip(GL_WEIGHTS.iter())
            .map(|(t, w)| {
                let xi = 2.0 * u / (3.0 * chi);
                w * (-xi * t.cosh() + t).exp() * (5.0 * t / 3.0).cosh() / t.cosh()
            })
            .sum();

        let target = k2_3 / (u * u * k2_3 / (1.0 + u) + int_k5_3);
        let error = (sv[1] - target).abs() / target;
        println!("Got sv[1] = {:.3e}, expected {:.3e}, error = {:.2}%", sv[1], target, 100.0 * error);
        assert!(error < 0.01);

        // Finally, project Stokes parameters onto detector in the x-y plane
        let photon = Particle::create(Species::Photon, [0.0; 4].into())
            .with_normalized_momentum([omega_mc2, 0.0, 0.0, omega_mc2].into())
            .with_polarization(sv);

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

#[allow(unused)]
static GL_NODES: [f64; 32] = [
	4.448936583326702e-2,
	2.345261095196185e-1,
	5.768846293018864e-1,
	1.072448753817818e+0,
	1.722408776444645e+0,
	2.528336706425795e+0,
	3.492213273021994e+0,
	4.616456769749767e+0,
	5.903958504174244e+0,
	7.358126733186241e+0,
	8.982940924212596e+0,
	1.078301863253997e+1,
	1.276369798674273e+1,
	1.493113975552256e+1,
	1.729245433671531e+1,
	1.985586094033605e+1,
	2.263088901319677e+1,
	2.562863602245925e+1,
	2.886210181632347e+1,
	3.234662915396474e+1,
	3.610049480575197e+1,
	4.014571977153944e+1,
	4.450920799575494e+1,
	4.922439498730864e+1,
	5.433372133339691e+1,
	5.989250916213402e+1,
	6.597537728793505e+1,
	7.268762809066271e+1,
	8.018744697791352e+1,
	8.873534041789240e+1,
	9.882954286828397e+1,
	1.117513980979377e+2,
];

#[allow(unused)]
static GL_WEIGHTS: [f64; 32] = [
	1.092183419523850e-1,
	2.104431079388132e-1,
	2.352132296698480e-1,
	1.959033359728810e-1,
	1.299837862860718e-1,
	7.057862386571744e-2,
	3.176091250917507e-2,
	1.191821483483856e-2,
	3.738816294611525e-3,
	9.808033066149551e-4,
	2.148649188013642e-4,
	3.920341967987947e-5,
	5.934541612868633e-6,
	7.416404578667552e-7,
	7.604567879120781e-8,
	6.350602226625807e-9,
	4.281382971040929e-10,
	2.305899491891336e-11,
	9.799379288727094e-13,
	3.237801657729266e-14,
	8.171823443420719e-16,
	1.542133833393823e-17,
	2.119792290163619e-19,
	2.054429673788045e-21,
	1.346982586637395e-23,
	5.661294130397359e-26,
	1.418560545463037e-28,
	1.913375494454224e-31,
	1.192248760098222e-34,
	2.671511219240137e-38,
	1.338616942106256e-42,
	4.510536193898974e-48,
];