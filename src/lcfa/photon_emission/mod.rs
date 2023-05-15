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
    let e1 = {
        let e1 = ThreeVector::new(k[3], 0.0, -k[1]);
        if e1.norm_sqr() == 0.0 {
            [1.0, 0.0, 0.0].into()
        } else {
            e1.normalize()
        }
    };

    let phi = (e1 * w).clamp(-1.0, 1.0).acos();
    // if sign positive, positive rotation by phi brings w to e1.
    let sign = (w.cross(e1) * n).signum();
    let phi = sign * phi;

    // which rotates Stokes parameters by -2 * phi
    let xi = ThreeVector::new(
        (2.0 * phi).cos() * xi[0] + (2.0 * phi).sin() * xi[1],
        -(2.0 * phi).sin() * xi[0] + (2.0 * phi).cos() * xi[1],
        xi[2],
    );

    // println!("after rotation: |xi| = {:.3e}, xi = [{:.3e} {:.3e} {:.3e}]", xi.norm_sqr().sqrt(), xi[0], xi[1], xi[2]);
    [1.0, xi[0], xi[1], xi[2]].into()
}

static GAUNT_FACTOR_TABLE: [[f64; 2]; 25] = [
    [-6.907755278982137, -0.005923910174592344],
    [-6.332109005733626, -0.01049345707375475],
    [-5.756462732485114, -0.01853326677632520],
    [-5.180816459236603, -0.03256813342821585],
    [-4.605170185988091, -0.05674922246157141],
    [-4.029523912739580, -0.09754530350543557],
    [-3.453877639491069, -0.1642291108677053],
    [-2.878231366242557, -0.2685646952510789],
    [-2.302585092994046, -0.4231867286719710],
    [-1.726938819745534, -0.6390383238080004],
    [-1.151292546497023, -0.9232385209997736],
    [-0.575646273248511, -1.278436642040118],
    [ 0.000000000000000, -1.703334719200724],
    [ 0.575646273248511, -2.193501426179741],
    [ 1.151292546497023, -2.742164851781891],
    [ 1.726938819745534, -3.341092923053680],
    [ 2.302585092994046, -3.981546623633449],
    [ 2.878231366242557, -4.655120578676737],
    [ 3.453877639491069, -5.354319063765299],
    [ 4.029523912739580, -6.072838862057530],
    [ 4.605170185988091, -6.805618017945314],
    [ 5.180816459236603, -7.548736622847868],
    [ 5.756462732485114, -8.299245624917244],
    [ 6.332109005733626, -9.054976772950048],
    [ 6.907755278982137, -9.814364636712657],
];

/// Returns the Gaunt factor `g(χ)`, the ratio between the radiated power in the quantum and
/// classical cases.
pub fn gaunt_factor(chi: f64) -> f64 {
    if chi <= 1.0e-3 {
        1.0 - 55.0 * 3_f64.sqrt() * chi / 16.0 + 48.0 * chi * chi
    } else if chi > 500.0 {
        let coeff = 0.5563810015417746;
        let chi_1_3 = chi.cbrt();
        let g = -1.0000341464898563 + coeff * chi_1_3 * chi_1_3;
        let g = 2.6235881916750565 + chi_1_3 * chi_1_3 * g;
        let g = -3.8106418922532588 + chi_1_3 * g;
        g / chi.powi(3)
    } else {
        pwmci::Interpolant::new(&GAUNT_FACTOR_TABLE)
            .evaluate(chi.ln())
            .map(f64::exp)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_xoshiro::*;
    use crate::{Particle, Species};
    use crate::quadrature::{GL_NODES, GL_WEIGHTS};
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

    #[test]
    fn gaunt_factor_table() {
        let pts = [
            (0.0025, 0.98540791120703309744),
            (0.012, 0.93473708092320246516),
            (0.12, 0.61600532094295304700),
            (0.7, 0.23882354190470674907),
            (2.5, 0.081136810639749760802),
            (7.0, 0.027867522424020332230),
            (12.0, 0.015120883496044083748),
            (25.0, 0.0063062413153582144900),
            (70.0, 0.0017465062499607358868),
            (250.0, 0.00033811666381767191642),
        ];

        for (chi, target) in pts.iter() {
            let result = gaunt_factor(*chi);
            let error = (result - target).abs() / target;
            println!("g({:>6}) = {:.6e} [expected {:.6e}], error = {:.3}%", chi, result, target, 100.0 * error);
            assert!(error < 1.0e-3);
        }
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
