use std::error::Error;
use std::path::{Path, PathBuf};
use std::f64::consts;

#[cfg(feature = "with-mpi")]
use mpi::traits::*;
#[cfg(not(feature = "with-mpi"))]
mod no_mpi;
#[cfg(not(feature = "with-mpi"))]
use no_mpi::*;
#[cfg(not(feature = "with-mpi"))]
use no_mpi as mpi;

use rand::prelude::*;
use rand_xoshiro::*;

#[cfg(feature = "hdf5-output")]
unzip_n::unzip_n!(pub 6);
#[cfg(feature = "hdf5-output")]
unzip_n::unzip_n!(pub 7);
#[cfg(feature = "hdf5-output")]
unzip_n::unzip_n!(pub 8);

mod constants;
mod field;
mod geometry;
mod particle;
mod nonlinear_compton;
mod pair_creation;
mod lcfa;
mod special_functions;
mod output;
mod input;

use constants::*;
use field::*;
use geometry::*;
use particle::*;
use output::*;
use input::*;

/// Specifies how to print information for all particles,
/// as requested by 'dump_all_particles' in the input file.
#[derive(Copy,Clone,PartialEq)]
enum OutputMode {
    None,
    PlainText,
    #[cfg(feature = "hdf5-output")]
    Hdf5,
}

#[allow(unused)]
fn collide<F: Field, R: Rng>(field: &F, incident: Particle, rng: &mut R, dt_multiplier: f64, current_id: &mut u64, rate_increase: f64, discard_bg_e: bool, rr: bool, tracking_photons: bool) -> Shower {
    let mut primaries = vec![incident];
    let mut secondaries: Vec<Particle> = Vec::new();
    let dt = field.max_timestep().unwrap_or(1.0);
    let dt = dt * dt_multiplier;
    let primary_id = incident.id();

    while let Some(mut pt) = primaries.pop() {
        match pt.species() {
            Species::Electron | Species::Positron => {
                while field.contains(pt.position()) {
                    let (r, mut u, dt_actual) = field.push(
                        pt.position(),
                        pt.normalized_momentum(),
                        pt.charge_to_mass_ratio(),
                        dt
                    );

                    if let Some((k, u_prime, a_eff)) = field.radiate(r, u, dt_actual, rng) {
                        let id = *current_id;
                        *current_id = *current_id + 1;
                        let photon = Particle::create(Species::Photon, r)
                            .with_payload(a_eff)
                            .with_weight(pt.weight())
                            .with_id(id)
                            .with_parent_id(pt.id())
                            .with_normalized_momentum(k);
                        primaries.push(photon);

                        if rr {
                            u = u_prime;
                        }

                        pt.update_interaction_count(1.0);
                    }

                    pt.with_position(r);
                    pt.with_normalized_momentum(u);
                }

                if pt.id() != primary_id || !discard_bg_e || pt.interaction_count() > 0.0 {
                    secondaries.push(pt);
                }
            },

            Species::Photon => {
                let mut has_decayed = false;
                while field.contains(pt.position()) && !has_decayed && tracking_photons {
                    let ell = pt.normalized_momentum();
                    let r: FourVector = pt.position() + SPEED_OF_LIGHT * ell * dt / ell[0];

                    let (prob, frac, momenta) = field.pair_create(r, ell, dt, rng, rate_increase);
                    if let Some((q_e, q_p, a_eff)) = momenta {
                        let id = *current_id;
                        *current_id = *current_id + 2;
                        let electron = Particle::create(Species::Electron, r)
                            .with_weight(frac * pt.weight())
                            .with_id(id)
                            .with_payload(a_eff)
                            .with_parent_id(pt.id())
                            .with_normalized_momentum(q_e);
                        let positron = Particle::create(Species::Positron, r)
                            .with_weight(frac * pt.weight())
                            .with_id(id + 1)
                            .with_payload(a_eff)
                            .with_parent_id(pt.id())
                            .with_normalized_momentum(q_p);
                        primaries.push(electron);
                        primaries.push(positron);
                        pt.with_weight(pt.weight() * (1.0 - frac));
                        if pt.weight() <= 0.0 {
                            has_decayed = true;
                        }
                    }

                    pt.update_interaction_count(prob);
                    pt.with_position(r);
                }

                if !has_decayed {
                    secondaries.push(pt);
                }
            }
        }
    }

    Shower {
        primary: incident,
        secondaries,
    }
}

/// Returns the ratio of the pair creation and photon emission rates,
/// for a photon (or electron) with normalized energy `gamma` in a
/// laser with amplitude `a0` and wavelength `wavelength`.
fn increase_pair_rate_by(gamma: f64, a0: f64, wavelength: f64) -> f64 {
    let kappa: FourVector = SPEED_OF_LIGHT * COMPTON_TIME * 2.0 * consts::PI * FourVector::new(1.0, 0.0, 0.0, 1.0) / wavelength;
    let ell: FourVector = FourVector::lightlike(0.0, 0.0, -gamma);
    let u: FourVector = FourVector::new(0.0, 0.0, 0.0, -gamma).unitize();
    let q: FourVector = u + a0 * a0 * kappa / (2.0 * kappa * u);
    let dt = wavelength / SPEED_OF_LIGHT;
    let pair_rate = pair_creation::probability(ell, kappa, a0, dt);
    let photon_rate = nonlinear_compton::probability(kappa, q, dt);
    if pair_rate.is_none() || photon_rate.is_none() {
        1.0
    } else {
        let ratio = photon_rate.unwrap() / pair_rate.unwrap();
        //println!("P_pair = {:.6e}, P_photon = {:.6e}, ratio = {:.3}", pair_rate.unwrap(), photon_rate.unwrap(), ratio);
        ratio.max(1.0)
    }
}

fn increase_lcfa_pair_rate_by(gamma: f64, a0: f64, wavelength: f64) -> f64 {
    let omega_mc2 = 1.26e-6 / (ELECTRON_MASS_MEV * 1.0e6 * wavelength);
    let chi = 2.0 * gamma * a0 * omega_mc2;
    let pair_rate = lcfa::pair_creation::rate(chi, gamma);
    let photon_rate = lcfa::photon_emission::rate(chi, gamma);
    photon_rate / pair_rate
}

fn main() -> Result<(), Box<dyn Error>> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let id = world.rank();
    let ntasks = world.size();

    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or(InputError::file())?;
    let path = PathBuf::from(path);
    let output_dir = path.parent().unwrap_or(Path::new("")).to_str().unwrap_or("");

    // Read input configuration with default context

    let raw_input = std::fs::read_to_string(&path)
        .map_err(|_| InputError::file())?;
    let mut input = Config::from_string(&raw_input)?;
    //let mut input = Config::from_file(&path)?;
    input.with_context("constants");

    let dt_multiplier = input.read("control:dt_multiplier").unwrap_or(1.0);
    let multiplicity: Option<usize> = input.read("control:select_multiplicity").ok();
    let using_lcfa = input.read("control:lcfa").unwrap_or(false);
    let rng_seed = input.read("control:rng_seed").unwrap_or(0usize);
    let finite_bandwidth = input.read("control:bandwidth_correction").unwrap_or(false);
    let rr = input.read("control:radiation_reaction").unwrap_or(true);
    let tracking_photons = input.read("control:pair_creation").unwrap_or(true);

    let a0: f64 = input.read("laser:a0")?;
    let wavelength: f64 = input
        .read("laser:wavelength")
        .or_else(|_e|
            // attempt to read a frequency instead, e.g. 'omega: 1.55 * eV'
            input.read("laser:omega").map(|omega: f64| 2.0 * consts::PI * COMPTON_TIME * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3) / omega)
        )?;

    let pol = match input.read::<String,_>("laser:polarization") {
        Ok(s) if s == "linear" => Polarization::Linear,
        Ok(s) if s == "circular" => Polarization::Circular,
        _ => Polarization::Circular
    };

    if !using_lcfa && pol == Polarization::Linear {
        panic!("LMA rates are implemented for circularly polarized waves only!");
    }

    let (focusing, waist) = input
        .read("laser:waist")
        .map(|w| (true, w))
        .unwrap_or((false, std::f64::INFINITY));

    let tau: f64 = if focusing && !cfg!(feature = "cos2-envelope-in-3d") {
        input.read("laser:fwhm_duration")?
    } else {
        input.read("laser:n_cycles")?
    };

    let chirp_b = if !focusing {
        input.read("laser:chirp_coeff").unwrap_or(0.0)
    } else {
        input.read("laser:chirp_coeff")
            .map(|_: f64| {
                eprintln!("Chirp parameter ignored for focusing laser pulses.");
                0.0
            })
            .unwrap_or(0.0)
    };

    let npart: usize = input.read("beam:n")
        .or_else(|_| input.read("beam:ne"))
        ?;
    let gamma: f64 = input.read("beam:gamma")?;
    let sigma: f64 = input.read("beam:sigma").unwrap_or(0.0);
    let length: f64 = input.read("beam:length").unwrap_or(0.0);
    let angle: f64 = input.read("beam:collision_angle").unwrap_or(0.0);
    let rms_div: f64 = input.read("beam:rms_divergence").unwrap_or(0.0);
    let weight = input.read("beam:charge")
        .map(|q: f64| q.abs() / (constants::ELEMENTARY_CHARGE * (npart as f64)))
        .unwrap_or(1.0);
    let (radius, normally_distributed) = input.read::<Vec<String>,_>("beam:radius")
        .and_then(|vs| {
            // whether a single f64 or a tuple of [f64, dstr],
            // the first value must be the radius
            let radius = vs.first().map(|s| input.evaluate(s)).flatten();
            // a second entry, if present, is a distribution spec
            let normally_distributed = match vs.get(1) {
                None => Some(true), // if not specified at all, assume normally distributed
                Some(s) if s == "normally_distributed" => Some(true),
                Some(s) if s == "uniformly_distributed" => Some(false),
                _ => None // anything else is an error
            };
            if let (Some(r), Some(b)) = (radius, normally_distributed) {
                Ok((r, b))
            } else {
                eprintln!("Beam radius must be specified with a single numerical value, e.g.,\n\
                            \tradius: 2.0e-6\n\
                            or as a numerical value and a distribution, e.g,\n\
                            \tradius: [2.0e-6, uniformly_distributed]\n\
                            \tradius: [2.0e-6, normally_distributed].");
                Err(InputError::conversion("beam:radius", "radius"))
                //Err(ConfigError::raise(ConfigErrorKind::ConversionFailure, "beam", "radius"))
            }
        })?;
    let species = input.read::<String,_>("beam:species")
        .map_or_else(
            |e| match e.kind() {
                // if the species is not specified, default to electron
                InputErrorKind::Location => Ok(Species::Electron),
                _ => Err(e)
            },
            |s| s.parse::<Species>().map_err(|_| InputError::conversion("beam:species", "species"))
        )?;
    let use_brem_spec = if species == Species::Photon {
        input.read("beam:bremsstrahlung_source").unwrap_or(false)
    } else {
        false
    };
    let gamma_min = if use_brem_spec {
        input.read("beam:gamma_min")?
    } else {
        1.0
    };

    let offset = input.read::<Vec<f64>,_>("beam:offset")
        // if missing, assume to be (0,0,0)
        .or_else(|e| match e.kind() {
            InputErrorKind::Location => Ok(vec![0.0; 3]),
            _ => Err(e),
        })
        .and_then(|v| match v.len() {
            3 => Ok(ThreeVector::new(v[0], v[1], v[2])),
            _ => {
                eprintln!("A collision offset must be expressed as a three-vector [dx, dy, dz].");
                Err(InputError::conversion("beam:offset", "offset"))
            }
        })
        ?;

    let ident: String = input.read("output:ident")
        .map(|s| {
            if s == "auto" {
                // use the name of the input file instead
                let stem = path.file_stem().map(|os| os.to_str()).flatten();
                if let Some(stem) = stem {
                    stem.to_owned()
                } else {
                    eprintln!("Unexpected failure to extract stem of input file ('{}'), as required for 'ident: auto'.", path.display());
                    "".to_owned()
                }
            } else {
                s
            }
        })
        .unwrap_or_else(|_| "".to_owned());

    let plain_text_output = match input.read::<String,_>("output:dump_all_particles") {
        Ok(s) if s == "plain_text" || s == "plain-text" => OutputMode::PlainText,
        #[cfg(feature = "hdf5-output")]
        Ok(s) if s == "hdf5" => OutputMode::Hdf5,
        _ => OutputMode::None,
    };

    let laser_defines_z = match input.read::<String,_>("output:coordinate_system") {
        Ok(s) if s == "beam" => false,
        _ => true,
    };

    let discard_bg_e = input.read("output:discard_background_e").unwrap_or(false);

    let min_energy: f64 = input
        .read("output:min_energy")
        .map(|e: f64| 1.0e-6 * e / -ELECTRON_CHARGE) // convert from J to MeV
        .unwrap_or(0.0);

    let eospec: Vec<String> = input.read("output:electron")
        .or_else(|e| match e.kind() {InputErrorKind::Location => Ok(vec![]), _ => Err(e)})?;
    let eospec: Vec<DistributionFunction> = eospec
        .iter()
        .map(|s| s.parse())
        .collect::<Result<Vec<_>,_>>()?;
    
    let gospec: Vec<String> = input.read("output:photon")
        .or_else(|e| match e.kind() {InputErrorKind::Location => Ok(vec![]), _ => Err(e)})?;
    let gospec: Vec<DistributionFunction> = gospec
        .iter()
        .map(|s| s.parse())
        .collect::<Result<Vec<_>,_>>()?;

    let pospec: Vec<String> = input.read("output:positron")
        .or_else(|e| match e.kind() {InputErrorKind::Location => Ok(vec![]), _ => Err(e)})?;
    let pospec: Vec<DistributionFunction> = pospec
        .iter()
        .map(|s| s.parse())
        .collect::<Result<Vec<_>,_>>()?;

    // Choose the system of units
    let units = input.read::<String,_>("output:units")
        // if not specified, default to "auto"
        .or_else(|e| match e.kind() {
            InputErrorKind::Location => Ok("auto".to_owned()),
            _ => Err(e),
        })
        .and_then(|s| match s.as_str() {
            "auto" => Ok(Default::default()),
            "hep" | "HEP" => Ok(UnitSystem::hep()),
            "si" | "SI" => Ok(UnitSystem::si()),
            _ => {
                eprintln!("Unit system requested, \"{}\", is not one of \"auto\", \"hep\", or \"si\".", s);
                Err(InputError::conversion("output:units", "units"))
            }
        })
        ?;

    let mut estats = input.read("stats:electron")
        .map_or_else(|_| Ok(vec![]), |strs: Vec<String>| {
            strs.iter()
                .map(|spec| SummaryStatistic::load(spec, |s| input.evaluate(s)))
                .collect::<Result<Vec<_>,_>>()
        })?;

    let mut gstats = input.read("stats:photon")
        .map_or_else(|_| Ok(vec![]), |strs: Vec<String>| {
            strs.iter()
                .map(|spec| SummaryStatistic::load(spec, |s| input.evaluate(s)))
                .collect::<Result<Vec<_>,_>>()
        })?;

    let mut pstats = input.read("stats:positron")
        .map_or_else(|_| Ok(vec![]), |strs: Vec<String>| {
            strs.iter()
                .map(|spec| SummaryStatistic::load(spec, |s| input.evaluate(s)))
                .collect::<Result<Vec<_>,_>>()
        })?;

    // Rare event sampling for pair creation
    let pair_rate_increase = input.read::<f64,_>("control:increase_pair_rate_by")
        // if increase is not specified at all, default to unity
        .or_else(|e| match e.kind() {
            InputErrorKind::Location => Ok(1.0),
            _ => Err(e),
        })
        // failing that, check for automatic increase
        .or_else(|e| match input.read::<String,_>("control:increase_pair_rate_by") {
            Ok(s) if s == "auto" => if using_lcfa {
                Ok(increase_lcfa_pair_rate_by(gamma, a0, wavelength))
            } else {
                Ok(increase_pair_rate_by(gamma, a0, wavelength))
            },
            _ => Err(e),
        })
        .and_then(|r| if r < 1.0 {
            eprintln!("Increase in pair creation rate must be >= 1.0.");
            Err(InputError::conversion("control:increase_pair_rate_by", "increase_pair_rate_by"))
        } else {
            Ok(r)
        })?;

    let seed = 0x8658b90036b165ebu64 + ((rng_seed as u64) * 0x32f55cddaebae910u64);
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    for _i in 0..id {
        rng.jump();
    }

    let nums: Vec<usize> = {
        let tasks = ntasks as usize;
        (0..tasks).map(|i| (npart * (i + 1) / tasks) - (npart * i / tasks)).collect()
    };
    assert_eq!(nums.iter().sum::<usize>(), npart);
    let num = nums[id as usize];

    if id == 0 {
        println!("Running {} task{} with {} primary particles per task...", ntasks, if ntasks > 1 {"s"} else {""}, num);
        #[cfg(feature = "with-mpi")] {
            println!("\t* with MPI support enabled");
        }
        #[cfg(feature = "fits-output")] {
            println!("\t* writing FITS output");
        }
        #[cfg(feature = "hdf5-output")] {
            println!("\t* writing HDF5 output");
        }
        #[cfg(feature = "cos2-envelope-in-3d")] {
            if focusing {
                println!("\t* with cos^2 temporal envelope");
            }
        }
        if pair_rate_increase > 1.0 {
            println!("\t* with pair creation rate increased by {:.3e}", pair_rate_increase);
        }
    }

    let initial_z = if focusing {
        if cfg!(feature = "cos2-envelope-in-3d") {
            wavelength * tau + 3.0 * length
        } else {
            2.0 * SPEED_OF_LIGHT * tau + 3.0 * length
        }
    } else {
        0.5 * wavelength * tau
    };

    let builder = BeamBuilder::new(species, num, initial_z)
        .with_weight(weight)
        .with_divergence(rms_div)
        .with_collision_angle(angle)
        .with_offset(offset)
        .with_length(length);

    let builder = if normally_distributed {
        builder.with_normally_distributed_xy(radius, radius)
    } else {
        builder.with_uniformly_distributed_xy(radius)
    };

    let builder = if use_brem_spec {
        builder.with_bremsstrahlung_spectrum(gamma_min, gamma)
    } else {
        builder.with_normal_energy_spectrum(gamma, sigma)
    };

    let primaries = builder.build(&mut rng);

    let mut current_id = num as u64;

    let merge = |(mut e, mut g, mut p): (Vec<Particle>, Vec<Particle>, Vec<Particle>), mut sh: Shower| {
        sh.secondaries.retain(|&pt| pt.momentum()[0] > min_energy);
        if multiplicity.is_none() || (multiplicity.is_some() && multiplicity.unwrap() == sh.multiplicity()) {
            while let Some(pt) = sh.secondaries.pop() {
                match pt.species() {
                    Species::Electron => e.push(pt),
                    Species::Photon => g.push(pt),
                    Species::Positron => p.push(pt),
                }
            }
        }
        (e, g, p)
    };

    let runtime = std::time::Instant::now();

    let (mut electrons, mut photons, mut positrons) = if focusing && !using_lcfa {
        let laser = FocusedLaser::new(a0, wavelength, waist, tau, pol);
        let laser = if finite_bandwidth {laser.with_finite_bandwidth()} else {laser};
        //println!("total energy = {}", laser.total_energy());
        primaries
            .chunks(num / 20)
            .enumerate()
            .map(|(i, chk)| {
                let tmp = chk.iter()
                    .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier, &mut current_id, pair_rate_increase, discard_bg_e, rr, tracking_photons))
                    .fold((Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()), merge);
                if id == 0 {
                    println!("Done {: >12} of {: >12} primaries, RT = {}, ETTC = {}...",
                    (i+1) * chk.len(), num,
                    PrettyDuration::from(runtime.elapsed()),
                    PrettyDuration::from(ettc(runtime, i+1, 20)));
                }
                tmp
            })
            .fold(
                (Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()),
                |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat(), [a.2,b.2].concat())
            )
    } else if focusing { // and using LCFA rates
        let laser = FastFocusedLaser::new(a0, wavelength, waist, tau, pol);
        primaries
            .chunks(num / 20)
            .enumerate()
            .map(|(i, chk)| {
                let tmp = chk.iter()
                    .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier, &mut current_id, pair_rate_increase, discard_bg_e, rr, tracking_photons))
                    .fold((Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()), merge);
                if id == 0 {
                    println!("Done {: >12} of {: >12} primaries, RT = {}, ETTC = {}...",
                    (i+1) * chk.len(), num,
                    PrettyDuration::from(runtime.elapsed()),
                    PrettyDuration::from(ettc(runtime, i+1, 20)));
                }
                tmp
            })
            .fold(
                (Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()),
                |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat(), [a.2,b.2].concat())
            )
    } else if !using_lcfa {
        let laser = PlaneWave::new(a0, wavelength, tau, pol, chirp_b);
        let laser = if finite_bandwidth {laser.with_finite_bandwidth()} else {laser};
        primaries
        .chunks(num / 20)
        .enumerate()
        .map(|(i, chk)| {
            let tmp = chk.iter()
                .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier, &mut current_id, pair_rate_increase, discard_bg_e, rr, tracking_photons))
                .fold((Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()), merge);
            if id == 0 {
                println!("Done {: >12} of {: >12} primaries, RT = {}, ETTC = {}...",
                (i+1) * chk.len(), num,
                PrettyDuration::from(runtime.elapsed()),
                PrettyDuration::from(ettc(runtime, i+1, 20)));
            }
            tmp
        })
        .fold(
            (Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()),
            |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat(), [a.2,b.2].concat())
        )
    } else { // plane wave and lcfa
        let laser = FastPlaneWave::new(a0, wavelength, tau, pol, chirp_b);
        primaries
        .chunks(num / 20)
        .enumerate()
        .map(|(i, chk)| {
            let tmp = chk.iter()
                .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier, &mut current_id, pair_rate_increase, discard_bg_e, rr, tracking_photons))
                .fold((Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()), merge);
            if id == 0 {
                println!("Done {: >12} of {: >12} primaries, RT = {}, ETTC = {}...",
                (i+1) * chk.len(), num,
                PrettyDuration::from(runtime.elapsed()),
                PrettyDuration::from(ettc(runtime, i+1, 20)));
            }
            tmp
        })
        .fold(
            (Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()),
            |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat(), [a.2,b.2].concat())
        )
    };

    // Particle/parent ids are only unique within a single parallel process
    let mut id_offsets = vec![0u64; world.size() as usize];
    #[cfg(feature = "with-mpi")]
    world.all_gather_into(&current_id, &mut id_offsets[..]);
    id_offsets.iter_mut().fold(0, |mut total, n| {total += *n; *n = total - *n; total});
    // task n adds id_offsets[n] to each particle/parent id
    for pt in electrons.iter_mut().chain(photons.iter_mut()).chain(positrons.iter_mut()) {
        pt.with_id(pt.id() + id_offsets[id as usize]);
        pt.with_parent_id(pt.parent_id() + id_offsets[id as usize]);
    }

    if !laser_defines_z {
        for pt in electrons.iter_mut().chain(photons.iter_mut()).chain(positrons.iter_mut()) {
            *pt = pt.to_beam_coordinate_basis(angle);
        }
    }

    for dstr in &eospec {
        let prefix = format!("{}{}{}{}electron", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
        dstr.write(&world, &electrons, &units, &prefix)?;
    }

    for dstr in &gospec {
        let prefix = format!("{}{}{}{}photon", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
        dstr.write(&world, &photons, &units, &prefix)?;
    }

    for dstr in &pospec {
        let prefix = format!("{}{}{}{}positron", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
        dstr.write(&world, &positrons, &units, &prefix)?;
    }

    for stat in estats.iter_mut() {
        stat.evaluate(&world, &electrons, "electron");
    }

    for stat in gstats.iter_mut() {
        stat.evaluate(&world, &photons, "photon");
    }

    for stat in pstats.iter_mut() {
        stat.evaluate(&world, &positrons, "positron");
    }

    if id == 0 {
        if !estats.is_empty() || !gstats.is_empty() || !pstats.is_empty() {
            use std::fs::File;
            use std::io::Write;
            let filename = format!("{}{}{}{}stats.txt", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
            let mut file = File::create(filename)?;
            for stat in &estats {
                writeln!(file, "{}", stat)?;
            }
            for stat in &pstats {
                writeln!(file, "{}", stat)?;
            }
            for stat in &gstats {
                writeln!(file, "{}", stat)?;
            }
        }
    }

    match plain_text_output {
        OutputMode::PlainText => {
            #[cfg(feature = "with-mpi")]
            let mut particles = [electrons, photons, positrons].concat();
            #[cfg(not(feature = "with-mpi"))]
            let particles = [electrons, photons, positrons].concat();

            if id == 0 {
                use std::fs::File;
                use std::io::Write;
                let filename = format!("{}{}{}{}particles.out", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
                let mut file = File::create(filename)?;
                writeln!(file, "#{:-^1$}", "", 170)?;
                writeln!(file, "# Particle properties when tracking stops")?;
                writeln!(file, "#{:-^1$}", "", 170)?;
                writeln!(file, "# First interacting species: {}\t\tSecond interacting species: laser", species.to_string())?;
                writeln!(file, "# First initial particle energy = {:.4} +/- {:.4} GeV, Sigma_xyz = {:.2} {:.2} {:.2} microns, count = {}", 1.0e-3 * gamma * ELECTRON_MASS_MEV, 1.0e-3 * sigma * ELECTRON_MASS_MEV, 1.0e6 * radius, 1.0e6 * radius, 1.0e6 * length, (npart as f64) * weight)?;
                writeln!(file, "# Laser peak intensity = {:.2} x 10^18 W/cm^2, wavelength = {:.2} nm, pulse length = {:.2} fs, beam waist = {:.2} microns", a0.powi(2) * 1.37 / (1.0e6 * wavelength).powi(2), 1.0e9 * wavelength, 1.0e15 * tau, 1.0e6 * waist)?;
                writeln!(file, "# Pulse peak xi = {:.4}, chi = {:.4}", a0, (2.0 * consts::PI * SPEED_OF_LIGHT * COMPTON_TIME / wavelength) * a0 * gamma * (1.0 + angle.cos()))?;
                writeln!(file, "#{:-^1$}", "", 170)?;
                writeln!(file, "# E (GeV)\tx (micron)\ty (micron)\tz (micron)\tp_x (GeV/c)\tp_y (GeV/c)\tp_z (GeV/c)\tPDG_NUM\tMP_Wgt\tMP_ID\tt (um/c)\txi")?;
                //writeln!(file, "# E (GeV)\tx (micron)\ty (micron)\tz (micron)\tbeta_x\tbeta_y\tbeta_z\tPDG_NUM\tMP_Wgt\tMP_ID\tt (um/c)\txi")?;
                writeln!(file, "#{:-^1$}", "", 170)?;

                for pt in &particles {
                    writeln!(file, "{}", pt)?;
                }

                #[cfg(feature = "with-mpi")]
                for recv_rank in 1..ntasks {
                    particles = world.process_at_rank(recv_rank).receive_vec::<Particle>().0;
                    for pt in &particles {
                        writeln!(file, "{}", pt)?;
                    }
                }
            }

            #[cfg(feature = "with-mpi")]
            if id != 0 {
                world.process_at_rank(0).synchronous_send(&particles[..]);
            }
        },
        #[cfg(feature = "hdf5-output")]
        OutputMode::Hdf5 => {
            if id == 0 {
                let filename = format!("{}{}{}{}particles.h5", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
                let file = hdf5::File::create(&filename)?;

                // Build info
                file.create_group("build")?
                    .new_data("version").write(env!("CARGO_PKG_VERSION"))?
                    .new_data("branch").write(env!("VERGEN_GIT_BRANCH"))?
                    .new_data("commit-hash").write(env!("VERGEN_GIT_SHA"))?
                    .new_data("features").write(env!("PTARMIGAN_ACTIVE_FEATURES"))?;

                // Top-level run information
                let conf = file.create_group("config")?;
                conf.new_data("mpi-tasks").write(&ntasks)?
                    .new_data("input-file").write(raw_input.as_str())?;

                conf.create_group("unit")?
                    .new_data("position").write(units.length.name())?
                    .new_data("momentum").write(units.momentum.name())?;

                // Parsed input configuration
                conf.create_group("control")?
                    .new_data("dt_multiplier").write(&dt_multiplier)?
                    .new_data("radiation_reaction").write(&rr)?
                    .new_data("pair_creation").write(&tracking_photons)?
                    .new_data("lcfa").write(&using_lcfa)?
                    .new_data("rng_seed").write(&rng_seed)?
                    .new_data("increase_pair_rate_by").write(&pair_rate_increase)?
                    .new_data("bandwidth_correction").write(&finite_bandwidth)?
                    .new_data("select_multiplicity").with_condition(|| multiplicity.is_some()).write(&multiplicity.unwrap_or(0))?
                    .new_data("select_multiplicity").with_condition(|| multiplicity.is_none()).write(&false)?;

                conf.create_group("laser")?
                    .new_data("a0")
                        .with_unit("1")
                        .with_desc("peak value of the laser normalized amplitude")
                        .write(&a0)?
                    .new_data("wavelength")
                        .with_unit(units.length.name())
                        .with_desc("wavelength of the carrier")
                        .write(&wavelength.convert(&units.length))?
                    .new_data("polarization")
                        .with_desc("linear/circular")
                        .write(&pol)?
                    .new_data("focusing")
                        .with_desc("true/false => pulse is modelled in 1d/3d")
                        .write(&focusing)?
                    .new_data("chirp_b")
                        .with_unit("1")
                        .with_desc("parameter that appears in carrier phase = phi + b phi^2")
                        .write(&chirp_b)?
                    .new_data("waist")
                        .with_unit(units.length.name())
                        .with_desc("radius in the focal plane at which intensity is 1/e^2 of its peak value")
                        .with_condition(|| focusing)
                        .write(&waist.convert(&units.length))?
                    .new_data("fwhm_duration")
                        .with_unit("s")
                        .with_desc("full width at half maximum of the temporal intensity profile")
                        .with_condition(|| focusing && !cfg!(feature = "cos2-envelope-in-3d"))
                        .write(&tau)?
                    .new_data("n_cycles")
                        .with_unit("1")
                        .with_desc("number of wavelengths corresponding to the total pulse duration")
                        .with_condition(|| !focusing || cfg!(feature = "cos2-envelope-in-3d"))
                        .write(&tau)?;

                let charge = match species {
                    Species::Electron => (npart as f64) * weight * ELECTRON_CHARGE,
                    Species::Positron => (npart as f64) * weight * -ELECTRON_CHARGE,
                    Species::Photon => 0.0,
                };

                conf.create_group("beam")?
                    .new_data("n")
                        .with_unit("1")
                        .with_desc("number of primary macroparticles")
                        .write(&npart)?
                    .new_data("n_real")
                        .with_unit("1")
                        .with_desc("total number of real particles represented by the primary macroparticles")
                        .write(&((npart as f64) * weight))?
                    .new_data("charge").with_unit("C").write(&charge)?
                    .new_data("species").write(species.to_string().as_str())?
                    .new_data("gamma").with_unit("1").write(&gamma)?
                    .new_data("sigma").with_unit("1").write(&sigma)?
                    .new_data("bremsstrahlung_source").write(&use_brem_spec)?
                    .new_data("gamma_min").with_unit("1").with_condition(|| use_brem_spec).write(&gamma_min)?
                    .new_data("radius").with_unit(units.length.name()).write(&radius.convert(&units.length))?
                    .new_data("length").with_unit(units.length.name()).write(&length.convert(&units.length))?
                    .new_data("collision_angle").with_unit("rad").write(&angle)?
                    .new_data("rms_divergence").with_unit("rad").write(&rms_div)?
                    .new_data("offset").with_unit(units.length.name()).write(&offset.convert(&units.length))?
                    .new_data("transverse_distribution_is_normal").write(&normally_distributed)?
                    .new_data("longitudinal_distribution_is_normal").write(&true)?;

                conf.create_group("output")?
                    .new_data("laser_defines_positive_z").write(&laser_defines_z)?
                    .new_data("beam_defines_positive_z").write(&!laser_defines_z)?
                    .new_data("discard_background_e").write(&discard_bg_e)?
                    .new_data("min_energy").with_unit(units.energy.name()).write(&min_energy.convert(&units.energy))?;

                // Write particle data
                let fs = file.create_group("final-state")?;

                let mut photons = photons;
                #[cfg(feature = "with-mpi")]
                for recv_rank in 1..ntasks {
                    let mut recv = world.process_at_rank(recv_rank).receive_vec::<Particle>().0;
                    photons.append(&mut recv);
                }
                let (x, p, w, a, n, id, pid) = photons
                    .iter()
                    .map(|pt| (
                        pt.position().convert(&units.length),
                        pt.momentum().convert(&units.momentum),
                        pt.weight(),
                        pt.payload(),
                        pt.interaction_count(),
                        pt.id(),
                        pt.parent_id()
                    ))
                    .unzip_n_vec();
                drop(photons);

                fs.create_group("photon")?
                    .new_data("weight")
                        .with_unit("1")
                        .with_desc("number of real photons each macrophoton represents")
                        .write(&w[..])?
                    .new_data("a0_at_creation")
                        .with_unit("1")
                        .with_desc("normalized amplitude at point of emission")
                        .write(&a[..])?
                    .new_data("n_pos")
                        .with_unit("1")
                        .with_desc("total probability of pair creation for the photon")
                        .write(&n[..])?
                    .new_data("id")
                        .with_desc("unique ID of the photon")
                        .write(&id[..])?
                    .new_data("parent_id")
                        .with_desc("ID of the particle that created the photon (for primary particles, parent_id = id")
                        .write(&pid[..])?
                    .new_data("position")
                        .with_unit(units.length.name())
                        .with_desc("four-position of the photon")
                        .write(&x[..])?
                    .new_data("momentum")
                        .with_unit(units.momentum.name())
                        .with_desc("four-momentum of the photon")
                        .write(&p[..])?;

                // Provide alias for a0
                fs.group("photon")?.link_soft("a0_at_creation", "xi")?;

                let mut electrons = electrons;
                #[cfg(feature = "with-mpi")]
                for recv_rank in 1..ntasks {
                    let mut recv = world.process_at_rank(recv_rank).receive_vec::<Particle>().0;
                    electrons.append(&mut recv);
                }
                let (x, p, w, n, id, pid) = electrons
                    .iter()
                    .map(|pt| (
                        pt.position().convert(&units.length),
                        pt.momentum().convert(&units.momentum),
                        pt.weight(),
                        pt.interaction_count(),
                        pt.id(),
                        pt.parent_id()
                    ))
                    .unzip_n_vec();
                drop(electrons);

                fs.create_group("electron")?
                    .new_data("weight")
                        .with_unit("1")
                        .with_desc("number of real electrons each macroelectron represents")
                        .write(&w[..])?
                    .new_data("n_gamma")
                        .with_unit("1")
                        .with_desc("total number of photons emitted by the electron")
                        .write(&n[..])?
                    .new_data("id")
                        .with_desc("unique ID of the electron")
                        .write(&id[..])?
                    .new_data("parent_id")
                        .with_desc("ID of the particle that created the electron (for primary particles, parent_id = id)")
                        .write(&pid[..])?
                    .new_data("position")
                        .with_unit(units.length.name())
                        .with_desc("four-position of the electron")
                        .write(&x[..])?
                    .new_data("momentum")
                        .with_unit(units.momentum.name())
                        .with_desc("four-momentum of the electron")
                        .write(&p[..])?;

                let mut positrons = positrons;
                #[cfg(feature = "with-mpi")]
                for recv_rank in 1..ntasks {
                    let mut recv = world.process_at_rank(recv_rank).receive_vec::<Particle>().0;
                    positrons.append(&mut recv);
                }
                let (x, x0, p, w, n, id, pid, a) = positrons
                    .iter()
                    .map(|pt| (
                        pt.position().convert(&units.length),
                        pt.was_created_at().convert(&units.length),
                        pt.momentum().convert(&units.momentum),
                        pt.weight(),
                        pt.interaction_count(),
                        pt.id(),
                        pt.parent_id(),
                        pt.payload()
                    ))
                    .unzip_n_vec();
                drop(positrons);

                fs.create_group("positron")?
                    .new_data("weight")
                        .with_unit("1")
                        .with_desc("number of real positrons each macropositron represents")
                        .write(&w[..])?
                    .new_data("a0_at_creation")
                        .with_unit("1")
                        .with_desc("normalized amplitude at point of creation")
                        .write(&a[..])?
                    .new_data("n_gamma")
                        .with_unit("1")
                        .with_desc("total number of photons emitted by the positron")
                        .write(&n[..])?
                    .new_data("id")
                        .with_desc("unique ID of the positron")
                        .write(&id[..])?
                    .new_data("parent_id")
                        .with_desc("ID of the particle that created the positron (for primary particles, parent_id = id)")
                        .write(&pid[..])?
                    .new_data("position")
                        .with_unit(units.length.name())
                        .with_desc("four-position of the positron")
                        .write(&x[..])?
                    .new_data("position_at_creation")
                        .with_unit(units.length.name())
                        .with_desc("four-position at which the positron was created")
                        .write(&x0[..])?
                    .new_data("momentum")
                        .with_unit(units.momentum.name())
                        .with_desc("four-momentum of the positron")
                        .write(&p[..])?;

                fs.group("positron")?.link_soft("a0_at_creation", "xi")?;
            } else {
                #[cfg(feature = "with-mpi")] {
                    world.process_at_rank(0).synchronous_send(&photons[..]);
                    drop(photons);
                    world.process_at_rank(0).synchronous_send(&electrons[..]);
                    drop(electrons);
                    world.process_at_rank(0).synchronous_send(&positrons[..]);
                    drop(positrons);
                }
            }
        },
        OutputMode::None => {},
    }

    if id == 0 {
        println!("Run complete after {}.", PrettyDuration::from(runtime.elapsed()));
    }

    Ok(())
}
