use std::error::Error;
use std::path::{Path, PathBuf};
use std::f64::consts;
use std::process::ExitCode;

use colored::Colorize;

#[cfg(feature = "with-mpi")]
use mpi::traits::*;

#[cfg(feature = "with-mpi")]
use mpi::collective::SystemOperation;

#[cfg(not(feature = "with-mpi"))]
extern crate no_mpi as mpi;

#[cfg(not(feature = "with-mpi"))]
use mpi::{Communicator, SystemOperation};

use rand::prelude::*;
use rand_xoshiro::*;

#[cfg(feature = "hdf5-output")]
use hdf5_writer;
#[cfg(feature = "hdf5-output")]
unzip_n::unzip_n!(pub 7);
#[cfg(feature = "hdf5-output")]
unzip_n::unzip_n!(pub 9);
#[cfg(feature = "hdf5-output")]
unzip_n::unzip_n!(pub 10);

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
mod pwmci;
mod quadrature;

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
    #[cfg(feature = "hdf5-output")]
    Hdf5,
}

enum StopAt {
    /// Continue until all particles have the same time coordinate.
    SameTime,
    /// Continue until all particles have the specified time coordinate.
    GivenTime(f64),
    /// Continue until all particles have fallen out of the high-field region.
    Automatic,
}

/// Wrapper around detailed optional arguments to [collide](collide)
#[derive(Copy, Clone)]
struct CollideOptions {
    /// Scale the automatic timestep by a factor (generally < 1)
    dt_multiplier: f64,
    /// Increase the pair creation rate by a factor (> 1)
    rate_increase: f64,
    /// Halt particle tracking (if not already stopped!) at a given time
    t_stop: f64,
    /// Discard electrons that have not radiated from output
    discard_bg_e: bool,
    /// Discard photons that have not pair-created from output
    discard_bg_ph: bool,
    /// Enable/disable recoil on photon emission
    rr: bool,
    /// Track/do not track photons through the EM field.
    tracking_photons: bool,
    /// Preserve data about decayed photons for later output
    keep_decayed_photons: bool,
    /// Use polarization-resolved pair creation rates
    pol_resolved: bool,
    /// Rotate Stokes vector in absence of pair creation
    rotate_stokes_pars: bool,
    /// Use classical emission rates
    classical: bool,
    /// Correct classical spectrum using Gaunt factor
    gaunt_factor: bool,
}

/// Type of diagnostic message that can be issued
pub enum Diagnostic {
    Warning,
    Error,
}

/// Wrapper around eprintln! that suppresses output from all but rank 0,
/// prefixing either "Error:" or "Warning:" as appropriate.
macro_rules! report {
    ($report:expr, $allowed:expr, $($args: tt)*) => {
        if $allowed {
            use colored::Colorize;
            use crate::Diagnostic;
            let class = match $report {
                Diagnostic::Warning => "Warning".bold().bright_yellow(),
                Diagnostic::Error => "Error".bold().red(),
            };
            eprintln!("{}: {}", class, format_args!($($args)*))
        };
    }
}

pub(crate) use report;

/// Propagates a single particle through a region of EM field, returning a Shower containing
/// the primary and any secondary particles generated.
/// `current_id` is incremented every time a new particle is generated.
fn collide<F: Field, R: Rng>(field: &F, incident: Particle, rng: &mut R, current_id: &mut u64, options: CollideOptions) -> Shower {
    let mut primaries = vec![incident];
    let mut secondaries: Vec<Particle> = Vec::new();
    let mut intermediates: Vec<Particle> = Vec::new();

    let dt = field.max_timestep().unwrap_or(1.0);
    let dt = dt * options.dt_multiplier;
    let primary_id = incident.id();

    let eqn = if options.classical && options.rr {
        if options.gaunt_factor {
            EquationOfMotion::ModifiedLandauLifshitz
        } else {
            EquationOfMotion::LandauLifshitz
        }
    } else {
        EquationOfMotion::Lorentz
    };

    let mode = if options.classical && !options.gaunt_factor {
        RadiationMode::Classical
    } else {
        RadiationMode::Quantum
    };

    let electron_recoils = !options.classical && options.rr;

    while let Some(mut pt) = primaries.pop() {
        match pt.species() {
            Species::Electron | Species::Positron => {
                while field.contains(pt.position()) && pt.time() < options.t_stop {
                    let (r, mut u, dt_actual, work_done) = field.push(
                        pt.position(),
                        pt.normalized_momentum(),
                        pt.charge_to_mass_ratio(),
                        dt,
                        eqn,
                    );

                    if let Some(event) = field.radiate(r, u, dt_actual, rng, mode) {
                        let id = *current_id;
                        *current_id = *current_id + 1;
                        let photon = Particle::create(Species::Photon, r)
                            .with_payload(event.a_eff)
                            .with_parent_chi(event.chi)
                            .with_weight(pt.weight())
                            .with_id(id)
                            .with_parent_id(pt.id())
                            .with_polarization(event.pol)
                            .with_normalized_momentum(event.k);
                        primaries.push(photon);

                        if electron_recoils {
                            u = event.u_prime;
                        }

                        pt.update_interaction_count(1.0);

                        if !options.classical {
                            pt.update_absorbed_energy(event.absorption);
                        }
                    }

                    pt.with_position(r);
                    pt.with_normalized_momentum(u);
                    pt.update_absorbed_energy(work_done);
                }

                if pt.id() != primary_id || !options.discard_bg_e || pt.interaction_count() > 0.0 {
                    secondaries.push(pt);
                }
            },

            Species::Photon => {
                let mut has_decayed = false;
                while field.contains(pt.position()) && pt.time() < options.t_stop && !has_decayed && options.tracking_photons {
                    let ell = pt.normalized_momentum();
                    let r: FourVector = pt.position() + SPEED_OF_LIGHT * ell * dt / ell[0];
                    let pol = if options.pol_resolved { pt.polarization() } else { StokesVector::unpolarized() };

                    let (prob, pol_new, event) = field.pair_create(r, ell, pol, dt, rng, options.rate_increase);

                    if let Some(event) = event {
                        let id = *current_id;
                        *current_id = *current_id + 2;
                        let electron = Particle::create(Species::Electron, r)
                            .with_weight(event.frac * pt.weight())
                            .with_id(id)
                            .with_payload(event.a_eff)
                            .with_parent_chi(event.chi)
                            .with_parent_id(pt.id())
                            .update_absorbed_energy(0.5 * event.absorption)
                            .with_normalized_momentum(event.u_e);
                        let positron = Particle::create(Species::Positron, r)
                            .with_weight(event.frac * pt.weight())
                            .with_id(id + 1)
                            .with_payload(event.a_eff)
                            .with_parent_chi(event.chi)
                            .with_parent_id(pt.id())
                            .update_absorbed_energy(0.5 * event.absorption)
                            .with_normalized_momentum(event.u_p);
                        primaries.push(electron);
                        primaries.push(positron);
                        pt.with_weight(pt.weight() * (1.0 - event.frac));
                        if pt.weight() <= 0.0 {
                            has_decayed = true;
                        }
                    }

                    if options.rotate_stokes_pars && options.pol_resolved {
                        pt.with_polarization(pol_new);
                    }

                    pt.update_interaction_count(prob);
                    pt.with_position(r);
                }

                if !has_decayed && (pt.id() != primary_id || !options.discard_bg_ph) {
                    secondaries.push(pt);
                }

                if has_decayed && options.keep_decayed_photons {
                    intermediates.push(pt);
                }
            }
        }
    }

    Shower {
        primary: incident,
        secondaries,
        intermediates,
    }
}

/// Returns the ratio of the pair creation and photon emission rates,
/// for a photon (or electron) with normalized energy `gamma` in a
/// laser with amplitude `a0` and wavelength `wavelength`.
fn increase_pair_rate_by(gamma: f64, a0: f64, wavelength: f64, pol: Polarization) -> f64 {
    let kappa: FourVector = SPEED_OF_LIGHT * COMPTON_TIME * 2.0 * consts::PI * FourVector::new(1.0, 0.0, 0.0, 1.0) / wavelength;
    let ell: FourVector = FourVector::lightlike(0.0, 0.0, -gamma);
    let u: FourVector = FourVector::new(0.0, 0.0, 0.0, -gamma).unitize();
    let a_rms = match pol { Polarization::Linear => a0 / consts::SQRT_2, Polarization::Circular => a0 };
    let q: FourVector = u + a_rms * a_rms * kappa / (2.0 * kappa * u);
    let dt = wavelength / SPEED_OF_LIGHT;
    let (pair_rate, _) = pair_creation::probability(ell, StokesVector::unpolarized(), kappa, a_rms, dt, pol, 0.0);
    let photon_rate = nonlinear_compton::probability(kappa, q, dt, pol, RadiationMode::Quantum);
    if pair_rate == 0.0 || photon_rate.is_none() {
        1.0
    } else {
        let ratio = photon_rate.unwrap() / pair_rate;
        //println!("P_pair = {:.6e}, P_photon = {:.6e}, ratio = {:.3}", pair_rate.unwrap(), photon_rate.unwrap(), ratio);
        ratio.max(1.0)
    }
}

fn increase_lcfa_pair_rate_by(gamma: f64, a0: f64, wavelength: f64) -> f64 {
    let omega_mc2 = 1.26e-6 / (ELECTRON_MASS_MEV * 1.0e6 * wavelength);
    let chi = 2.0 * gamma * a0 * omega_mc2;
    let ell: FourVector = FourVector::lightlike(0.0, 0.0, -gamma);
    let pair_rate = lcfa::pair_creation::probability(ell, StokesVector::unpolarized(), chi, [1.0, 0.0, 0.0].into(), 1.0).0;
    let photon_rate = lcfa::photon_emission::rate(chi, gamma);
    photon_rate / pair_rate
}

fn ptarmigan_main<C: Communicator>(world: C) -> Result<(), Box<dyn Error>> {
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
    input.with_context("constants")?;

    let dt_multiplier = input.read("control:dt_multiplier").unwrap_or(1.0);
    let multiplicity: Option<usize> = input.read("control:select_multiplicity").ok();
    let using_lcfa = input.read("control:lcfa").unwrap_or(false);
    let rng_seed = input.read("control:rng_seed").unwrap_or(0usize);
    let finite_bandwidth = input.read("control:bandwidth_correction").unwrap_or(false);
    let rr = input.read("control:radiation_reaction").unwrap_or(true);

    let t_stop = input.read::<String, _>("control:stop_at_time")
        .and_then(|s| match s.as_str() {
            "auto" => Ok(StopAt::SameTime),
            _ => input.evaluate(s)
                    .map(|t| StopAt::GivenTime(t))
                    .ok_or_else(|| {
                        report!(Diagnostic::Error, id == 0, "control:stop_at_time must be a numerical value or 'auto'.");
                        InputError::conversion("control:stop_at_time", "stop_at_time")
                    })
        })
        .or_else(|e| match e.kind() {
            InputErrorKind::Conversion => Err(e), // preserve error if parsing failed
            _ => Ok(StopAt::Automatic),
        })
        ?;

    let (classical, gaunt_factor) = input.read::<String, _>("control:classical")
        .and_then(|s| match s.as_str() {
            "true" => Ok((true, false)),
            "false" => Ok((false, false)),
            "gaunt_factor_corrected" => Ok((true, true)),
            _ => {
                report!(Diagnostic::Error, id == 0, "control:classical must be one of 'true', 'false' or 'gaunt_factor_corrected'.");
                Err(InputError::conversion("control:classical", "classical"))
            }
        })
        // Gaunt factor correction is available only under LCFA
        .and_then(|(classical, gaunt_factor)| {
            if gaunt_factor && !using_lcfa {
                report!(Diagnostic::Error, id == 0, "Gaunt factor correction is only available under the LCFA.");
                Err(InputError::conversion("control:classical", "classical"))
            } else {
                Ok((classical, gaunt_factor))
            }
        })
        .or_else(|e| match e.kind() {
            // preserve error if 'control:classical' is present, but parsing failed
            InputErrorKind::Conversion => Err(e),
            // if 'control:classical' is not present, default to qed rates
            _ => Ok((false, false)),
        })
        ?;

    // pair creation is enabled by default, unless classical = true
    let tracking_photons = input.read("control:pair_creation").unwrap_or(!classical);
    let pol_resolved = input.read("control:pol_resolved").unwrap_or(false);
    let rotate_stokes_pars = input.read("control:rotate_stokes_pars")
        .map_or(true, |val| {
            report!(Diagnostic::Warning, id == 0, "use of undocumented option control:rotate_stokes_pars, intended for debugging purposes only.");
            val
        });

    let a0_values: Vec<f64> = input.read_loop("laser:a0")?;
    let wavelength: f64 = input
        .read("laser:wavelength")
        .or_else(|_e|
            // attempt to read a frequency instead, e.g. 'omega: 1.55 * eV'
            input.read("laser:omega").map(|omega: f64| 2.0 * consts::PI * COMPTON_TIME * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3) / omega)
        )?;

    let (pol, pol_angle) = input.read::<String, _>("laser:polarization")
        .and_then(|s| match s.as_str() {
            "circular" => Ok((Polarization::Circular, 0.0)),
            "linear" | "linear || x" => Ok((Polarization::Linear, 0.0)),
            "linear || y" => Ok((Polarization::Linear, consts::FRAC_PI_2)),
            _ => {
                if let Some(expr) = s.strip_prefix("linear @") {
                    if let Some(pol_angle) = input.evaluate(expr) {
                        Ok((Polarization::Linear, pol_angle))
                    } else {
                        report!(Diagnostic::Error, id == 0, "linear polarization specified, but '{}' is not a valid angle.", expr.trim());
                        Err(InputError::conversion("laser:polarization", "polarization"))
                    }
                } else {
                    report!(Diagnostic::Error, id == 0, "laser polarization must be 'linear [|| x or y]' or 'circular'.");
                    Err(InputError::conversion("laser:polarization", "polarization"))
                }
            }
        })
        ?;

    let (focusing, waist) = input
        .read("laser:waist")
        .map(|w| (true, w))
        .unwrap_or((false, std::f64::INFINITY));

    let envelope = input.read::<String, _>("laser:envelope")
        .and_then(|s| match s.as_str() {
            "cos2" | "cos^2" | "cos_sqd" | "cos_squared" => Ok(Envelope::CosSquared),
            "flattop" | "flat-top" => Ok(Envelope::Flattop),
            "gauss" | "gaussian" => Ok(Envelope::Gaussian),
            _ => {
                report!(Diagnostic::Error, id == 0, "laser envelope must be one of 'cos^2', 'flattop' or 'gaussian'.");
                Err(InputError::conversion("laser:envelope", "envelope"))
            }
        })
        .unwrap_or_else(|_| if focusing {Envelope::Gaussian} else {Envelope::CosSquared});

    let n_cycles: f64 = match envelope {
        Envelope::CosSquared => input.read("laser:n_cycles")?,
        Envelope::Flattop => {
            input.read("laser:n_cycles")
                .and_then(|n: f64| if n < 1.0 {
                    report!(Diagnostic::Error, id == 0, "'n_cycles' must be >= 1.0 for flattop lasers.");
                    Err(InputError::conversion("laser:envelope", "envelope"))
                } else {
                    Ok(n)
                })?
        },
        Envelope::Gaussian => {
            input.read("laser:fwhm_duration")
                .map(|t: f64| SPEED_OF_LIGHT * t / wavelength)
                .or_else(|_e| input.read("laser:n_cycles"))?
        }
    };

    let chirp_b = if !focusing {
        input.read("laser:chirp_coeff").unwrap_or(0.0)
    } else {
        input.read("laser:chirp_coeff")
            .map(|_: f64| {
                report!(Diagnostic::Warning, id == 0, "chirp parameter ignored for focusing laser pulses.");
                0.0
            })
            .unwrap_or(0.0)
    };

    // Particle beam properties

    // Following are common to both RNG-distributed and loaded beams:
    // * species
    // * collision angles
    // * offset

    let species = input.read::<String,_>("beam:species")
        .map_or_else(
            |e| match e.kind() {
                // if the species is not specified, default to electron
                InputErrorKind::Location => Ok(Species::Electron),
                _ => Err(e)
            },
            |s| s.parse::<Species>().map_err(|_| InputError::conversion("beam:species", "species"))
        )?;

    let angle: f64 = input.read("beam:collision_angle").unwrap_or(0.0);

    let angle2: f64 = input.read::<String, _>("beam:collision_plane")
        .and_then(|s| match s.as_str() {
            "horizontal" => Ok(0.0),
            "vertical" => Ok(consts::FRAC_PI_2),
            _ => input.evaluate(s).ok_or_else(|| InputError::conversion("beam:collision_plane", "collision_plane")),
        })
        .or_else(|e| match e.kind() {
            // preserve error if parsing failed
            InputErrorKind::Conversion => Err(e),
            // if not present, default to horizontal
            _ => Ok(0.0),
        })
        ?;

    let offset = input.read::<Vec<f64>,_>("beam:offset")
        // if missing, assume to be (0,0,0)
        .or_else(|e| match e.kind() {
            InputErrorKind::Location => Ok(vec![0.0; 3]),
            _ => Err(e),
        })
        .and_then(|v| match v.len() {
            3 => Ok(ThreeVector::new(v[0], v[1], v[2])),
            _ => {
                report!(Diagnostic::Error, id == 0, "a collision offset must be expressed as a three-vector [dx, dy, dz].");
                Err(InputError::conversion("beam:offset", "offset"))
            }
        })
        ?;

    if let Ok(s) = input.read::<String, _>("beam:file") {
        report!(
            Diagnostic::Warning, id == 0,
            concat!(
                "Are you trying to import particle data from a file? If so, please ident\n",
                "           file: {}\n",
                "         so that it is part of the subsection \"{}\"."),
            s, "from_hdf5"
        );
    }

    let beam = if input.read::<String, _>("beam:from_hdf5:file").is_ok() {
        #[cfg(not(feature = "hdf5-output"))] {
            report!(Diagnostic::Error, id == 0, "cannot import particles from file (Ptarmigan not compiled with HDF5 support).");
            return Err(InputError::conversion("beam:from_hdf5:file", "file").into());
        }

        #[cfg(feature = "hdf5-output")] {
        let filename: String = input.read("beam:from_hdf5:file")?;
        let filename = format!("{}{}{}", output_dir, if output_dir.is_empty() {""} else {"/"}, filename);

        let distance: f64 = input.read("beam:from_hdf5:distance_bt_ips")?;

        let min_energy: f64 = input.read("beam:from_hdf5:min_energy")
            .map(|e: f64| 1.0e-6 * e / -ELECTRON_CHARGE)
            .unwrap_or(0.0);

        let max_angle: f64 = input.read("beam:from_hdf5:max_angle")
            .unwrap_or(consts::PI);

        let loader = BeamLoader::from_file(&filename, species.to_string().as_str(), distance)
            .with_collision_angle(angle)
            .with_offset(offset)
            .with_min_energy(min_energy)
            .with_max_angle(max_angle);

        BeamParameters::FromHdf5 { loader }
        }
    } else {
        // Number, species and weight

        let npart: usize = input.read("beam:n")
            .or_else(|_| input.read("beam:ne"))
            ?;

        let weight = input.read("beam:charge")
            .map(|q: f64| q.abs() / (constants::ELEMENTARY_CHARGE * (npart as f64)))
            .unwrap_or(1.0);

        // Energy distribution

        let gamma: f64 = input.read("beam:gamma")?;
        let sigma: f64 = input.read("beam:sigma").unwrap_or(0.0);

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

        // Spatial structure

        let length: f64 = input.read("beam:length").unwrap_or(0.0);

        let rms_div: f64 = input.read("beam:rms_divergence").unwrap_or(0.0);

        let (radius, normally_distributed, max_radius) = input.read::<Vec<String>,_>("beam:radius")
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

                // a third entry, if present, would be the optional cutoff for a
                // normal distribution
                let max_radius = vs.get(2).and_then(|s| input.evaluate(s));

                if let (Some(r), Some(b)) = (radius, normally_distributed) {
                    Ok((r, b, max_radius))
                } else {
                    report!(
                        Diagnostic::Error, id == 0, concat!(
                        "beam radius must be specified with a single numerical value, e.g.,\n",
                        "         radius: 2.0e-6\n",
                        "       or as a numerical value and a distribution, e.g.,\n",
                        "         radius: [2.0e-6, uniformly_distributed]\n",
                        "         radius: [2.0e-6, normally_distributed]."
                    ));
                    Err(InputError::conversion("beam:radius", "radius"))
                }
            })
            .or_else(|e| {
                // if radius is just missing (as opposed to malformed), return 0.0
                if e.kind() == InputErrorKind::Conversion {
                    Err(e)
                } else {
                    Ok((0.0, true, None))
                }
            })?;

        let energy_chirp = input.read::<f64, _>("beam:energy_chirp")
            .or_else(|e| match e.kind() {
                InputErrorKind::Location => Ok(0.0),
                _ => Err(e),
            })
            .and_then(|rho| {
                if use_brem_spec && rho != 0.0 {
                    report!(Diagnostic::Warning, id == 0, "energy chirp ignored for bremsstrahlung photons.");
                    Ok(0.0)
                } else if rho.abs() > 1.0 {
                    report!(Diagnostic::Error, id == 0, "absolute value of energy chirp parameter {} must be <= 1.", rho);
                    Err(InputError::conversion("beam:energy_chirp", "energy_chirp"))
                } else {
                    Ok(rho)
                }
            })
            ?;

        // Spin/polarization

        let sv = input.read::<Vec<f64>,_>("beam:stokes_pars")
            // if missing, assume to be (0,0,0)
            .or_else(|e| match e.kind() {
                InputErrorKind::Location => Ok(vec![0.0; 3]),
                _ => Err(e),
            })
            .and_then(|v| match v.len() {
                3 => {
                    let sv = StokesVector::new(1.0, v[0], v[1], v[2]);
                    if sv.dop() <= 1.0 {
                        Ok(sv)
                    } else {
                        report!(Diagnostic::Error, id == 0, "specified particle polarization does not satisfy S_1^2 + S_2^2 + S_3^2 <= 1.");
                        Err(InputError::conversion("beam:stokes_pars", "stokes_pars"))
                    }
                },
                _ => {
                    report!(Diagnostic::Error, id == 0, "particle polarization must be specified as three Stokes parameters [S_1, S_2, S_3].");
                    Err(InputError::conversion("beam:stokes_pars", "stokes_pars"))
                }
            })
            ?;

        // Number per MPI task
        let nums: Vec<usize> = {
                let tasks = ntasks as usize;
                (0..tasks).map(|i| (npart * (i + 1) / tasks) - (npart * i / tasks)).collect()
            };
        assert_eq!(nums.iter().sum::<usize>(), npart);
        let num = nums[id as usize];

        let builder = BeamBuilder::new(species, num)
            .with_weight(weight)
            .with_divergence(rms_div)
            .with_collision_angle(angle)
            .with_collision_plane_at(angle2)
            .with_offset(offset)
            .with_energy_chirp(energy_chirp)
            .with_polarization(sv)
            .with_length(length);

        let builder = if normally_distributed {
            if let Some(r_max) = max_radius {
                builder.with_trunc_normally_distributed_xy(radius, radius, r_max, r_max)
            } else {
                builder.with_normally_distributed_xy(radius, radius)
            }
        } else {
            builder.with_uniformly_distributed_xy(radius)
        };

        let builder = if use_brem_spec {
            builder.with_bremsstrahlung_spectrum(gamma_min, gamma)
        } else {
            builder.with_normal_energy_spectrum(gamma, sigma)
        };

        BeamParameters::FromRng { builder }
    };

    let ident: String = input.read("output:ident")
        .map(|s| {
            if s == "auto" {
                // use the name of the input file instead
                let stem = path.file_stem().map(|os| os.to_str()).flatten();
                if let Some(stem) = stem {
                    stem.to_owned()
                } else {
                    report!(Diagnostic::Error, id == 0, "unexpected failure to extract stem of input file ('{}'), as required for 'ident: auto'.", path.display());
                    "".to_owned()
                }
            } else {
                s
            }
        })
        .unwrap_or_else(|_| "".to_owned());

    let output_mode = input.read::<String, _>("output:dump_all_particles")
        .and_then(|s| match s.as_str() {
            #[cfg(feature = "hdf5-output")]
            "hdf5" => Ok(OutputMode::Hdf5),
            #[cfg(not(feature = "hdf5-output"))]
            "hdf5" => {
                report!(
                    Diagnostic::Warning, id == 0, concat!(
                    "complete data output has been requested (dump_all_particles: hdf5), but Ptarmigan\n",
                    "         has not been compiled with HDF5 support. No output will be generated."
                ));
                Ok(OutputMode::None)
            },
            _ => {
                report!(Diagnostic::Error, id == 0, "specified format for complete data output (dump_all_particles: ...) is invalid.");
                Err(InputError::conversion("output:dump_all_particles", "dump_all_particles"))
            }
        })
        .or_else(|e| match e.kind() {
            InputErrorKind::Conversion => Err(e),
            _ => Ok(OutputMode::None)
        })
        ?;

    let keep_decayed_photons = input.read::<bool, _>("output:dump_decayed_photons")
        .unwrap_or(false);

    let laser_defines_z = match input.read::<String,_>("output:coordinate_system") {
        Ok(s) if s == "beam" => false,
        _ => true,
    };

    let (discard_bg_e, discard_bg_ph) = input.read::<bool, _>("output:discard_background")
        .map_or_else(
            |_e| {
                // for backwards compatibility
                let discard_bg_e = input.read("output:discard_background_e").unwrap_or(false);
                (discard_bg_e, false)
            },
            |val| (val, val)
        );

    let min_energy: f64 = input
        .read("output:min_energy")
        .map(|e: f64| 1.0e-6 * e / -ELECTRON_CHARGE) // convert from J to MeV
        .unwrap_or(0.0);

    let max_angle: f64 = input
        .read("output:max_angle")
        .unwrap_or(consts::PI);

    let eospec: Vec<String> = input.read("output:electron")
        .or_else(|e| match e.kind() {InputErrorKind::Location => Ok(vec![]), _ => Err(e)})?;
    let eospec: Vec<DistributionFunction> = eospec
        .iter()
        .map(|spec| DistributionFunction::load(spec, |s| input.evaluate(s)))
        .collect::<Result<Vec<_>,_>>()?;
    
    let gospec: Vec<String> = input.read("output:photon")
        .or_else(|e| match e.kind() {InputErrorKind::Location => Ok(vec![]), _ => Err(e)})?;
    let gospec: Vec<DistributionFunction> = gospec
        .iter()
        .map(|spec| DistributionFunction::load(spec, |s| input.evaluate(s)))
        .collect::<Result<Vec<_>,_>>()?;

    let pospec: Vec<String> = input.read("output:positron")
        .or_else(|e| match e.kind() {InputErrorKind::Location => Ok(vec![]), _ => Err(e)})?;
    let pospec: Vec<DistributionFunction> = pospec
        .iter()
        .map(|spec| DistributionFunction::load(spec, |s| input.evaluate(s)))
        .collect::<Result<Vec<_>,_>>()?;

    let file_format = input.read::<String,_>("output:file_format")
        .and_then(|s| match s.as_str() {
            "plain_text" | "plain-text" | "ascii" => Ok(FileFormat::PlainText),
            "fits" => Ok(FileFormat::Fits),
            _ => Err(InputError::conversion("output", "file_format")),
        })
        .unwrap_or_else(|_| {
            // Error only if dstr output is requested
            let writing_dstrs = eospec.len() + gospec.len() + pospec.len() > 0;
            if writing_dstrs {
                report!(
                    Diagnostic::Warning, id == 0,
                    "file format for distribution output invalid ('plain_text' | 'fits').\n         Continuing with default 'plain_text'."
                );
            }
            FileFormat::PlainText
        });

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
                report!(Diagnostic::Error, id == 0, "unit system requested, \"{}\", is not one of \"auto\", \"hep\", or \"si\".", s);
                Err(InputError::conversion("output:units", "units"))
            }
        })
        ?;

    let statsexpr = input.read("stats:expression")
        .map_or_else(|_| Ok(vec!{}), |strs: Vec<String>| {
            strs.iter()
                .map(|spec| StatsExpression::load(spec, |s| input.evaluate(s)))
                .collect::<Result<Vec<_>, _>>()
        })?;
    
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

    // Temporary warning that stats parsing has changed
    for stat in estats.iter().chain(gstats.iter()).chain(pstats.iter()) {
        if stat.has_version_incompatible_filter() {
            report!(
                Diagnostic::Warning, id == 0, concat!(
                "as of Ptarmigan v1.4.0, all bounds in summary statistics are assumed to be in\n",
                "         SI units. Energy and momentum bounds, previously interpreted as MeV and MeV/c,\n",
                "         must be corrected, using conversion constants if necessary. Continuing..."
            ));
            break;
        }
    }

    // Simulation building and running starts here
    for (run, a0_v) in a0_values.iter().enumerate() {
        let a0: f64 = *a0_v; 
        // Rare event sampling for pair creation
        let pair_rate_increase = input.read::<f64,_>("control:increase_pair_rate_by")
            // if increase is not specified at all, default to unity
            .or_else(|e| match e.kind() {
                InputErrorKind::Location => Ok(1.0),
                _ => Err(e),
            })
            // failing that, check for automatic increase
            .or_else(|e| match input.read::<String,_>("control:increase_pair_rate_by") {
                Ok(s) if s == "auto" => match beam {
                    BeamParameters::FromRng { builder } => {
                        let gamma = builder.gamma;
                        if using_lcfa {
                            Ok(increase_lcfa_pair_rate_by(gamma, a0, wavelength))
                        } else {
                            Ok(increase_pair_rate_by(gamma, a0, wavelength, pol))
                        }
                    },
                    #[cfg(feature = "hdf5-output")]
                    BeamParameters::FromHdf5 { .. } => {
                        report!(Diagnostic::Error, id == 0, "automatic increase in pair creation not available when loading particles from file.");
                        Err(InputError::conversion("control:increase_pair_rate_by", "increase_pair_rate_by"))
                    }
                },
                _ => Err(e),
            })
            .and_then(|r| if r < 1.0 {
                report!(Diagnostic::Error, id == 0, "increase in pair creation rate must be >= 1.0.");
                Err(InputError::conversion("control:increase_pair_rate_by", "increase_pair_rate_by"))
            } else {
                Ok(r)
            })?;

        let seed = 0x8658b90036b165ebu64 + ((rng_seed as u64) * 0x32f55cddaebae910u64);
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        for _i in 0..id {
            rng.jump();
        }

        let laser: Laser = if focusing && !using_lcfa {
            FocusedLaser::new(a0, wavelength, waist, n_cycles, pol, pol_angle)
                .with_envelope(envelope)
                .with_finite_bandwidth(finite_bandwidth)
                .into()
        } else if focusing {
            FastFocusedLaser::new(a0, wavelength, waist, n_cycles, pol, pol_angle)
                .with_envelope(envelope)
                .into()
        } else if !using_lcfa {
            PlaneWave::new(a0, wavelength, n_cycles, pol, pol_angle, chirp_b)
                .with_envelope(envelope)
                .with_finite_bandwidth(finite_bandwidth)
                .into()
        } else {
            FastPlaneWave::new(a0, wavelength, n_cycles, pol, pol_angle, chirp_b)
                .with_envelope(envelope)
                .into()
        };

        let primaries = match beam {
            BeamParameters::FromRng { builder } => {
                let initial_z = laser.ideal_initial_z() + 3.0 * builder.sigma_z;
                builder.with_initial_z(initial_z).build(&mut rng)
            },
            #[cfg(feature = "hdf5-output")]
            BeamParameters::FromHdf5 { ref loader } => {
                let initial_z = laser.ideal_initial_z();
                loader.clone().with_initial_z(initial_z).build(&world)?
            }
        };

        let num = primaries.len();

        let mut current_id = num as u64;

        let merge = |(mut e, mut g, mut p, mut d): (Vec<Particle>, Vec<Particle>, Vec<Particle>, Vec<Particle>), mut sh: Shower| {
            let n0 = ThreeVector::from(sh.primary.momentum()).normalize();
            sh.secondaries.retain(|&pt| {
                let p = pt.momentum();
                let n = ThreeVector::from(p).normalize();
                p[0] > min_energy && n0 * n > max_angle.cos()
            });
            if multiplicity.is_none() || (multiplicity.is_some() && multiplicity.unwrap() == sh.multiplicity()) {
                while let Some(pt) = sh.secondaries.pop() {
                    match pt.species() {
                        Species::Electron => e.push(pt),
                        Species::Photon => g.push(pt),
                        Species::Positron => p.push(pt),
                    }
                }
            }
            d.append(&mut sh.intermediates);
            (e, g, p, d)
        };

        if id == 0 {
            println!("{} {} task{} with {} primary particles per task...", "Running".bold().cyan(), ntasks, if ntasks > 1 {"s"} else {""}, num);
            if a0_values.len() > 1 {
                println!("\t* sim {} of {} at a0 = {}", run + 1, a0_values.len(), a0);
            }
            #[cfg(feature = "with-mpi")] {
                println!("\t* with MPI support enabled");
            }
            #[cfg(feature = "hdf5-output")] {
                println!("\t* writing HDF5 output");
            }
            if pair_rate_increase > 1.0 {
                println!("\t* with pair creation rate increased by {:.3e}", pair_rate_increase);
            }
        }

        let runtime = std::time::Instant::now();

        let options = CollideOptions {
            dt_multiplier,
            rate_increase: pair_rate_increase,
            t_stop: match t_stop {
                StopAt::SameTime | StopAt::Automatic => std::f64::INFINITY,
                StopAt::GivenTime(t) => t
            },
            discard_bg_e,
            discard_bg_ph,
            rr,
            tracking_photons,
            keep_decayed_photons,
            pol_resolved,
            rotate_stokes_pars,
            classical,
            gaunt_factor,
        };

        let (mut electrons, mut photons, mut positrons, mut decayed_photons) = primaries
            .chunks((num / 20).max(1))
            .enumerate()
            .map(|(i, chk)| {
                let tmp = chk.iter()
                    .map(|pt| collide(&laser, *pt, &mut rng, &mut current_id, options))
                    .fold((Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()), merge);
                if id == 0 {
                    println!(
                        "Done {: >12} of {: >12} primaries, RT = {}, ETTC = {}...",
                        (i+1) * chk.len(), num,
                        PrettyDuration::from(runtime.elapsed()),
                        PrettyDuration::from(ettc(runtime, i+1, 20))
                    );
                }
                tmp
            })
            .fold(
                (Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new(), Vec::<Particle>::new()),
                |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat(), [a.2,b.2].concat(), [a.3,b.3].concat())
            );

        // Particle/parent ids are only unique within a single parallel process
        let mut absorption = 0.0;
        let mut id_offsets = vec![0u64; world.size() as usize];
        #[cfg(feature = "with-mpi")]
        world.all_gather_into(&current_id, &mut id_offsets[..]);
        id_offsets.iter_mut().fold(0, |mut total, n| {total += *n; *n = total - *n; total});
        // task n adds id_offsets[n] to each particle/parent id
        for pt in electrons.iter_mut().chain(photons.iter_mut()).chain(positrons.iter_mut()).chain(decayed_photons.iter_mut()) {
            pt.with_id(pt.id() + id_offsets[id as usize]);
            pt.with_parent_id(pt.parent_id() + id_offsets[id as usize]);
            absorption += pt.weight() * pt.absorbed_energy();
        }

        // Fix time coordinates, if necessary
        let t_stop_global = match t_stop {
            StopAt::SameTime => {
                // which particle got furthest?
                let mut ct_local = std::f64::NEG_INFINITY;
                for pt in electrons.iter().chain(photons.iter()).chain(positrons.iter()) {
                    let ct = pt.position()[0];
                    if ct > ct_local { ct_local = ct; }
                }

                // across all MPI tasks
                let mut ct_global = ct_local;
                world.all_reduce_into(&ct_local, &mut ct_global, SystemOperation::max());

                // update all positions
                for pt in electrons.iter_mut().chain(photons.iter_mut()).chain(positrons.iter_mut()) {
                    let r = pt.position();
                    let p = pt.normalized_momentum();
                    let r = r + p * (ct_global - r[0]) / p[0];
                    pt.with_position(r);
                }

                ct_global / SPEED_OF_LIGHT
            },
            StopAt::GivenTime(t) => {
                let ct_global = t * SPEED_OF_LIGHT;

                for pt in electrons.iter_mut().chain(photons.iter_mut()).chain(positrons.iter_mut()) {
                    let r = pt.position();
                    let p = pt.normalized_momentum();
                    let r = r + p * (ct_global - r[0]) / p[0];
                    pt.with_position(r);
                }

                ct_global / SPEED_OF_LIGHT
            },
            _ => std::f64::INFINITY,
        };

        let mut total_absorption = 0.0;
        world.all_reduce_into(&absorption, &mut total_absorption, SystemOperation::sum());
        let total_absorption = total_absorption * 1.0e6 * ELEMENTARY_CHARGE; // in J

        if !laser_defines_z {
            for pt in electrons.iter_mut().chain(photons.iter_mut()).chain(positrons.iter_mut()).chain(decayed_photons.iter_mut()) {
                *pt = pt.to_beam_coordinate_basis(angle, angle2);
            }
        }

        #[cfg(feature = "hdf5-output")]
        let (energy, energy_unit) = laser.energy();
        #[cfg(not(feature = "hdf5-output"))]
        let (energy, _) = laser.energy();

        let f_abs = total_absorption / energy;
        if f_abs > 0.1 {
            report!(
                Diagnostic::Warning, id == 0,
                concat!("obtained laser energy depletion of {}.\n",
                "         Background field approximation likely to be invalid."),
                format!("{:.2}%", 100.0 * f_abs).bold()
            );
        }

        // Updating 'ident' in case of a0 looping
        let current_ident: String = if a0_values.len() > 1 {
            format!("{}{}a0_{:.3}", ident, if ident.is_empty() {""} else {"_"}, a0)
        }
        else {
            ident.to_owned()
        };

        if id == 0 {
            println!("{} distribution output...", "Generating".bold().cyan());
        }

        for dstr in &eospec {
            let prefix = format!("{}{}{}{}electron", output_dir, if output_dir.is_empty() {""} else {"/"}, current_ident, if current_ident.is_empty() {""} else {"_"});
            dstr.write(&world, &electrons, &units, &prefix, file_format)?;
        }

        for dstr in &gospec {
            let prefix = format!("{}{}{}{}photon", output_dir, if output_dir.is_empty() {""} else {"/"}, current_ident, if current_ident.is_empty() {""} else {"_"});
            dstr.write(&world, &photons, &units, &prefix, file_format)?;
        }

        for dstr in &pospec {
            let prefix = format!("{}{}{}{}positron", output_dir, if output_dir.is_empty() {""} else {"/"}, current_ident, if current_ident.is_empty() {""} else {"_"});
            dstr.write(&world, &positrons, &units, &prefix, file_format)?;
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
            if !estats.is_empty() || !gstats.is_empty() || !pstats.is_empty() || !statsexpr.is_empty() {
                use std::fs::File;
                use std::io::Write;
                let filename = format!("{}{}{}{}stats.txt", output_dir, if output_dir.is_empty() {""} else {"/"}, 
                                                                      current_ident, if current_ident.is_empty() {""} else {"_"});
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
                for stat in &statsexpr {
                    writeln!(file, "{}", stat)?;
                }
            }
        }

        match output_mode {
            #[cfg(feature = "hdf5-output")]
            OutputMode::Hdf5 => {
                use hdf5_writer::GroupHolder;
                let filename = format!("{}{}{}{}particles.h5", output_dir, if output_dir.is_empty() {""} else {"/"}, 
                                                               current_ident, if current_ident.is_empty() {""} else {"_"});
                let file = hdf5_writer::ParallelFile::create(&world, &filename)?;

                if id == 0 {
                    println!("{} HDF5 output to {}...", "Writing".bold().cyan(), filename.bold().blue());
                }

                // Build info
                file.new_group("build")?
                    .only_task(0)
                    .new_dataset("version")?
                        .write(env!("CARGO_PKG_VERSION"))?
                    .new_dataset("branch")?
                        .write(env!("VERGEN_GIT_BRANCH"))?
                    .new_dataset("commit-hash")?
                        .write(env!("VERGEN_GIT_SHA"))?
                    .new_dataset("features")?
                        .write(env!("PTARMIGAN_ACTIVE_FEATURES"))?;

                // Top-level run information
                let conf = file.new_group("config")?
                    .only_task(0);

                conf.new_dataset("mpi-tasks")?.write(&ntasks)?
                    .new_dataset("input-file")?.write(raw_input.as_str())?;

                conf.new_group("unit")?
                    .new_dataset("position")?.write(units.length.name())?
                    .new_dataset("momentum")?.write(units.momentum.name())?;

                // Parsed input configuration
                conf.new_group("control")?
                    .new_dataset("dt_multiplier")?.write(&dt_multiplier)?
                    .new_dataset("radiation_reaction")?.write(&rr)?
                    .new_dataset("classical")?.write(&classical)?
                    .new_dataset("pair_creation")?.write(&tracking_photons)?
                    .new_dataset("pair_creation_is_pol_resolved")?.write(&pol_resolved)?
                    .new_dataset("lcfa")?.write(&using_lcfa)?
                    .new_dataset("rng_seed")?.write(&rng_seed)?
                    .new_dataset("increase_pair_rate_by")?.write(&pair_rate_increase)?
                    .new_dataset("bandwidth_correction")?.write(&finite_bandwidth)?
                    .new_dataset("stop_at_time")?
                        .with_unit("s")?
                        .with_desc("tracking has continued until all particles have this time coordinate")?
                        .with_condition(|| matches!(t_stop, StopAt::SameTime | StopAt::GivenTime(_)))
                        .write(&t_stop_global)?
                    .new_dataset("select_multiplicity")?.with_condition(|| multiplicity.is_some()).write(&multiplicity.unwrap_or(0))?
                    .new_dataset("select_multiplicity")?.with_condition(|| multiplicity.is_none()).write(&false)?;

                conf.new_group("laser")?
                    .new_dataset("a0")?
                        .with_unit("1")?
                        .with_desc("peak value of the laser normalized amplitude")?
                        .write(&a0)?
                    .new_dataset("wavelength")?
                        .with_unit(units.length.name())?
                        .with_desc("wavelength of the carrier")?
                        .write(&wavelength.convert(&units.length))?
                    .new_dataset("polarization")?
                        .with_desc("linear/circular")?
                        .write(&pol)?
                    .new_dataset("polarization_angle")?
                        .with_unit("rad")?
                        .with_desc("angle between x-axis and electric field")?
                        .with_condition(|| pol == Polarization::Linear)
                        .write(&pol_angle)?
                    .new_dataset("focusing")?
                        .with_desc("true/false => pulse is modelled in 3d/1d")?
                        .write(&focusing)?
                    .new_dataset("envelope")?
                        .with_desc("pulse envelope of the laser vector potential")?
                        .write(&envelope)?
                    .new_dataset("chirp_b")?
                        .with_unit("1")?
                        .with_desc("parameter that appears in carrier phase = phi + b phi^2")?
                        .write(&chirp_b)?
                    .new_dataset("waist")?
                        .with_unit(units.length.name())?
                        .with_desc("radius in the focal plane at which intensity is 1/e^2 of its peak value")?
                        .with_condition(|| focusing)
                        .write(&waist.convert(&units.length))?
                    .new_dataset("fwhm_duration")?
                        .with_unit("s")?
                        .with_desc("full width at half maximum of the temporal intensity profile")?
                        .with_condition(|| envelope == Envelope::Gaussian)
                        .write(&(n_cycles * wavelength / SPEED_OF_LIGHT))?
                    .new_dataset("n_cycles")?
                        .with_unit("1")?
                        .with_desc("number of wavelengths corresponding to the total pulse duration")?
                        .with_condition(|| matches!(envelope, Envelope::CosSquared | Envelope::Flattop))
                        .write(&n_cycles)?;

                let npart = {
                    let mut npart: usize = 0;
                    world.all_reduce_into(&num, &mut npart, SystemOperation::sum());
                    npart
                };

                let group = conf.new_group("beam")?;

                group.new_dataset("n")?
                        .with_unit("1")?
                        .with_desc("number of primary macroparticles")?
                        .write(&npart)?
                    .new_dataset("species")?
                        .write(species.to_string().as_str())?
                    .new_dataset("collision_angle")?
                        .with_unit("rad")?
                        .with_desc("polar angle between beam momentum and laser wavevector, zero if counterpropagating")?
                        .write(&angle)?
                    .new_dataset("collision_plane_angle")?
                        .with_unit("rad")?
                        .with_desc("azimuthal angle between beam momentum and laser wavevector, zero if p in E, k plane")?
                        .write(&angle2)?
                    .new_dataset("offset")?
                        .with_unit(units.length.name())?
                        .write(&offset.convert(&units.length))?;

                match beam {
                    BeamParameters::FromRng { ref builder } => {
                        let charge = match species {
                            Species::Electron => (npart as f64) * builder.weight * ELECTRON_CHARGE,
                            Species::Positron => (npart as f64) * builder.weight * -ELECTRON_CHARGE,
                            Species::Photon => 0.0,
                        };

                        let (radius, r_max) = builder.radius();

                        group.new_dataset("imported_from_file")?
                                .write(&false)?
                            .new_dataset("n_real")?
                                .with_unit("1")?
                                .with_desc("total number of real particles represented by the primary macroparticles")?
                                .write(&((npart as f64) * builder.weight))?
                            .new_dataset("charge")?.with_unit("C")?.write(&charge)?
                            .new_dataset("gamma")?.with_unit("1")?.write(&builder.gamma)?
                            .new_dataset("sigma")?.with_unit("1")?.write(&builder.sigma)?
                            .new_dataset("bremsstrahlung_source")?.write(&builder.has_brem_spec())?
                            .new_dataset("gamma_min")?.with_unit("1")?.with_condition(|| builder.has_brem_spec()).write(&builder.gamma_min)?
                            .new_dataset("radius")?.with_unit(units.length.name())?.write(&radius.convert(&units.length))?
                            .new_dataset("radius_max")?
                                .with_unit(units.length.name())?
                                .with_desc("density distribution is cut off at this perpendicular distance from the beam axis")?
                                .write(&r_max.convert(&units.length))?
                            .new_dataset("length")?.with_unit(units.length.name())?.write(&builder.sigma_z.convert(&units.length))?
                            .new_dataset("rms_divergence")?.with_unit("rad")?.write(&builder.rms_div)?
                            .new_dataset("polarization")?
                                .with_unit("1")?
                                .with_desc("Stokes parameters of the primary particles: I, Q, U, V")?
                                .with_alias("polarisation")?
                                .write(&builder.pol)?
                            .new_dataset("transverse_distribution_is_normal")?.write(&builder.transverse_dstr_is_normal())?
                            .new_dataset("longitudinal_distribution_is_normal")?.write(&true)?;
                    },
                    BeamParameters::FromHdf5 { ref loader } => {
                        group.new_dataset("imported_from_file")?
                                .write(&true)?
                            .new_dataset("distance_bt_ips")?
                                .with_desc("distance between particle beam source and focal spot")?
                                .with_unit(units.length.name())?
                                .write(&loader.distance_bt_ips.convert(&units.length))?
                            .new_dataset("min_energy")?
                                .with_desc("primaries with energies below this threshold are discarded")?
                                .with_unit(units.energy.name())?
                                .write(&loader.min_energy.convert(&units.energy))?
                            .new_dataset("max_angle")?
                                .with_desc("primaries propagating at angles larger than this are discarded")?
                                .with_unit("rad")?
                                .write(&loader.max_angle)?;
                    }
                }

                conf.new_group("output")?
                    .new_dataset("laser_defines_positive_z")?.write(&laser_defines_z)?
                    .new_dataset("beam_defines_positive_z")?.write(&!laser_defines_z)?
                    .new_dataset("discard_background_e")?.write(&discard_bg_e)?
                    .new_dataset("discard_background_ph")?.write(&discard_bg_ph)?
                    .new_dataset("min_energy")?.with_unit(units.energy.name())?.write(&min_energy.convert(&units.energy))?;

                // Write particle data
                let fs = file.new_group("final-state")?;

                let (x, p, pol, w, a, chi, n, id, pid) = photons
                    .iter()
                    .map(|pt| (
                        pt.position().convert(&units.length),
                        pt.momentum().convert(&units.momentum),
                        pt.polarization(),
                        pt.weight(),
                        pt.payload(),
                        pt.parent_chi(),
                        pt.interaction_count(),
                        pt.id(),
                        pt.parent_id()
                    ))
                    .unzip_n_vec();

                drop(photons);

                fs.new_group("photon")?
                    .new_dataset("weight")?
                        .with_unit("1")?
                        .with_desc("number of real photons each macrophoton represents")?
                        .write(&w[..])?
                    .new_dataset("a0_at_creation")?
                        .with_unit("1")?
                        .with_desc("normalized amplitude (RMS under LMA) at point of emission")?
                        .with_alias("xi")?
                        .write(&a[..])?
                    .new_dataset("parent_chi")?
                        .with_unit("1")?
                        .with_desc("quantum parameter (RMS under LMA) of parent at point of emission")?
                        .write(&chi[..])?
                    .new_dataset("n_pos")?
                        .with_unit("1")?
                        .with_desc("total probability of pair creation for the photon")?
                        .write(&n[..])?
                    .new_dataset("id")?
                        .with_desc("unique ID of the photon")?
                        .write(&id[..])?
                    .new_dataset("parent_id")?
                        .with_desc("ID of the particle that created the photon (for primary particles, parent_id = id")?
                        .write(&pid[..])?
                    .new_dataset("polarization")?
                        .with_desc("Stokes parameters of the photon: I, Q, U, V")?
                        .with_unit("1")?
                        .with_alias("polarisation")?
                        .write(&pol[..])?
                    .new_dataset("position")?
                        .with_unit(units.length.name())?
                        .with_desc("four-position of the photon")?
                        .write(&x[..])?
                    .new_dataset("momentum")?
                        .with_unit(units.momentum.name())?
                        .with_desc("four-momentum of the photon")?
                        .write(&p[..])?;

                let (x, p, w, n, abs, id, pid) = electrons
                    .iter()
                    .map(|pt| (
                        pt.position().convert(&units.length),
                        pt.momentum().convert(&units.momentum),
                        pt.weight(),
                        pt.interaction_count(),
                        pt.absorbed_energy().convert(&units.energy),
                        pt.id(),
                        pt.parent_id()
                    ))
                    .unzip_n_vec();

                drop(electrons);

                fs.new_group("electron")?
                    .new_dataset("weight")?
                        .with_unit("1")?
                        .with_desc("number of real electrons each macroelectron represents")?
                        .write(&w[..])?
                    .new_dataset("n_gamma")?
                        .with_unit("1")?
                        .with_desc("total number of photons emitted by the electron")?
                        .write(&n[..])?
                    .new_dataset("id")?
                        .with_desc("unique ID of the electron")?
                        .write(&id[..])?
                    .new_dataset("parent_id")?
                        .with_desc("ID of the particle that created the electron (for primary particles, parent_id = id)")?
                        .write(&pid[..])?
                    .new_dataset("absorption")?
                        .with_unit(units.energy.name())?
                        .with_desc("energy absorbed from the laser")?
                        .write(&abs[..])?
                    .new_dataset("position")?
                        .with_unit(units.length.name())?
                        .with_desc("four-position of the electron")?
                        .write(&x[..])?
                    .new_dataset("momentum")?
                        .with_unit(units.momentum.name())?
                        .with_desc("four-momentum of the electron")?
                        .write(&p[..])?;

                let (x, x0, p, w, n, abs, id, pid, a, chi) = positrons
                    .iter()
                    .map(|pt| (
                        pt.position().convert(&units.length),
                        pt.was_created_at().convert(&units.length),
                        pt.momentum().convert(&units.momentum),
                        pt.weight(),
                        pt.interaction_count(),
                        pt.absorbed_energy().convert(&units.energy),
                        pt.id(),
                        pt.parent_id(),
                        pt.payload(),
                        pt.parent_chi(),
                    ))
                    .unzip_n_vec();

                drop(positrons);

                fs.new_group("positron")?
                    .new_dataset("weight")?
                        .with_unit("1")?
                        .with_desc("number of real positrons each macropositron represents")?
                        .write(&w[..])?
                    .new_dataset("a0_at_creation")?
                        .with_unit("1")?
                        .with_desc("normalized amplitude (RMS under LMA) at point of creation")?
                        .with_alias("xi")?
                        .write(&a[..])?
                    .new_dataset("parent_chi")?
                        .with_unit("1")?
                        .with_desc("quantum parameter (RMS under LMA) of parent at point of creation")?
                        .write(&chi[..])?
                    .new_dataset("n_gamma")?
                        .with_unit("1")?
                        .with_desc("total number of photons emitted by the positron")?
                        .write(&n[..])?
                    .new_dataset("id")?
                        .with_desc("unique ID of the positron")?
                        .write(&id[..])?
                    .new_dataset("parent_id")?
                        .with_desc("ID of the particle that created the positron (for primary particles, parent_id = id)")?
                        .write(&pid[..])?
                    .new_dataset("absorption")?
                        .with_unit(units.energy.name())?
                        .with_desc("energy absorbed from the laser")?
                        .write(&abs[..])?
                    .new_dataset("position")?
                        .with_unit(units.length.name())?
                        .with_desc("four-position of the positron")?
                        .write(&x[..])?
                    .new_dataset("position_at_creation")?
                        .with_unit(units.length.name())?
                        .with_desc("four-position at which the positron was created")?
                        .write(&x0[..])?
                    .new_dataset("momentum")?
                        .with_unit(units.momentum.name())?
                        .with_desc("four-momentum of the positron")?
                        .write(&p[..])?;

                if keep_decayed_photons {
                    let is = file.new_group("intermediate-state")?;

                    let (x, p, pol, w, a, chi, n, id, pid) = decayed_photons
                        .iter()
                        .map(|pt| (
                            pt.position().convert(&units.length),
                            // pt.was_created_at().convert(&units.length),
                            pt.momentum().convert(&units.momentum),
                            pt.polarization(),
                            pt.weight(),
                            pt.payload(),
                            pt.parent_chi(),
                            pt.interaction_count(),
                            pt.id(),
                            pt.parent_id()
                        ))
                        .unzip_n_vec();

                    drop(decayed_photons);

                    is.new_group("photon")?
                        .new_dataset("weight")?
                            .with_unit("1")?
                            .with_desc("number of real photons each macrophoton represents")?
                            .write(&w[..])?
                        .new_dataset("a0_at_creation")?
                            .with_unit("1")?
                            .with_desc("normalized amplitude (RMS under LMA) at point of emission")?
                            .with_alias("xi")?
                            .write(&a[..])?
                        .new_dataset("parent_chi")?
                            .with_unit("1")?
                            .with_desc("quantum parameter (RMS under LMA) of parent at point of emission")?
                            .write(&chi[..])?
                        .new_dataset("n_pos")?
                            .with_unit("1")?
                            .with_desc("total probability of pair creation for the photon")?
                            .write(&n[..])?
                        .new_dataset("id")?
                            .with_desc("unique ID of the photon")?
                            .write(&id[..])?
                        .new_dataset("parent_id")?
                            .with_desc("ID of the particle that created the photon (for primary particles, parent_id = id")?
                            .write(&pid[..])?
                        .new_dataset("polarization")?
                            .with_desc("Stokes parameters of the photon: I, Q, U, V")?
                            .with_unit("1")?
                            .with_alias("polarisation")?
                            .write(&pol[..])?
                        .new_dataset("position")?
                            .with_unit(units.length.name())?
                            .with_desc("four-position of the photon")?
                            .write(&x[..])?
                        // .new_dataset("position_at_creation")?
                        //     .with_unit(units.length.name())?
                        //     .with_desc("four-position at which the photon was emitted")?
                        //     .write(&x0[..])?
                        .new_dataset("momentum")?
                            .with_unit(units.momentum.name())?
                            .with_desc("four-momentum of the photon")?
                            .write(&p[..])?;
                }

                fs.new_group("laser")?
                    .only_task(0)
                    .new_dataset("energy")?
                        .with_unit(&energy_unit)?
                        .with_desc("total energy of the laser pulse")?
                        .with_condition(|| focusing)
                        .write(&energy)?
                    .new_dataset("energy_flux")?
                        .with_unit(&energy_unit)?
                        .with_desc("total energy of the laser pulse, per unit area")?
                        .with_condition(|| !focusing)
                        .write(&energy)?
                    .new_dataset("absorption")?
                        .with_unit("J")?
                        .with_desc("energy absorbed from the laser")?
                        .write(&total_absorption)?;
            },
            OutputMode::None => {},
        }

        if id == 0 {
            println!("{} run after {}.", "Completed".bold().bright_green(), PrettyDuration::from(runtime.elapsed()));
        }
    }
    Ok(())
}

/// Wraps ptarmigan_main for error reporting.
fn main() -> ExitCode {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let id = world.rank();

    if std::env::var("SLURM_JOB_ID").is_ok() || std::env::var("PBS_JOBID").is_ok() {
        // Running as a batch job, disable colouring
        colored::control::set_override(false);
    }

    let base = if cfg!(target_os = "windows") { "*" } else { "═" };
    let bar = base.repeat(36);
    let title = concat!(" ", "Ptarmigan v", env!("CARGO_PKG_VERSION"), " ");

    if id == 0 {
        println!("{}{}{}", bar.blue(), title.bold(), bar.blue());
    }

    let exit_code = match ptarmigan_main(world) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            report!(Diagnostic::Error, id == 0, "{:?}", e);
            ExitCode::FAILURE
        }
    };

    if id == 0 {
        let centre = base.repeat(title.len());
        println!("{}{}{}", bar.blue(), centre.blue(), bar.blue());
    }

    exit_code
}
