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
use rand_distr::{Exp1, StandardNormal};
use rand_xoshiro::*;

#[cfg(feature = "hdf5-output")]
unzip_n::unzip_n!(pub 4);

mod constants;
mod field;
mod geometry;
mod particle;
mod nonlinear_compton;
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

#[cfg(feature = "hdf5-output")]
trait Writeable<T> {
    fn write(&self, name: &str, value: T) -> hdf5::Result<&Self>;
    fn write_all(&self, name: &str, value: &[T]) -> hdf5::Result<&Self>;
    fn write_if(&self, condition: bool, name: &str, value: T) -> hdf5::Result<&Self> {
        if condition {
            self.write(name, value)
        } else {
            Ok(self)
        }
    }
}

#[cfg(feature = "hdf5-output")]
trait WriteableString {
    fn write_str(&self, name: &str, value: &str) -> hdf5::Result<&Self>;
}

#[cfg(feature = "hdf5-output")]
impl<T: hdf5::types::H5Type> Writeable<T> for hdf5::Group {
    fn write(&self, name: &str, value: T) -> hdf5::Result<&Self> {
        self.new_dataset::<T>().create(name, ())?.write_scalar(&value).map(|_| self)
    }

    fn write_all(&self, name: &str, value: &[T]) -> hdf5::Result<&Self> {
        self.new_dataset::<T>().create(name, value.len())?.write(value).map(|_| self)
    }
}

#[cfg(feature = "hdf5-output")]
impl WriteableString for hdf5::Group {
    fn write_str(&self, name: &str, value: &str) -> hdf5::Result<&Self> {
        use std::str::FromStr;
        use hdf5::types::VarLenUnicode;
        match VarLenUnicode::from_str(value) {
            Ok(vlu) => self.new_dataset::<VarLenUnicode>().create(name, ())?.write_scalar(&vlu).map(|_| self),
            Err(e) => Err(hdf5::Error::Internal(e.to_string()))
        }
    }
}

fn collide<F: Field, R: Rng>(field: &F, incident: Particle, rng: &mut R, dt_multiplier: f64, current_id: &mut u64) -> Shower {
    let mut primary = incident;
    let mut secondaries: Vec<Particle> = Vec::new();
    let dt = field.max_timestep().unwrap_or(1.0);
    let dt = dt * dt_multiplier;

    while field.contains(primary.position()) {
        let (r, mut u) = field.push(
            primary.position(), 
            primary.normalized_momentum(),
            primary.charge_to_mass_ratio(),
            dt
        );

        if let Some(k) = field.radiate(r, u, dt, rng) {
            let id = *current_id;
            *current_id = *current_id + 1;
            let photon = Particle::create(Species::Photon, r)
                .with_payload((u * u - 1.0).max(0.0).sqrt())
                .with_weight(primary.weight())
                .with_id(id)
                .with_normalized_momentum(k);
            secondaries.push(photon);

            #[cfg(not(feature = "no-radiation-reaction"))] {
                u = u - k;
            }

            primary.update_interaction_count(1.0);
        }

        primary.with_position(r);
        primary.with_normalized_momentum(u);
        //primary.with_payload((u * u - 1.0).max(0.0).sqrt());
    }

    Shower {
        primary,
        secondaries,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let id = world.rank();
    let ntasks = world.size();

    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or(ConfigError::raise(ConfigErrorKind::MissingFile, "", ""))?;
    let path = PathBuf::from(path);
    let output_dir = path.parent().unwrap_or(Path::new("")).to_str().unwrap_or("");

    // Read input configuration with default context

    let raw_input = std::fs::read_to_string(&path)
        .map_err(|_| ConfigError::raise(ConfigErrorKind::MissingFile, "", ""))?;
    let mut input = Config::from_string(&raw_input)?;
    //let mut input = Config::from_file(&path)?;
    input.with_context("constants");

    let dt_multiplier = input.read("control", "dt_multiplier").unwrap_or(1.0);
    let multiplicity: Option<usize> = input.read("control", "select_multiplicity").ok();
    let using_lcfa = input.read("control", "lcfa").unwrap_or(false);
    let rng_seed = input.read("control", "rng_seed").unwrap_or(0usize);

    let a0: f64 = input.read("laser", "a0")?;
    let wavelength: f64 = input
        .read("laser", "wavelength")
        .or_else(|_e|
            // attempt to read a frequency instead, e.g. 'omega: 1.55 * eV'
            input.read("laser", "omega").map(|omega: f64| 2.0 * consts::PI * COMPTON_TIME * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3) / omega)
        )?;

    let pol = match input.read::<String>("laser", "polarization") {
        Ok(s) if s == "linear" => Polarization::Linear,
        Ok(s) if s == "circular" => Polarization::Circular,
        _ => Polarization::Circular
    };

    if !using_lcfa && pol == Polarization::Linear {
        panic!("LMA rates are implemented for circularly polarized waves only!");
    }

    let (focusing, waist) = input
        .read("laser", "waist")
        .map(|w| (true, w))
        .unwrap_or((false, std::f64::INFINITY));

    let tau: f64 = if focusing && !cfg!(feature = "cos2-envelope-in-3d") {
        input.read("laser", "fwhm_duration")?
    } else {
        input.read("laser", "n_cycles")?
    };

    let chirp_b = if !focusing {
        input.read("laser", "chirp_coeff").unwrap_or(0.0)
    } else {
        input.read("laser", "chirp_coeff")
            .map(|_: f64| {
                eprintln!("Chirp parameter ignored for focusing laser pulses.");
                0.0
            })
            .unwrap_or(0.0)
    };

    let num: usize = input.read("beam", "ne")?;
    let gamma: f64 = input.read("beam", "gamma")?;
    let sigma: f64 = input.read("beam", "sigma").unwrap_or(0.0);
    let length: f64 = input.read("beam", "length").unwrap_or(0.0);
    let angle: f64 = input.read("beam", "collision_angle").unwrap_or(0.0);
    let rms_div: f64 = input.read("beam", "rms_divergence").unwrap_or(0.0);
    let weight = input.read("beam", "charge")
        .map(|q: f64| q.abs() / (constants::ELEMENTARY_CHARGE * (num as f64)))
        .unwrap_or(1.0);
    let (radius, normally_distributed) = input.read::<Vec<String>>("beam", "radius")
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
                Err(ConfigError::raise(ConfigErrorKind::ConversionFailure, "beam", "radius"))
            }
        })?;

    let ident: String = input.read("output", "ident").unwrap_or_else(|_| "".to_owned());

    let plain_text_output = match input.read::<String>("output", "dump_all_particles") {
        Ok(s) if s == "plain_text" || s == "plain-text" => OutputMode::PlainText,
        #[cfg(feature = "hdf5-output")]
        Ok(s) if s == "hdf5" => OutputMode::Hdf5,
        _ => OutputMode::None,
    };

    let laser_defines_z = match input.read::<String>("output", "coordinate_system") {
        Ok(s) if s == "beam" => false,
        _ => true,
    };

    let min_energy: f64 = input
        .read("output", "min_energy")
        .map(|e: f64| 1.0e-6 * e / -ELECTRON_CHARGE) // convert from J to MeV
        .unwrap_or(0.0);

    let eospec: Vec<String> = input.read("output", "electron")?;
    let eospec: Vec<DistributionFunction> = eospec
        .iter()
        .map(|s| s.parse())
        .collect::<Result<Vec<_>,_>>()?;
    
    let pospec: Vec<String> = input.read("output", "photon")?;
    let pospec: Vec<DistributionFunction> = pospec
        .iter()
        .map(|s| s.parse())
        .collect::<Result<Vec<_>,_>>()?;

    let mut estats = input.read("stats", "electron")
        .map_or_else(|_| Ok(vec![]), |strs: Vec<String>| {
            strs.iter()
                .map(|spec| SummaryStatistic::load(spec, |s| input.evaluate(s)))
                .collect::<Result<Vec<_>,_>>()
        })?;

    let mut pstats = input.read("stats", "photon")
        .map_or_else(|_| Ok(vec![]), |strs: Vec<String>| {
            strs.iter()
                .map(|spec| SummaryStatistic::load(spec, |s| input.evaluate(s)))
                .collect::<Result<Vec<_>,_>>()
        })?;

    let local_seed = (id as u64) * (1 + rng_seed as u64);
    let mut rng = Xoshiro256StarStar::seed_from_u64(local_seed);
    let num = num / (ntasks as usize);

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
        #[cfg(feature = "no-radiation-reaction")] {
            println!("\t* with radiation reaction disabled");
        }
        #[cfg(feature = "cos2-envelope-in-3d")] {
            if focusing {
                println!("\t* with cos^2 temporal envelope");
            }
        }
    }

    let primaries: Vec<Particle> = (0..num).into_iter()
        .map(|i| {
            let z = if focusing {
                if cfg!(feature = "cos2-envelope-in-3d") {
                    wavelength * tau + 3.0 * length
                } else {
                    2.0 * SPEED_OF_LIGHT * tau + 3.0 * length
                }
            } else {
                0.5 * wavelength * tau
            };
            let t = -z;
            let z = z + length * rng.sample::<f64,_>(StandardNormal);
            let (x, y) = if normally_distributed {
                (
                    radius * rng.sample::<f64,_>(StandardNormal),
                    radius * rng.sample::<f64,_>(StandardNormal)
                )
            } else { // uniformly distributed
                let r = radius * rng.gen::<f64>().sqrt();
                let theta = 2.0 * consts::PI * rng.gen::<f64>();
                (r * theta.cos(), r * theta.sin())
            };
            let r = ThreeVector::new(x, y, z);
            let r = r.rotate_around_y(angle);
            let r = FourVector::new(t, r[0], r[1], r[2]);
            let u = -(gamma * gamma - 1.0f64).sqrt();
            let u = u + sigma * rng.sample::<f64,_>(StandardNormal);
            let theta_x = angle + rms_div * rng.sample::<f64,_>(StandardNormal);
            let theta_y = rms_div * rng.sample::<f64,_>(StandardNormal);
            let u = FourVector::new(0.0, u * theta_x.sin() * theta_y.cos(), u * theta_y.sin(), u * theta_x.cos() * theta_y.cos()).unitize();
            Particle::create(Species::Electron, r)
                .with_normalized_momentum(u)
                .with_optical_depth(rng.sample(Exp1))
                .with_weight(weight)
                .with_id(i as u64)
        })
        .collect();

    let mut current_id = num as u64;

    let merge = |(mut p, mut s): (Vec<Particle>, Vec<Particle>), mut sh: Shower| {
        sh.secondaries.retain(|&pt| pt.momentum()[0] > min_energy);
        if let Some(m) = multiplicity {
            if m == sh.multiplicity() {
                p.push(sh.primary);
                s.append(&mut sh.secondaries);
            }
        } else {
            p.push(sh.primary);
            s.append(&mut sh.secondaries);
        }
        (p, s)
    };

    let runtime = std::time::Instant::now();

    let (mut electrons, mut photons) = if focusing && !using_lcfa {
        let laser = FocusedLaser::new(a0, wavelength, waist, tau, pol);
        //println!("total energy = {}", laser.total_energy());
        primaries
            .chunks(num / 20)
            .enumerate()
            .map(|(i, chk)| {
                let tmp = chk.iter()
                    .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier, &mut current_id))
                    .fold((Vec::<Particle>::new(), Vec::<Particle>::new()), merge);
                if id == 0 {
                    println!("Done {: >12} of {: >12} primaries, RT = {}, ETTC = {}...",
                    (i+1) * chk.len(), num,
                    PrettyDuration::from(runtime.elapsed()),
                    PrettyDuration::from(ettc(runtime, i+1, 20)));
                }
                tmp
            })
            .fold(
                (Vec::<Particle>::new(), Vec::<Particle>::new()),
                |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat())
            )
    } else if focusing { // and using LCFA rates
        let laser = FastFocusedLaser::new(a0, wavelength, waist, tau, pol);
        primaries
            .chunks(num / 20)
            .enumerate()
            .map(|(i, chk)| {
                let tmp = chk.iter()
                    .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier, &mut current_id))
                    .fold((Vec::<Particle>::new(), Vec::<Particle>::new()), merge);
                if id == 0 {
                    println!("Done {: >12} of {: >12} primaries, RT = {}, ETTC = {}...",
                    (i+1) * chk.len(), num,
                    PrettyDuration::from(runtime.elapsed()),
                    PrettyDuration::from(ettc(runtime, i+1, 20)));
                }
                tmp
            })
            .fold(
                (Vec::<Particle>::new(), Vec::<Particle>::new()),
                |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat())
            )
    } else if !using_lcfa {
        let laser = PlaneWave::new(a0, wavelength, tau, pol, chirp_b);
        primaries
        .chunks(num / 20)
        .enumerate()
        .map(|(i, chk)| {
            let tmp = chk.iter()
                .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier, &mut current_id))
                .fold((Vec::<Particle>::new(), Vec::<Particle>::new()), merge);
            if id == 0 {
                println!("Done {: >12} of {: >12} primaries, RT = {}, ETTC = {}...",
                (i+1) * chk.len(), num,
                PrettyDuration::from(runtime.elapsed()),
                PrettyDuration::from(ettc(runtime, i+1, 20)));
            }
            tmp
        })
        .fold(
            (Vec::<Particle>::new(), Vec::<Particle>::new()),
            |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat())
        )
    } else { // plane wave and lcfa
        let laser = FastPlaneWave::new(a0, wavelength, tau, pol, chirp_b);
        primaries
        .chunks(num / 20)
        .enumerate()
        .map(|(i, chk)| {
            let tmp = chk.iter()
                .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier, &mut current_id))
                .fold((Vec::<Particle>::new(), Vec::<Particle>::new()), merge);
            if id == 0 {
                println!("Done {: >12} of {: >12} primaries, RT = {}, ETTC = {}...",
                (i+1) * chk.len(), num,
                PrettyDuration::from(runtime.elapsed()),
                PrettyDuration::from(ettc(runtime, i+1, 20)));
            }
            tmp
        })
        .fold(
            (Vec::<Particle>::new(), Vec::<Particle>::new()),
            |a, b| ([a.0,b.0].concat(), [a.1,b.1].concat())
        )
    };

    if !laser_defines_z {
        electrons.iter_mut().for_each(|pt| *pt = pt.to_beam_coordinate_basis(angle));
        photons.iter_mut().for_each(|pt| *pt = pt.to_beam_coordinate_basis(angle));
    }

    for dstr in &eospec {
        let prefix = format!("{}{}{}{}electron", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
        dstr.write(&world, &electrons, &prefix)?;
    }

    for dstr in &pospec {
        let prefix = format!("{}{}{}{}photon", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
        dstr.write(&world, &photons, &prefix)?;
    }

    for stat in estats.iter_mut() {
        stat.evaluate(&world, &electrons, "electron");
    }

    for stat in pstats.iter_mut() {
        stat.evaluate(&world, &photons, "photon");
    }

    if id == 0 {
        if !estats.is_empty() || !pstats.is_empty() {
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
        }
    }

    match plain_text_output {
        OutputMode::PlainText => {
            let mut particles = [electrons, photons].concat();

            if id == 0 {
                use std::fs::File;
                use std::io::Write;
                let filename = format!("{}{}{}{}particles.out", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
                let mut file = File::create(filename)?;
                writeln!(file, "#{:-^1$}", "", 170)?;
                writeln!(file, "# Particle properties when tracking stops")?;
                writeln!(file, "#{:-^1$}", "", 170)?;
                writeln!(file, "# First interacting species: electron\t\tSecond interacting species: laser")?;
                writeln!(file, "# First initial particle energy = {:.4} +/- {:.4} GeV, Sigma_xyz = {:.2} {:.2} {:.2} microns", 1.0e-3 * gamma * ELECTRON_MASS_MEV, 1.0e-3 * sigma * ELECTRON_MASS_MEV, 1.0e6 * radius, 1.0e6 * radius, 1.0e6 * length)?;
                writeln!(file, "# Laser peak intensity = {:.2} x 10^18 W/cm^2, wavelength = {:.2} nm, pulse length = {:.2} fs, beam waist = {:.2} microns", a0.powi(2) * 1.37 / (1.0e6 * wavelength).powi(2), 1.0e9 * wavelength, 1.0e15 * tau, 1.0e6 * waist)?;
                writeln!(file, "# Pulse peak xi = {:.4}, chi = {:.4}", a0, (2.0 * consts::PI * SPEED_OF_LIGHT * COMPTON_TIME / wavelength) * a0 * gamma * (1.0 + angle.cos()))?;
                writeln!(file, "#{:-^1$}", "", 170)?;
                writeln!(file, "# E (GeV)\tx (micron)\ty (micron)\tz (micron)\tp_x (GeV/c)\tp_y (GeV/c)\tp_z (GeV/c)\tPDG_NUM\tMP_Wgt\tMP_ID\tt (um/c)\txi")?;
                //writeln!(file, "# E (GeV)\tx (micron)\ty (micron)\tz (micron)\tbeta_x\tbeta_y\tbeta_z\tPDG_NUM\tMP_Wgt\tMP_ID\tt (um/c)\txi")?;
                writeln!(file, "#{:-^1$}", "", 170)?;

                let mut current_id = particles.len() as u64;

                #[cfg(feature = "with-mpi")]
                for recv_rank in 1..ntasks {
                    particles = world.process_at_rank(recv_rank).receive_vec::<Particle>().0;
                    for pt in &particles {
                        writeln!(file, "{}", pt.clone().with_id(pt.id() + current_id))?;
                    }
                    current_id += particles.len() as u64;
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
                    .write_str("version", env!("CARGO_PKG_VERSION"))?
                    .write_str("branch",env!("VERGEN_GIT_BRANCH"))?
                    .write_str("commit-hash",env!("VERGEN_GIT_SHA"))?
                    .write_str("features", env!("PTARMIGAN_ACTIVE_FEATURES"))?;

                // Top-level run information
                let conf = file.create_group("config")?;
                conf.write("mpi-tasks", ntasks)?
                    .write_str("input-file", &raw_input)?;

                // Parsed input configuration
                conf.create_group("control")?
                    .write("dt_multiplier", dt_multiplier)?
                    .write("lcfa", using_lcfa)?
                    .write("rng_seed", rng_seed)?
                    .write_if(multiplicity.is_some(), "select_multiplicity", multiplicity.unwrap_or(0))?
                    .write_if(multiplicity.is_none(), "select_multiplicity", false)?;

                conf.create_group("laser")?
                    .write("a0", a0)?
                    .write("wavelength", wavelength)?
                    .write("polarization", pol)?
                    .write("focusing", focusing)?
                    .write("chirp_b", chirp_b)?
                    .write_if(focusing, "waist", waist)?
                    .write_if(focusing, "fwhm_duration", tau)?
                    .write_if(!focusing, "n_cycles", tau)?;

                conf.create_group("beam")?
                    .write("ne", num)?
                    .write("gamma", gamma)?
                    .write("sigma", sigma)?
                    .write("radius", radius)?
                    .write("length", length)?
                    .write("collision_angle", angle)?
                    .write("rms_divergence", rms_div)?
                    .write("transverse_distribution_is_normal", normally_distributed)?
                    .write("longitudinal_distribution_is_normal", true)?;

                conf.create_group("output")?
                    .write("laser_defines_positive_z", laser_defines_z)?
                    .write("beam_defines_positive_z", !laser_defines_z)?
                    .write("min_energy", min_energy)?;

                // Write particle data
                let fs = file.create_group("final-state")?;

                let mut photons = photons;
                #[cfg(feature = "with-mpi")]
                for recv_rank in 1..ntasks {
                    let mut recv = world.process_at_rank(recv_rank).receive_vec::<Particle>().0;
                    photons.append(&mut recv);
                }
                let (x, p, w, a) = photons
                    .iter()
                    .map(|pt| (pt.position(), pt.momentum(), pt.weight(), pt.payload()))
                    .unzip_n_vec();
                drop(photons);

                fs.create_group("photon")?
                    .write_all("weight", &w)?
                    .write_all("a0_at_creation", &a)?
                    .write_all("position", &x)?
                    .write_all("momentum", &p)?;

                // Provide alias for a0
                fs.group("photon")?.link_soft("a0_at_creation", "xi")?;

                let mut electrons = electrons;
                #[cfg(feature = "with-mpi")]
                for recv_rank in 1..ntasks {
                    let mut recv = world.process_at_rank(recv_rank).receive_vec::<Particle>().0;
                    electrons.append(&mut recv);
                }
                let (x, p, w, n) = electrons
                    .iter()
                    .map(|pt| (pt.position(), pt.momentum(), pt.weight(), pt.interaction_count()))
                    .unzip_n_vec();
                drop(electrons);

                fs.create_group("electron")?
                    .write_all("weight", &w)?
                    .write_all("n_gamma", &n)?
                    .write_all("position", &x)?
                    .write_all("momentum", &p)?;
            } else {
                #[cfg(feature = "with-mpi")] {
                    world.process_at_rank(0).synchronous_send(&photons[..]);
                    drop(photons);
                    world.process_at_rank(0).synchronous_send(&electrons[..]);
                    drop(electrons);
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
