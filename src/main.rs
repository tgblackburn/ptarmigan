use std::error::Error;
use std::path::{Path, PathBuf};

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

mod constants;
mod field;
mod geometry;
mod particle;
mod nonlinear_compton;
mod special_functions;
mod output;
mod input;

use constants::*;
use field::*;
use geometry::*;
use particle::*;
use output::*;
use input::*;

fn collide<F: Field, R: Rng>(field: &F, incident: Particle, rng: &mut R, dt_multiplier: f64) -> Shower {
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
            let photon = Particle::create(Species::Photon, r)
                .with_normalized_momentum(k);
            secondaries.push(photon);
            u = u - k;
        }

        primary.with_position(r);
        primary.with_normalized_momentum(u);
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

    let mut input = Config::from_file(&path)?;
    input.with_context("constants");

    let dt_multiplier = input.read("control", "dt_multiplier").unwrap_or(1.0);
    let multiplicity: Option<usize> = input.read("control", "select_multiplicity").ok();

    let a0: f64 = input.read("laser", "a0")?;
    let wavelength: f64 = input.read("laser", "wavelength")?;
    let pol = Polarization::Circular;

    let (focusing, waist) = input
        .read("laser", "waist")
        .map(|w| (true, w))
        .unwrap_or((false, std::f64::INFINITY));

    let tau: f64 = if focusing {
        input.read("laser", "fwhm_duration")?
    } else {
        input.read("laser", "n_cycles")?
    };

    let num: usize = input.read("beam", "ne")?;
    let gamma: f64 = input.read("beam", "gamma")?;
    let sigma: f64 = input.read("beam", "sigma").unwrap_or(0.0);
    let radius: f64 = input.read("beam", "radius")?;
    let length: f64 = input.read("beam", "length").unwrap_or(0.0);

    let ident: String = input.read("output", "ident").unwrap_or_else(|_| "".to_owned());

    let eospec: Vec<String> = input.read("output", "electron")?;
    let eospec: Vec<DistributionFunction> = eospec
        .iter()
        .map(|s| s.parse::<DistributionFunction>().unwrap())
        .collect();
    
    let pospec: Vec<String> = input.read("output", "photon")?;
    let pospec: Vec<DistributionFunction> = pospec
        .iter()
        .map(|s| s.parse::<DistributionFunction>().unwrap())
        .collect();

    let mut rng = Xoshiro256StarStar::seed_from_u64(id as u64);
    let num = num / (ntasks as usize);

    if id == 0 {
        println!("Running {} task{} with {} primary particles per task...", ntasks, if ntasks > 1 {"s"} else {""}, num);
        #[cfg(feature = "with-mpi")] {
            println!("\t* with MPI support enabled");
        }
        #[cfg(feature = "fits-output")] {
            println!("\t* writing FITS output");
        }
    }

    let primaries: Vec<Particle> = (0..num).into_iter()
        .map(|_i| {
            let z = if focusing {
                2.0 * SPEED_OF_LIGHT * tau
            } else {
                0.5 * wavelength * tau
            };
            let z = z + length * rng.sample::<f64,_>(StandardNormal);
            let x = radius * rng.sample::<f64,_>(StandardNormal);
            let y = radius * rng.sample::<f64,_>(StandardNormal);
            let r = FourVector::new(-z, x, y, z);
            let u = -(gamma * gamma - 1.0f64).sqrt();
            let u = u + sigma * rng.sample::<f64,_>(StandardNormal);
            let u = FourVector::new(0.0, 0.0, 0.0, u).unitize();
            Particle::create(Species::Electron, r)
                .with_normalized_momentum(u)
                .with_optical_depth(rng.sample(Exp1))
        })
        .collect();

    let merge = |(mut p, mut s): (Vec<Particle>, Vec<Particle>), mut sh: Shower| {
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

    let (electrons, photons) = if focusing {
        let laser = FocusedLaser::new(a0, wavelength, waist, tau, pol);
        primaries
            .chunks(num / 20)
            .enumerate()
            .map(|(i, chk)| {
                let tmp = chk.iter()
                    .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier))
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
    } else {
        let laser = PlaneWave::new(a0, wavelength, tau, pol);
        primaries
        .chunks(num / 20)
        .enumerate()
        .map(|(i, chk)| {
            let tmp = chk.iter()
                .map(|pt| collide(&laser, *pt, &mut rng, dt_multiplier))
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

    for dstr in &eospec {
        let prefix = format!("{}{}{}{}electron", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
        dstr.write(&world, &electrons, &prefix)?;
    }

    for dstr in &pospec {
        let prefix = format!("{}{}{}{}photon", output_dir, if output_dir.is_empty() {""} else {"/"}, ident, if ident.is_empty() {""} else {"_"});
        dstr.write(&world, &photons, &prefix)?;
    }

    if id == 0 {
        println!("Run complete after {}.", PrettyDuration::from(runtime.elapsed()));
    }

    Ok(())
}
