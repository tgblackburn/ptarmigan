//! Loading a particle beam from a binary file

use colored::Colorize;

use hdf5_writer;
use hdf5_writer::{
    GroupHolder,
    ParallelFile,
    OutputError,
};

#[cfg(feature = "with-mpi")]
use mpi::traits::*;

#[cfg(not(feature = "with-mpi"))]
extern crate no_mpi as mpi;

#[cfg(not(feature = "with-mpi"))]
use mpi::Communicator;

use super::{Particle, Species};
use crate::constants::*;
use crate::{HasUnit, Unit};
use crate::geometry::{FourVector, ThreeVector, StokesVector};

#[derive(Clone)]
pub struct BeamLoader {
    filename: String,
    species: Species,
    weight_path: String,
    position_path: String,
    momentum_path: String,
    polarization_path: String,
    pub distance_bt_ips: f64,
    auto_timing: bool,
    initial_z: f64,
    collision_angle: f64,
    offset: ThreeVector,
    pub min_energy: f64,
    pub max_angle: f64,
}

impl BeamLoader {
    pub fn from_file(filename: &str, species: &str, distance_bt_ips: f64) -> Self {
        let weight_path = format!("final-state/{}/weight", species);
        let position_path = format!("final-state/{}/position", species);
        let momentum_path = format!("final-state/{}/momentum", species);
        let polarization_path = format!("final-state/{}/polarization", species);

        Self {
            filename: filename.to_owned(),
            species: species.parse().unwrap(),
            weight_path,
            position_path,
            momentum_path,
            polarization_path,
            distance_bt_ips,
            auto_timing: true,
            initial_z: 0.0,
            collision_angle: 0.0,
            offset: [0.0, 0.0, 0.0].into(),
            min_energy: 0.0,
            max_angle: std::f64::consts::PI,
        }
    }

    pub fn with_auto_timing(self, auto_timing: bool) -> Self {
        BeamLoader {
            auto_timing,
            ..self
        }
    }

    pub fn with_initial_z(self, initial_z: f64) -> Self {
        BeamLoader {
            initial_z,
            ..self
        }
    }

    pub fn with_collision_angle(self, angle: f64) -> Self {
        BeamLoader {
            collision_angle: angle,
            ..self
        }
    }

    pub fn with_offset(self, offset: ThreeVector) -> Self {
        BeamLoader {
            offset,
            ..self
        }
    }

    /// Discard particles that have energy less than the threshold
    /// `min_energy` (given in MeV).
    pub fn with_min_energy(self, min_energy: f64) -> Self {
        BeamLoader {
            min_energy,
            ..self
        }
    }

    pub fn with_max_angle(self, max_angle: f64) -> Self {
        BeamLoader {
            max_angle,
            ..self
        }
    }

    fn rotation_angle(&self, beam_axis: ThreeVector) -> f64 {
        let new_axis = ThreeVector::new(-self.collision_angle.sin(), 0.0, -self.collision_angle.cos());
        let theta = (new_axis * beam_axis).acos().abs().copysign(beam_axis.cross(new_axis)[1]);
        theta
    }

    fn realign_particle(&self, r: FourVector, p: FourVector, theta: f64) -> (FourVector, FourVector) {
        let p = p.rotate_around_y(theta);
        let r = r.rotate_around_y(theta);

        // timing/alignment errors
        let r = r + self.offset.rotate_around_y(self.collision_angle).with_time(0.0);

        if self.auto_timing {
            // The particle worldline as a function of lab time is given by
            //   r = r' + L (-1, sin theta, 0, cos theta) + (t + L) beta
            // where r' is the position relative to the old origin and
            // beta = p / p0 is the velocity.
            // It needs to be initialised suitably far from the laser pulse, i.e.
            // for some negative t^- = t - z = -2 z0.
            let beta = p / p[0];
            let offset: FourVector = [-1.0, self.collision_angle.sin(), 0.0, self.collision_angle.cos()].into();
            let offset = self.distance_bt_ips * (offset + (1.0 + self.collision_angle.cos()) * beta / (beta[0] - beta[3]));
            let offset = offset - (2.0 * self.initial_z + r[0] - r[3]) * beta / (beta[0] - beta[3]);
            let r = r + offset;
            (r, p)
        } else {
            (r, p)
        }
    }

    pub fn build<C>(&self, comm: &C) -> Result<Vec<Particle>, OutputError> where C: Communicator {
        use crate::report;

        let id = comm.rank();
        if id == 0 {
            println!("{} incident particle beam from {}...", "Importing".bold().cyan(), self.filename.bold().blue());
        }

        let file = ParallelFile::open(comm, &self.filename)?;

        // What unit system did the output use?
        let x_unit: Unit = file
            .open_dataset("config/unit/position")?
            .read::<String>()?
            .parse()
            .map_err(|_| OutputError::TypeMismatch("valid length unit".to_owned()))
            ?;

        let p_unit: Unit = file
            .open_dataset("config/unit/momentum")?
            .read::<String>()?
            .parse()
            .map_err(|_| OutputError::TypeMismatch("valid momentum unit".to_owned()))
            ?;

        // println!("\tgot length unit = {:?} and momentum unit = {:?}", x_unit, p_unit);

        // Translating the coordinate system
        let beam_axis = file
            .open_dataset("config/output/beam_defines_positive_z")
            .and_then(|ds| {
                let beam_defines_positive_z = ds.read::<bool>().unwrap_or(false);
                if beam_defines_positive_z {
                    Ok(ThreeVector::new(0.0, 0.0, 1.0))
                } else {
                    Err(OutputError::Missing("coordinate system".to_owned()))
                }
            })
            .or_else(|_| {
                let laser_defines_positive_z = file.open_dataset("config/output/laser_defines_positive_z")
                    .and_then(|ds| ds.read::<bool>())
                    .unwrap_or(false);
                let collision_angle = file.open_dataset("config/beam/collision_angle")
                    .and_then(|ds| ds.read::<f64>());

                if laser_defines_positive_z && collision_angle.is_ok() {
                    let theta = collision_angle.unwrap();
                    Ok(ThreeVector::new(-theta.sin(), 0.0, -theta.cos()))
                } else {
                    Err(OutputError::Missing("coordinate_system".to_owned()))
                }
            })
            .or_else(|_| {
                file.open_dataset("beam_axis")
                    .and_then(|ds| ds.read::<String>())
                    .and_then(|s| match s.as_str() {
                        "+x" => Ok(ThreeVector::new(1.0, 0.0, 0.0)),
                        "-x" => Ok(ThreeVector::new(-1.0, 0.0, 0.0)),
                        "+z" => Ok(ThreeVector::new(0.0, 0.0, 1.0)),
                        "-z" => Ok(ThreeVector::new(0.0, 0.0, -1.0)),
                        _ => {
                            report!(Diagnostic::Error, id == 0, "beam axis \"{}\" not recognised, halting.", s);
                            return Err(OutputError::Missing("beam_axis".to_owned()));
                        }
                    })
            })
            .unwrap_or_else(|_| {
                report!(Diagnostic::Warning, id == 0, "no coordinate system specified, assuming beam propagates towards positive z...");
                ThreeVector::new(0.0, 0.0, 1.0) // assume beam defines z
            });

        let theta = self.rotation_angle(beam_axis);

        let weight = file.open_dataset(&self.weight_path)?
            .read::<[f64]>()?
            .take();

        // println!("\tgot {} weights ({}, ..., {})", weight.len(), weight[0], weight[weight.len()-1]);

        let dataset = file.open_dataset(&self.position_path)?;
        // let unit: Result<Unit, _> = dataset.open_attribute("unit")
        //     .and_then(|ds| ds.read::<String>())
        //     .and_then(|s| s.parse().map_err(|_| OutputError::TypeMismatch("valid position unit".to_owned())))
        //     ;
        // println!("\tunit from attribute = {:?}", unit);
        let position = dataset.read::<[FourVector]>()?.take();

        let dataset = file.open_dataset(&self.momentum_path)?;
        let momentum = dataset.read::<[FourVector]>()?.take();

        let polarization = file.open_dataset(&self.polarization_path)
            .and_then(|ds| ds.read::<[StokesVector]>())
            .map(|sd| sd.take())
            .ok();

        if polarization.is_none() {
            report!(Diagnostic::Warning, id == 0, "no polarization data found, continuing with unpolarized particles...");
        }

        // println!("\tgot {} momenta ([{}], ..., [{}])", momentum.len(), momentum[0].convert_from(&p_unit) / ELECTRON_MASS_MEV, momentum[momentum.len() - 1].convert_from(&p_unit) / ELECTRON_MASS_MEV);
        // println!("\tgot {} positions ([{}], ..., [{}])", position.len(), 1.0e6 * position[0].convert_from(&x_unit), 1.0e6 * position[position.len()-1].convert_from(&x_unit));
        // if let Some(ref pol) = polarization {
        //     println!("\tgot {} polarizations ([{:?}], ..., [{:?}])", pol.len(), pol[0], pol[pol.len()-1]);
        // }

        let num = weight.len().min(position.len()).min(momentum.len());
        let mut particles = Vec::with_capacity(num);

        for i in 0..num {
            let pol = if let Some(ref pol) = polarization { pol[i] } else { StokesVector::unpolarized() };
            let r = position[i].convert_from(&x_unit);
            let p = momentum[i].convert_from(&p_unit) / ELECTRON_MASS_MEV;

            let (r, p) = self.realign_particle(r, p, theta);

            let pt = Particle::create(self.species, r)
                .with_weight(weight[i])
                .with_normalized_momentum(p)
                .with_polarization(pol)
                .with_id(i as u64)
                .with_parent_id(i as u64);

            let keep_particle = {
                let p = pt.momentum();
                let n = ThreeVector::from(p).normalize();
                let n0: ThreeVector = [-self.collision_angle.sin(), 0.0, -self.collision_angle.cos()].into();
                p[0] > self.min_energy && n0 * n > self.max_angle.cos()
            };

            if keep_particle {
                particles.push(pt);
            }
        }

        // println!("\tgot {} momenta ([{}], ...)", particles.len(), particles[0].normalized_momentum());
        // println!("\tgot {} positions ([{}], ...)", particles.len(), 1.0e6 * particles[0].position());

        if id == 0 {
            println!("{} import, {} {}s per task.", "Completed".bold().bright_green(), particles.len(), self.species);
        }

        Ok(particles)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "with-mpi"))]
    extern crate no_mpi as mpi;

    use super::*;

    #[test]
    #[ignore]
    fn beam_loading() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let loader = BeamLoader::from_file("particles.h5", "photon", 1.0).with_initial_z(10.0e-6)
            .with_min_energy(1000.0);
        assert!(loader.build(&world).is_ok())
    }
}