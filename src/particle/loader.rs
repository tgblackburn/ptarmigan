//! Loading a particle beam from a binary file

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
            initial_z: 0.0,
            collision_angle: 0.0,
            offset: [0.0, 0.0, 0.0].into(),
            min_energy: 0.0,
            max_angle: std::f64::consts::PI,
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

    fn realign_particle(&self, r: FourVector, p: FourVector, old_collision_angle: f64, laser_defines_z: bool) -> (FourVector, FourVector) {
        // ignore collision plane angle for now
        let p = if laser_defines_z {
            p.rotate_around_y(self.collision_angle - old_collision_angle)
        } else {
            p.rotate_around_y(std::f64::consts::PI + self.collision_angle)
        };

        // ignore collision plane angle for now
        let r = if laser_defines_z {
            r.rotate_around_y(self.collision_angle - old_collision_angle)
        } else {
            r.rotate_around_y(std::f64::consts::PI + self.collision_angle)
        };

        // timing/alignment errors
        let r = r + self.offset.rotate_around_y(self.collision_angle).with_time(0.0);

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
    }

    pub fn build<C>(&self, comm: &C) -> Result<Vec<Particle>, OutputError> where C: Communicator {
        let id = comm.rank();
        if id == 0 {
            println!("Importing incident particle beam from '{}'...", self.filename);
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
        let laser_defines_z = file
            .open_dataset("config/output/laser_defines_positive_z")?
            .read::<bool>()?;

        let collision_angle = file
            .open_dataset("config/beam/collision_angle")?
            .read::<f64>()?;

        // introduced in v1.3.4, optionally supported
        let _collision_plane_angle = file
            .open_dataset("config/beam/collision_plane_angle")
            .and_then(|ds| ds.read::<f64>())
            .unwrap_or(0.0);

        // println!("\tlaser coordinate system? {}, collision angle = {} deg, collision plane angle = {} deg", laser_defines_z, collision_angle * (180.0 / std::f64::consts::PI), collision_plane_angle * (180.0 / std::f64::consts::PI));

        let weight = file.open_dataset(&self.weight_path)?
            .read::<[f64]>()?
            .take();

        let dataset = file.open_dataset(&self.position_path)?;
        // let unit = dataset.open_attribute("unit")?.read::<String>();
        let position = dataset.read::<[FourVector]>()?.take();

        let dataset = file.open_dataset(&self.momentum_path)?;
        // let unit = dataset.open_attribute("unit")?.read::<String>();
        let momentum = dataset.read::<[FourVector]>()?.take();

        let polarization = file.open_dataset(&self.polarization_path)
            .and_then(|ds| ds.read::<[StokesVector]>())
            .map(|sd| sd.take())
            .ok();

        if id == 0 && polarization.is_none() {
            println!("No polarization data found, continuing with unpolarized particles...")
        }

        // println!("\tgot {} momenta ([{}], ...)", momentum.len(), momentum[0].convert_from(&p_unit) / ELECTRON_MASS_MEV);
        // println!("\tgot {} positions ([{}], ...)", position.len(), 1.0e6 * position[0].convert_from(&x_unit));

        let num = weight.len().min(position.len()).min(momentum.len());
        let mut particles = Vec::with_capacity(num);

        for i in 0..num {
            let pol = if let Some(ref pol) = polarization { pol[i] } else { StokesVector::unpolarized() };
            let r = position[i].convert_from(&x_unit);
            let p = momentum[i].convert_from(&p_unit) / ELECTRON_MASS_MEV;

            let (r, p) = self.realign_particle(
                r, p, collision_angle, laser_defines_z
            );

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
            println!("Import complete, {} {}s per task.", particles.len(), self.species);
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
    fn beam_loading() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let loader = BeamLoader::from_file("particles.h5", "photon", 1.0).with_initial_z(10.0e-6)
            .with_min_energy(1000.0);
        assert!(loader.build(&world).is_ok())
    }
}