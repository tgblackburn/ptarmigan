//! Utilities for handling groups of particles, i.e. *showers* (a primary and its daughters) and
//! *bunches*.

use super::{ Species, Particle };

/// A shower, or cascade, consists of the primary
/// particle and all the secondaries it produces by
/// various QED processes
pub struct Shower {
    #[allow(unused)]
    pub primary: Particle,
    pub secondaries: Vec<Particle>,
    pub intermediates: Vec<Particle>,
}

pub struct ParticleBunch {
    pub electrons: Vec<Particle>,
    pub photons: Vec<Particle>,
    pub positrons: Vec<Particle>,
    pub intermediates: Vec<Particle>,
}

impl ParticleBunch {
    pub fn empty() -> Self {
        Self {
            electrons: vec![],
            photons: vec![],
            positrons: vec![],
            intermediates: vec![],
        }
    }

    pub fn append(mut self, shower: Shower) -> Self {
        let mut shower = shower;

        while let Some(pt) = shower.secondaries.pop() {
            match pt.species() {
                Species::Electron => self.electrons.push(pt),
                Species::Photon => self.photons.push(pt),
                Species::Positron => self.positrons.push(pt),
            }
        }

        self.intermediates.append(&mut shower.intermediates);
        self
    }

    /// Iterator over all electrons, photons and positrons in this bunch
    pub fn iter(&self) -> impl Iterator<Item = &Particle> {
        self.electrons.iter()
            .chain(self.photons.iter())
            .chain(self.positrons.iter())
    }

    /// Mutable iterator over all electrons, photons and positrons in this bunch
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Particle> {
        self.electrons.iter_mut()
            .chain(self.photons.iter_mut())
            .chain(self.positrons.iter_mut())
    }

    /// Mutable over all electrons, photons and positrons in this bunch,
    /// **including intermediate particles**
    pub fn iter_all_mut(&mut self) -> impl Iterator<Item = &mut Particle> {
        self.electrons.iter_mut()
            .chain(self.photons.iter_mut())
            .chain(self.positrons.iter_mut())
            .chain(self.intermediates.iter_mut())
    }
}

impl std::ops::Add for ParticleBunch {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        self.electrons.append(&mut rhs.electrons);
        self.photons.append(&mut rhs.photons);
        self.positrons.append(&mut rhs.positrons);
        self.intermediates.append(&mut rhs.intermediates);
        self
    }
}

impl std::iter::Sum for ParticleBunch {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::empty(), |a, b| a + b)
    }
}