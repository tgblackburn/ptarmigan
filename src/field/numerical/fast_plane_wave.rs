//! A numerically defined plane-wave laser pulse, for use with the LCFA

use num_complex::Complex64;

use crate::constants::*;
use crate::field::Field;
use crate::geometry::{ThreeVector, FourVector};

use super::FieldData;

/// Represents a plane-wave laser pulse, including the
/// fast oscillating carrier wave, that is defined by
/// a numerical electric field
pub struct NumericalFastPW {
    inner: FieldData,
}

impl NumericalFastPW {
    /// Returns the dominant frequency component of the pulse.
    fn omega(&self) -> f64 {
        self.inner.omega
    }
}

impl From<FieldData> for NumericalFastPW {
    fn from(laser: FieldData) -> Self {
        Self {inner: laser}
    }
}

impl From<&FieldData> for NumericalFastPW {
    fn from(laser: &FieldData) -> Self {
        Self {inner: laser.clone()}
    }
}

impl Field for NumericalFastPW {
    fn max_timestep(&self) -> Option<f64> {
        let dt = 0.5 * self.inner.step / self.omega();
        Some(dt)
    }

    fn contains(&self, r: FourVector) -> bool {
        let phase = self.omega() * (r[0] - r[3]) / SPEED_OF_LIGHT;
        phase < self.inner.end
    }

    fn ideal_initial_z(&self) -> f64 {
        -SPEED_OF_LIGHT * self.inner.start / (2.0 * self.omega())
    }

    #[allow(non_snake_case)]
    #[inline(always)]
    fn fields(&self, r: FourVector) -> (ThreeVector, ThreeVector, f64) {
        let phi = self.omega() * (r[0] - r[3]) / SPEED_OF_LIGHT;
        let index = (phi - self.inner.start) / self.inner.step;

        // Particle hasn't reached the laser pulse yet
        if index <= 0.0 {
            return ([0.0; 3].into(), [0.0; 3].into(), 0.0);
        }

        let i = index.trunc() as usize;
        let di = index.fract();

        if i >= self.inner.field.len() - 1 {
            return ([0.0; 3].into(), [0.0; 3].into(), 0.0);
        }

        // Linear interpolation between specified points
        let ex = (1.0 - di) * self.inner.field[i] + di * self.inner.field[i+1];
        let by = ex / SPEED_OF_LIGHT;

        // Transverse profile, lowest order paraxial
        // Longitudinal field components, required for focusing
        let (tp, ez, bz) = if self.inner.params.focusing {
            use std::f64::consts;
            let waist = self.inner.params.waist;
            let wavelength = 2.0 * consts::PI * SPEED_OF_LIGHT / self.omega();
            let z_r = consts::PI * waist * waist / wavelength;
            let width_sqd = 1.0 + (r[3] / z_r).powi(2);
            let rho_sqd = (r[1].powi(2) + r[2].powi(2)) / waist.powi(2);
            let tp = (-rho_sqd / width_sqd).exp() / width_sqd.sqrt();

            let ex = (1.0 - di) * self.inner.complex_field[i] + di * self.inner.complex_field[i+1];
            // ez = i ex epsilon f xi = -ex (x / z_R) / (i + z_R)
            let ez = -ex * (r[1] / z_r) / (z_r + Complex64::i());
            let ez = ez.re;
            // bz = i by epsilon f nu = -(ex/c) (y / z_R) / (i + z_R)
            let bz = -ex * (r[2] / z_r) / (z_r + Complex64::i());
            let bz = bz.re / SPEED_OF_LIGHT;

            (tp, ez, bz)
        } else {
            (1.0, 0.0, 0.0)
        };

        ( [ex * tp, 0.0, ez * tp].into(), [0.0, by * tp, bz * tp].into(), 0.0 )
    }

    fn energy(&self) -> (f64, &'static str) {
        (self.inner.energy_flux, "J/m^2")
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts;
    use crate::EquationOfMotion;
    use crate::Coordinate;
    use super::*;

    #[test]
    fn trajectory() {
        use std::fs::File;
        use std::io::Write;

        let lambda = 0.8e-6;
        let dz = 0.05e-6;
        let a0 = 100.0;
        let ex0 = 2.0 * consts::PI * ELECTRON_MASS * SPEED_OF_LIGHT_SQD * a0 / (ELEMENTARY_CHARGE * lambda);

        let field: Vec<f64> = (0..1000)
            .map(|i| {
                let z = dz * ((i as f64) - 500.0);
                let ex = ex0 * (2.0 * consts::PI * z / lambda).sin() * (-(z / 8.0e-6).powi(2)).exp();
                ex
            })
            .collect();

        let laser: NumericalFastPW = FieldData::preprocess(Coordinate::Space, dz, &field, std::f64::INFINITY).unwrap().into();

        let mut u = FourVector::new(0.0, 0.0, 0.0, -100.0).unitize();
        let z0 = laser.ideal_initial_z();
        let dt = 0.5 * laser.max_timestep().unwrap();
        let mut r = FourVector::new(-z0, 0.0, 0.0, z0);
        
        let mut u_perp_max = 0.0;
        let mut phase_max = 0.0;

        let print_trajectory = false;
        let mut file = if print_trajectory {
            File::create("output/custom_laser.dat").ok()
        } else {
            None
        };

        while laser.contains(r) {
            let (r_new, u_new, _, _) = laser.push(r, u, ELECTRON_CHARGE / ELECTRON_MASS, dt, EquationOfMotion::Lorentz);
            r = r_new;
            u = u_new;
            let u_perp = u[1].hypot(u[2]);
            let phase = laser.omega() * (r[0] - r[3]) / SPEED_OF_LIGHT;
            if u_perp > u_perp_max {
                u_perp_max = u_perp;
                phase_max = phase;
            }

            if let Some(ref mut f) = file {
                writeln!(f, "{:.6e} {:.6e} {:.6e}", phase, u[1], u[2]).unwrap();
            }
        }

        let err = (u_perp_max - a0).abs() / a0;
        println!("max u_perp = {:.3e}, occured at phi = {:.3e}, err = {:.3e}", u_perp_max, phase_max, err);
        println!("u_perp = {:.3e} {:.3e}, |u|^2 = 1 + {:.3e}", u[1], u[2], u * u - 1.0);

        assert!(err < 1.0e-2);
        assert!((u * u - 1.0).abs() < 1.0e-3);
    }
}