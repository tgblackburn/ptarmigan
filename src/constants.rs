//! Physical constants

/// Speed of light in vacuum, units of m/s
pub const SPEED_OF_LIGHT: f64 = 2.997925e8;
/// Speed of light in vacuum, squared, units of m^2/s^2
pub const SPEED_OF_LIGHT_SQD: f64 = 89875517873681764.0;
/// epsilon_0
pub const VACUUM_PERMITTIVITY: f64 = 8.854188e-12;
/// mu_0
pub const VACUUM_PERMEABILITY: f64 = 1.256637e-6;
/// Units of C
pub const ELECTRON_CHARGE: f64 = -1.602177e-19;
/// The absolute value of the electron charge, units of C
pub const ELEMENTARY_CHARGE: f64 = -ELECTRON_CHARGE;
/// Electron mass, units of kg
pub const ELECTRON_MASS: f64 = 9.109383e-31;
/// Proton mass, units of kg
pub const PROTON_MASS: f64 = 1.672622e-27;
/// Electron mass in natural units, i.e. MeV
pub const ELECTRON_MASS_MEV: f64 = 0.510999;
/// Sauter-Schwinger field, E = m^2 c^3 / (e hbar)
pub const CRITICAL_FIELD: f64 = 1.323285e18;
/// Fine-structure constant
pub const ALPHA_FINE: f64 = 7.29735257e-3;
/// Reduced Compton length / speed of light = hbar / (m c^2)
pub const COMPTON_TIME: f64 = 1.28808867e-21;
/// Classical electron radius = alpha * Compton length
pub const CLASSICAL_ELECTRON_RADIUS: f64 = 2.817940e-15;