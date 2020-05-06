//! Implements the factorial

use std::sync::Once;

pub trait Factorial {
    /// Evaluates the factorial function n!
    fn factorial(&self) -> f64;
}

impl Factorial for i32 {
    fn factorial(&self) -> f64 {
        get_fcache()[*self as usize]
    }
}

const CACHE_SIZE: usize = 171;
static mut FCACHE: &'static mut [f64; CACHE_SIZE] = &mut [1.0; CACHE_SIZE];
static START: Once = Once::new();

fn get_fcache() -> &'static [f64; CACHE_SIZE] {
    unsafe {
        START.call_once(|| {
            (1..CACHE_SIZE).fold(FCACHE[0], |acc, i| {
                let fac = acc * i as f64;
                FCACHE[i] = fac;
                fac
            });
        });
        FCACHE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorial() {
        assert!(4i32.factorial() == 24.0);
        assert!(18i32.factorial() == 6402373705728000.0);
    }
}