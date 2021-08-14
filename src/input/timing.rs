//! Measuring and printing the runtime of the simulation

use std::fmt;

/// Estimated time to completion, based on amount of work done
pub fn ettc(start: std::time::Instant, current: usize, total: usize) -> std::time::Duration {
    let rt = start.elapsed().as_secs_f64();
    let ettc = if current < total {
        rt * ((total - current) as f64) / (current as f64)
    } else {
        0.0
    };
    std::time::Duration::from_secs_f64(ettc)
}

/// Wrapper around std::time::Duration
pub struct PrettyDuration {
    pub duration: std::time::Duration,
}

impl From<std::time::Duration> for PrettyDuration {
    fn from(duration: std::time::Duration) -> PrettyDuration {
        PrettyDuration {duration: duration}
    }
}

impl fmt::Display for PrettyDuration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut t = self.duration.as_secs();
        let s = t % 60;
        t /= 60;
        let min = t % 60;
        t /= 60;
        let hr = t % 24;
        let d = t / 24;
        if d > 0 {
            write!(f, "{}d {:02}:{:02}:{:02}", d, hr, min, s)
        } else {
            write!(f, "{:02}:{:02}:{:02}", hr, min, s)
        }
    }
}

/// Wrapper around the simulation time (in seconds)
pub struct SimulationTime(pub f64);

impl fmt::Display for SimulationTime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // find nearest SI prefix
        let power = 3.0 * ((self.0.abs().log10() + 0.0) / 3.0).floor();
        // and clip to -18 <= x <= 0
        let power = power.min(0.0f64).max(-18.0f64);
        let power = power as i32;
        let (unit, scale) = match power {
            -18 => ("as", 1.0e18),
            -15 => ("fs", 1.0e15),
            -12 => ("ps", 1.0e12),
            -9  => ("ns", 1.0e9),
            -6  => ("\u{03bc}s", 1.0e6),
            -3  => ("ms", 1.0e3),
            _   => (" s", 1.0)
        };
        write!(f, "{: >8.2} {}", scale * self.0, unit)
    }
}


mod tests {
    #[test]
    fn time_format() {
        let t = 2.6e-4_f64;
        let output = super::SimulationTime(t).to_string();
        println!("\"{}\" => \"{}\"", t, output);
        assert_eq!(output, "  260.00 \u{03bc}s");   
    }
}
