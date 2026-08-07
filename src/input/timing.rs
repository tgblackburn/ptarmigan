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
