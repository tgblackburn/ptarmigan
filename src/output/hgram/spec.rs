//! Bin and height specifications

use std::fmt;

#[derive(Copy,Clone,PartialEq)]
pub enum BinSpec {
    Automatic,
    LogScaled,
    FixedNumberLog(usize),
    FixedNumber(usize),
    FixedSize(f64),
}

impl fmt::Display for BinSpec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinSpec::Automatic => write!(f, "BinSpec:Automatic"),
            BinSpec::LogScaled => write!(f, "BinSpec:LogScaled"),
            BinSpec::FixedNumberLog(n) => write!(f, "BinSpec:FixedNumberLog({})", n),
            BinSpec::FixedNumber(n) => write!(f, "BinSpec:FixedNumber({})", n),
            BinSpec::FixedSize(dx) => write!(f, "BinSpec:FixedSize({})", dx),
        }
    }
}

impl From<&str> for BinSpec {
    fn from(s: &str) -> Self {
        let s: String = s.split_whitespace().collect();
        if let Ok(nbins) = s.parse::<usize>() {
            BinSpec::FixedNumber(nbins)
        } else if let Ok(dx) = s.parse::<f64>() {
            BinSpec::FixedSize(dx)
        } else if s == "auto" {
            BinSpec::Automatic
        } else if s == "log" {
            BinSpec::LogScaled
        } else if let Some(n) = s.strip_prefix("log&").and_then(|s| s.parse::<usize>().ok()) {
            BinSpec::FixedNumberLog(n)
        } else {
            BinSpec::Automatic
        }
    }
}

impl BinSpec {
    pub fn is_log_scaled(&self) -> bool {
        matches!(self, BinSpec::LogScaled | BinSpec::FixedNumberLog(_))
    }
}

#[derive(Copy,Clone,PartialEq)]
pub enum HeightSpec {
    Count,
    Density,
    ProbabilityDensity,
}

impl From<&str> for HeightSpec {
    fn from(s: &str) -> Self {
        match s {
            "count" => HeightSpec::Count,
            "density" | "auto" => HeightSpec::Density,
            "probablity_density" | "pdf" => HeightSpec::ProbabilityDensity,
            _ => HeightSpec::Density,
        }
    }
}

impl fmt::Display for HeightSpec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            HeightSpec::Count => write!(f, "count"),
            HeightSpec::Density => write!(f, "density"),
            HeightSpec::ProbabilityDensity => write!(f, "pdf"),
        }
    }

}