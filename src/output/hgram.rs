//! MPI-aware data binning and histogram generation

use std::{fmt, io::BufWriter};
use std::fs::File;
use std::io::Write;

/// Writer that keeps track of how many bytes it's written.
/// A single write! call will never write more than `limit` bytes.
struct WriteCounter<W: Write> {
    inner: W,
    limit: usize,
    count: usize,
}

impl<W> WriteCounter<W> where W: Write {
    fn new(inner: W, limit: usize) -> Self {
        Self {inner, limit, count: 0}
    }

    fn bytes_written(&self) -> usize {
        self.count
    }

    /// Writes the given bytes, ignoring the limit set.
    fn write_unchecked(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let res = self.inner.write(buf);
        if let Ok(count) = res {
            self.count += count;
        }
        res
    }
}

impl<W> Write for WriteCounter<W> where W: Write {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let end = buf.len().min(self.limit);
        let res = self.inner.write(&buf[..end]);
        if let Ok(count) = res {
            self.count += count;
        }
        res
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(feature = "with-mpi")]
use mpi::{traits::*, collective::SystemOperation};
#[cfg(not(feature = "with-mpi"))]
use no_mpi::*;

#[derive(Copy,Clone,PartialEq)]
pub enum BinSpec {
    Automatic,
    LogScaled,
    FixedNumber(usize),
    FixedSize(f64),
}

impl fmt::Display for BinSpec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinSpec::Automatic => write!(f, "BinSpec:Automatic"),
            BinSpec::LogScaled => write!(f, "BinSpec:LogScaled"),
            BinSpec::FixedNumber(n) => write!(f, "BinSpec:FixedNumber({})", n),
            BinSpec::FixedSize(dx) => write!(f, "BinSpec:FixedSize({})", dx),
        }
    }
}

impl From<&str> for BinSpec {
    fn from(s: &str) -> Self {
        if let Ok(nbins) = s.parse::<usize>() {
            BinSpec::FixedNumber(nbins)
        } else if let Ok(dx) = s.parse::<f64>() {
            BinSpec::FixedSize(dx)
        } else if s == "auto" {
            BinSpec::Automatic
        } else if s == "log" {
            BinSpec::LogScaled
        } else {
            BinSpec::Automatic
        }
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

#[allow(unused)]
pub struct Histogram {
    dim: usize,
    total: f64,
    unweighted_total: f64,
    bin_vol: f64,
    min: Vec<f64>,
    max: Vec<f64>,
    cts: Vec<f64>,
    bins: Vec<usize>,
    bin_sz: Vec<f64>,
    name: String,
    bunit: String,
    axis: Vec<String>,
    unit: Vec<String>,
}

impl fmt::Display for Histogram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Histogram {}d, \"{}\" [\"{}\"] {{", self.dim, self.name, self.bunit)?;
        for i in 0..self.dim {
            writeln!(f, "\taxis {}: min = {:e}, max = {:e}, {} bins of size {:e} [\"{}\"]", i+1, self.min[i], self.max[i], self.bins[i], self.bin_sz[i], self.unit[i])?;
        }
        if self.cts.len() < 5 {
            writeln!(f, "\ttotal = {:e}, cts = {:?}", self.total, self.cts)?;
        } else {
            writeln!(f, "\ttotal = {:e}, cts = [..., len = {}]", self.total, self.cts.len())?;
        }
        write!(f, "}}")
    }
}

fn min_max_by<T>(base: &[T], f: &impl Fn(&T) -> f64, filter: &impl Fn(&T) -> bool, wrapper: impl Fn(f64) -> f64) -> Option<(f64, f64)> {
    let mut min: Option<f64> = None;
    let mut max: Option<f64> = None;
    for t in base.iter() {
        let v = wrapper(f(t));
        if v.is_finite() && filter(t) {
            min = min.map_or(Some(v), |x| Some(x.min(v)));
            max = max.map_or(Some(v), |x| Some(x.max(v)));
        }
    }
    if min.is_some() && max.is_some() {
        Some((min.unwrap(), max.unwrap()))
    } else {
        None
    }
}

fn linear_bin_vol(min: f64, bin_sz: f64, bin: usize) -> f64 {
    (min + (bin as f64) * bin_sz).exp() * bin_sz.exp_m1()
}

fn number_of_bins(min: f64, max: f64, n: usize, bspec: BinSpec) -> usize {
    if min == max {
        1
    } else {
        match bspec {
            BinSpec::Automatic | BinSpec::LogScaled =>
                (2.0 * (n as f64).cbrt()).ceil() as usize,
            BinSpec::FixedNumber(n) =>
                n,
            BinSpec::FixedSize(dx) =>
                ((max - min) / dx).ceil() as usize,
        }
    }
}

fn bin_size_and_volume(dim: usize, min: &[f64], max: &[f64], nbins: &[usize], bspec: &[BinSpec]) -> (Vec<f64>, f64) {
    let mut size: Vec<f64> = Vec::new();
    let mut volume = 1.0;
    for i in 0..dim {
        if min[i] == max[i] {
            volume *= 1.0;
            size.push(0.0);
        } else {
            let dx = match bspec[i] {
                BinSpec::Automatic | BinSpec::LogScaled | BinSpec::FixedNumber(_) =>
                    (max[i] - min[i]) / (nbins[i] as f64),
                BinSpec::FixedSize(dx) =>
                    dx,
            };
            volume *= dx;
            size.push(dx);
        }
    }
    (size, volume)
}

impl Histogram {
    #[allow(unused)]
    pub fn generate_1d<T>(
        comm: &impl Communicator,
        base: &[T], accessor: &impl Fn(&T) -> f64, weight: &impl Fn(&T) -> f64,
        filter: &impl Fn(&T) -> bool,
        name: &str, unit: &str,
        bspec: BinSpec, hspec: HeightSpec) -> Option<Histogram> {
        //let rank = comm.rank();

        // Local min and max
        // Adjust for log-scaling!
        let (min, max) = if bspec == BinSpec::LogScaled {
            min_max_by(base, accessor, filter, f64::ln).unwrap_or((std::f64::MAX, -std::f64::MAX))
        } else {
            min_max_by(base, accessor, filter, std::convert::identity).unwrap_or((std::f64::MAX, -std::f64::MAX))
        };
        //let min = base.iter().map(accessor).min_by(|a,b| a.partial_cmp(b).unwrap() ).unwrap_or(std::f64::MAX);
        //let max = base.iter().map(accessor).max_by(|a,b| a.partial_cmp(b).unwrap() ).unwrap_or(-std::f64::MAX);
        //println!("{}: Local min = {:e}, max = {:e}, num = {}", name, min, max, base.len());

        //Global min and max
        let mut gmin = 0.0;
        let mut gmax = 0.0;
        let mut gnum: usize = 0;
        comm.all_reduce_into(&min, &mut gmin, SystemOperation::min());
        comm.all_reduce_into(&max, &mut gmax, SystemOperation::max());
        comm.all_reduce_into(&base.len(), &mut gnum, SystemOperation::sum());
        //println!("{}: Global min = {:e} and max = {:e}, num = {}", rank, gmin, gmax, gnum);

        if gnum == 0 {
            return None;
        }
        
        // Prep bins
        let nbins = number_of_bins(gmin, gmax, gnum, bspec);

        let bin_vol = if gmin == gmax {
            1.0
        } else {
            match bspec {
                BinSpec::Automatic | BinSpec::LogScaled | BinSpec::FixedNumber(_) =>
                    (gmax - gmin) / (nbins as f64),
                BinSpec::FixedSize(dx) =>
                    dx,
            }
        };

        //println!("{}: number of bins = {}, bin volume = {:e}", rank, nbins, bin_vol);

        // Binning
        let mut cts: Vec<f64> = vec![0.0; nbins];
        let mut total = 0.0;

        for e in base.iter() {
            let value = if bspec == BinSpec::LogScaled {
                accessor(e).ln()
            } else {
                accessor(e)
            };

            let bin = ((value - gmin) / bin_vol).floor() as usize;

            let w = weight(e);
            total = total + w; // count everything, even if not binned

            if !value.is_finite() {
                continue;
            }

            if !filter(e) {
                continue;
            }

            // adjust weight to include actual size of bin / log-scaled size
            let w = if bspec == BinSpec::LogScaled && (hspec == HeightSpec::Density || hspec == HeightSpec::ProbabilityDensity) {
                w * bin_vol / linear_bin_vol(gmin, bin_vol, bin)
            } else {w};

            // access by row-major order
            let fbin = bin;
            if fbin < cts.len() {
                cts[fbin] = cts[fbin] + w;
            }
        }

        // total weight across world
        let mut gtotal = 0.0;
        comm.all_reduce_into(&total, &mut gtotal, SystemOperation::sum());
        //println!("{}: local total = {:e}, global = {:e}", rank, total, gtotal);

        let cts = match hspec {
            HeightSpec::Count => cts,
            HeightSpec::Density => cts.iter().map(|ct| ct / bin_vol).collect(),
            HeightSpec::ProbabilityDensity => cts.iter().map(|ct| ct / (bin_vol * gtotal)).collect()
        };

        // All reduce?
        let mut gcts: Vec<f64> = vec![0.0; nbins];
        comm.all_reduce_into(&cts[..], &mut gcts[..], SystemOperation::sum());

        Some(Histogram {
            dim: 1,
            total: gtotal,
            unweighted_total: gnum as f64,
            bin_vol: bin_vol,
            min: vec![gmin],
            max: vec![gmax],
            cts: gcts,
            bins: vec![nbins],
            bin_sz: if nbins <= 1 {vec![0.0]} else {vec![bin_vol]},
            name: format!("hgram/{}/{}", hspec, name),
            bunit: format!("1/{}", unit),
            axis: vec![format!("{}", name)],
            unit: vec![unit.to_string()],
        })
    }

    #[allow(unused)]
    pub fn generate_2d<T>(
        comm: &impl Communicator,
        base: &[T], fx: &impl Fn(&T) -> f64, fy: &impl Fn(&T) -> f64, weight: &impl Fn(&T) -> f64,
        filter: &impl Fn(&T) -> bool,
        name: [&str; 2], unit: [&str; 2],
        bspec: [BinSpec; 2], hspec: HeightSpec) -> Option<Histogram> {
        //let rank = comm.rank();

        // Local min and max

        let (xmin, xmax) = if bspec[0] == BinSpec::LogScaled {
            min_max_by(base, fx, filter, f64::ln).unwrap_or((std::f64::MAX, -std::f64::MAX))
        } else {
            min_max_by(base, fx, filter, std::convert::identity).unwrap_or((std::f64::MAX, -std::f64::MAX))
        };

        let (ymin, ymax) = if bspec[1] == BinSpec::LogScaled {
            min_max_by(base, fy, filter, f64::ln).unwrap_or((std::f64::MAX, -std::f64::MAX))
        } else {
            min_max_by(base, fy, filter, std::convert::identity).unwrap_or((std::f64::MAX, -std::f64::MAX))
        };

        let min = [xmin, ymin];
        let max = [xmax, ymax];

        //Global min and max

        let mut gmin = [0.0; 2];
        let mut gmax = [0.0; 2];
        let mut gnum: usize = 0;
        comm.all_reduce_into(&min[..], &mut gmin[..], SystemOperation::min());
        comm.all_reduce_into(&max[..], &mut gmax[..], SystemOperation::max());
        comm.all_reduce_into(&base.len(), &mut gnum, SystemOperation::sum());

        if gnum == 0 {
            return None;
        }
        
        // Prep bins

        let nbins = [
            number_of_bins(gmin[0], gmax[0], gnum, bspec[0]),
            number_of_bins(gmin[1], gmax[1], gnum, bspec[1]),
        ];

        let (bin_sz, bin_vol) = bin_size_and_volume(2, &gmin, &gmax, &nbins, &bspec);

        //println!("{}: number of bins = {:?}, bin volume = {:e}", rank, nbins, bin_vol);

        // Binning
        let mut cts: Vec<f64> = vec![0.0; nbins[0] * nbins[1]];
        let mut total = 0.0;

        for e in base.iter() {
            let value = [
                if bspec[0] == BinSpec::LogScaled {fx(e).ln()} else {fx(e)},
                if bspec[1] == BinSpec::LogScaled {fy(e).ln()} else {fy(e)},
            ];

            let mut w = weight(e);
            total = total + w; // count everything, even if not binned

            if value.iter().any(|&x| !x.is_finite()) {
                continue; // all of value[i] must be finite
            }

            if !filter(e) {
                continue;
            }

            let bin = [
                if bin_sz[0] == 0.0 {0} else {((value[0] - gmin[0]) / bin_sz[0]).floor() as usize},
                if bin_sz[1] == 0.0 {0} else {((value[1] - gmin[1]) / bin_sz[1]).floor() as usize},
            ];

            // adjust weight to include actual size of bin / log-scaled size
            if bspec[0] == BinSpec::LogScaled && (hspec == HeightSpec::Density || hspec == HeightSpec::ProbabilityDensity) {
                w *= if bin_sz[0] == 0.0 {1.0} else {bin_sz[0] / linear_bin_vol(gmin[0], bin_sz[0], bin[0])};
            }

            if bspec[1] == BinSpec::LogScaled && (hspec == HeightSpec::Density || hspec == HeightSpec::ProbabilityDensity) {
                w *= if bin_sz[1] == 0.0 {1.0} else {bin_sz[1] / linear_bin_vol(gmin[1], bin_sz[1], bin[1])};
            }

            let fbin = bin[1] * nbins[1] + bin[0]; // row_index * elements_in_row + column_index
            if fbin < cts.len() {
                cts[fbin] = cts[fbin] + w;
            }
        }

        // total weight across world
        let mut gtotal = 0.0;
        comm.all_reduce_into(&total, &mut gtotal, SystemOperation::sum());
        //println!("{}: local total = {:e}, global = {:e}", rank, total, gtotal);

        let cts = match hspec {
            HeightSpec::Count => cts,
            HeightSpec::Density => cts.iter().map(|ct| ct / bin_vol).collect(),
            HeightSpec::ProbabilityDensity => cts.iter().map(|ct| ct / (bin_vol * gtotal)).collect()
        };

        // All reduce?
        let mut gcts: Vec<f64> = vec![0.0; nbins[0] * nbins[1]];
        comm.all_reduce_into(&cts[..], &mut gcts[..], SystemOperation::sum());

        Some(Histogram {
            dim: 2,
            total: gtotal,
            unweighted_total: gnum as f64,
            bin_vol: bin_vol,
            min: gmin.to_vec(),
            max: gmax.to_vec(),
            cts: gcts,
            bins: nbins.to_vec(),
            bin_sz: bin_sz,
            name: format!("hgram/{}/{}_{}", hspec, name[0], name[1]),
            bunit: format!("1/({}.{})", unit[0], unit[1]),
            axis: vec![name[0].to_owned(), name[1].to_owned()],
            unit: vec![unit[0].to_owned(), unit[1].to_owned()],
        })
    }

    /// Writes the histogram to file.
    /// The relevant extension is added to `filename`.
    pub fn write_fits(&self, filename: &str) -> std::io::Result<()> {
        let filename = filename.to_owned() + ".fits";
        let file = File::create(&filename)?;
        let file = BufWriter::new(file);
        let mut file = WriteCounter::new(file, 80);
        let naxis = self.dim;

        // Write header information
        write!(file, "SIMPLE  = {:>20} / {:<47}", 'T', "file conforms to FITS standard")?;
        write!(file, "BITPIX  = {:>20} / {:<47}", -64, "number of bits per data pixel")?;
        write!(file, "NAXIS   = {:>20} / {:<47}", naxis, "number of data axes")?;

        for n in 1..=naxis {
            write!(file, "NAXIS{:<3}= {:>20} / {:<47}", n, self.bins[n-1], "number of pixels along this axis")?;
        }

        write!(file, "EXTEND  = {:>20} / {:<47}", 'T', "dataset may contain extensions")?;

        for n in 1..=naxis {
            // 0.5 => left aligned, 1.0 => centred
            write!(file, "CRPIX{:<3}= {:>20.9E}{:<50}", n, 0.5, "")?;
            write!(file, "CRVAL{:<3}= {:>20.9E}{:<50}", n, self.min[n-1], "")?;
            write!(file, "CDELT{:<3}= {:>20.9E}{:<50}", n, self.bin_sz[n-1], "")?;
            write!(file, "CNAME{:<3}= '{}'{:<3$}", n, self.axis[n-1], "", 80 - 12 - self.axis[n-1].len())?;
            write!(file, "CUNIT{:<3}= '{}'{:<3$}", n, self.unit[n-1], "", 80 - 12 - self.unit[n-1].len())?;
        }

        write!(file, "BUNIT   = '{}'{:<2$}", self.bunit, "", 80 - 12 - self.bunit.len())?;
        write!(file, "TOTAL   = {:>20.9E}{:<50}", self.total, "")?;
        write!(file, "UWTOTAL = {:>20.9E}{:<50}", self.unweighted_total, "")?;
        write!(file, "OBJECT  = '{}'{:<2$}", self.name, "", 80 - 12 - self.name.len())?;

        if !self.cts.is_empty() {
            let mut min = self.cts[0];
            let mut max = min;
            for elem in self.cts.iter() {
                if *elem < min {
                    min = *elem;
                } else if *elem > max {
                    max = *elem;
                }
            }
            write!(file, "DATAMIN = {:>20.9E}{:<50}", min, "")?;
            write!(file, "DATAMAX = {:>20.9E}{:<50}", max, "")?;
        }

        let version = env!("CARGO_PKG_VERSION");
        let sha = env!("VERGEN_GIT_SHA_SHORT");
        write!(file, "COMMENT   Generated by Ptarmigan v{} ({:<7}){:<3$}", version, sha, "", 80 - 37 - version.len() - sha.len())?;
        write!(file, "{:80}", "END")?;

        // Header padding
        let count = file.bytes_written();
        assert_eq!(count % 80, 0);
        if count < 2880 {
            let padding = vec![b' '; 2880 - count];
            file.write_unchecked(&padding)?;
        }
        assert_eq!(file.bytes_written(), 2880);

        for elem in self.cts.iter() {
            // FITS standard requires big-endian
            let raw = elem.to_be_bytes();
            file.write_unchecked(&raw)?;
        }

        // Padding
        let count = file.bytes_written();
        let excess = count % 2880; // how far we wrote into the next block
        if excess > 0 {
            let padding = vec![0; 2880 - excess];
            file.write_unchecked(&padding)?;
        }

        Ok(())
    }

    /// Writes the histogram to file.
    /// The relevant extension is added to `filename`.
    pub fn write_plain_text(&self, filename: &str) -> std::io::Result<()> {
        let filename = format!("{}.dat", filename);
        let mut file = File::create(filename)?;

        let mut axes = self.axis.join("\t");
        axes.push('\t');
        axes.push_str(&self.name);

        let mut units = self.unit.join("\t");
        units.push('\t');
        units.push_str(&self.bunit);

        writeln!(file, "{}", axes)?;
        writeln!(file, "{}", units)?;

        let mut index = vec![0usize; self.dim];
        let mut coord = vec![0.0; self.dim];
        for ct in self.cts.iter() {
            for j in 0..(self.dim-1) {
                if index[j] >= self.bins[j] {
                    index[j] -= self.bins[j];
                    index[j+1] += 1;
                }
            }
            for j in 0..self.dim {
                coord[j] = self.min[j] + (0.5 + (index[j] as f64)) * self.bin_sz[j];
            }
            for j in 0..self.dim {
                write!(file, "{:.9e}\t", coord[j])?;
            }
            writeln!(file, "{:.9e}", ct)?;
            index[0] += 1;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;
    static INIT: Once = Once::new();

    #[cfg(feature = "with-mpi")]
    use mpi::environment::Universe;
    #[cfg(not(feature = "with-mpi"))]
    extern crate no_mpi as mpi;

    static mut UNIVERSE: Option<Universe> = None;

    #[test]
    #[ignore]
    fn single_2d() {
        // Safely init (or reinit) MPI
        let universe = unsafe {
            INIT.call_once(|| {
                UNIVERSE = mpi::initialize();
            });
            UNIVERSE.as_ref().unwrap()
        };
        let world = universe.world();

        let data = vec![[1.0, 2.0, 0.5]; 1];
        type Accessor<'a,T> = Box<dyn Fn(&T) -> f64 + 'a>;
        let fx = Box::new(|pt: &[f64; 3]| pt[0]) as Accessor<[f64; 3]>;
        let fy = Box::new(|pt: &[f64; 3]| pt[1]) as Accessor<[f64; 3]>;
        let weight = |_pt: &[f64; 3]| 1.0;
        let filter = |_pt: &[f64; 3]| true;
        let hgram = Histogram::generate_2d(&world, &data, &fx, &fy, &weight, &filter, ["x", "y"], ["1", "1"], [BinSpec::Automatic; 2], HeightSpec::Density);
        assert!(hgram.is_some());
        let hgram = hgram.unwrap();
        println!("hgram = {}", hgram);
        let status = hgram.write_fits("output/single_point");
        println!("status = {:?}", status);
        assert!(status.is_ok());
    }

    #[test]
    #[ignore]
    fn single_log_2d() {
        // Safely init (or reinit) MPI
        let universe = unsafe {
            INIT.call_once(|| {
                UNIVERSE = mpi::initialize();
            });
            UNIVERSE.as_ref().unwrap()
        };
        let world = universe.world();

        let data = vec![[1.0, 2.0, 0.5]; 1];
        type Accessor<'a,T> = Box<dyn Fn(&T) -> f64 + 'a>;
        let fx = Box::new(|pt: &[f64; 3]| pt[0]) as Accessor<[f64; 3]>;
        let fy = Box::new(|pt: &[f64; 3]| pt[1]) as Accessor<[f64; 3]>;
        let weight = |_pt: &[f64; 3]| 1.0;
        let filter = |_pt: &[f64; 3]| true;
        let hgram = Histogram::generate_2d(&world, &data, &fx, &fy, &weight, &filter, ["x", "y"], ["1", "1"], [BinSpec::LogScaled; 2], HeightSpec::Density);
        assert!(hgram.is_some());
        let hgram = hgram.unwrap();
        println!("hgram = {}", hgram);
        let status = hgram.write_fits("output/single_point_log");
        println!("status = {:?}", status);
        assert!(status.is_ok());
    }

    #[test]
    #[ignore]
    fn empty_2d() {
        // Safely init (or reinit) MPI
        let universe = unsafe {
            INIT.call_once(|| {
                UNIVERSE = mpi::initialize();
            });
            UNIVERSE.as_ref().unwrap()
        };
        let world = universe.world();

        let data: Vec<[f64; 3]> = Vec::new();
        type Accessor<'a,T> = Box<dyn Fn(&T) -> f64 + 'a>;
        let fx = Box::new(|pt: &[f64; 3]| pt[0]) as Accessor<[f64; 3]>;
        let fy = Box::new(|pt: &[f64; 3]| pt[1]) as Accessor<[f64; 3]>;
        let weight = |_pt: &[f64; 3]| 1.0;
        let filter = |_pt: &[f64; 3]| true;
        let hgram = Histogram::generate_2d(&world, &data, &fx, &fy, &weight, &filter, ["x", "y"], ["1", "1"], [BinSpec::Automatic; 2], HeightSpec::Density);
        assert!(hgram.is_none());
    }
}