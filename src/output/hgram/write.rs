//! Helper routines for writing to FITS files

use std::io::Write;

/// Writer that keeps track of how many bytes it's written.
/// A single write! call will never write more than `limit` bytes.
pub struct WriteCounter<W: Write> {
    inner: W,
    limit: usize,
    count: usize,
}

impl<W> WriteCounter<W> where W: Write {
    pub fn new(inner: W, limit: usize) -> Self {
        Self {inner, limit, count: 0}
    }

    pub fn bytes_written(&self) -> usize {
        self.count
    }

    /// Writes the given bytes, ignoring the limit set.
    pub fn write_unchecked(&mut self, buf: &[u8]) -> std::io::Result<usize> {
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