//! Use if no MPI implementation is available to replace the relevant
//! with no-ops or straightforward copies, as if MPI_COMM_SELF were
//! being used

pub trait AsSlice {
    fn copy_from(&mut self, src: &Self);
}

impl<T: Copy> AsSlice for T {
    fn copy_from(&mut self, src: &Self) {
        *self = *src;
    }
}

impl<T: Copy> AsSlice for [T] {
    fn copy_from(&mut self, src: &Self) {
        self.as_mut().copy_from_slice(src.as_ref())
    }
}

pub trait Operation {}

#[allow(non_camel_case_types)]
pub enum SystemOperation {
    min(),
    max(),
    sum(),
    logical_and(),
}

impl Operation for SystemOperation {}

pub trait Communicator {
    fn all_reduce_into<S: AsSlice + ?Sized, O: Operation>(&self, send: &S, recv: &mut S, _op: O) {
        recv.copy_from(send);
    }

    fn all_gather_into<T: Copy>(&self, send: &T, recv: &mut [T]) {
        recv[0] = *send;
    }

    fn rank(&self) -> i32 {
        0
    }

    fn size(&self) -> i32 {
        1
    }
}

pub struct SingleTask {}
impl Communicator for SingleTask {}

pub struct Universe {}

impl Universe {
    pub fn world(&self) -> SingleTask {
        SingleTask {}
    }
}

pub fn initialize() -> Option<Universe> {
    Some(Universe {})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_reduce() {
        let comm = SingleTask {};

        let a = 2.0;
        let mut b = 0.0;
    
        comm.all_reduce_into(&a, &mut b, SystemOperation::min());
        println!("a = {}, b = {}", a, b);
        assert_eq!(a, b);
    
        let a: usize = 7;
        let mut b: usize = 0;
    
        comm.all_reduce_into(&a, &mut b, SystemOperation::max());
        println!("a = {}, b = {}", a, b);
        assert_eq!(a, b);
    
        let a = [0.0, 1.0, 3.0];
        let mut b = [0.0; 3];
    
        comm.all_reduce_into(&a[..], &mut b[..], SystemOperation::sum());
        println!("a = {:?}, b = {:?}", a, b);
        assert!(a[0] == b[0] && a[1] == b[1] && a[2] == b[2]);
    }
}

