//! Input parsing errors

use std::fmt;
use std::error::Error;

/// Why did Config::read fail?
#[derive(Debug,Copy,Clone,PartialEq)]
pub enum InputErrorKind {
    File,
    Location,
    Conversion,
}

/// Error returned when Config::read fails.
pub struct InputError {
    kind: InputErrorKind,
    #[allow(unused)]
    path: String,
    cause: String,
}

impl fmt::Debug for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let help_msg = "Usage: mpirun -n np ./ptarmigan input-file";
        match self.kind {
            InputErrorKind::File => write!(f, "unable to open configuration file.\n       {}.\n{}", self.cause, help_msg),
            InputErrorKind::Location => write!(f, "failed to follow specified path \"{}\": component \"{}\" is missing.", self.path, self.cause),
            InputErrorKind::Conversion => write!(f, "could not convert field \"{}\" to target type.", self.cause),
        }
    }
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for InputError {}

impl InputError {
    pub fn file(cause: &str) -> Self {
        Self {
            kind: InputErrorKind::File,
            path: String::new(),
            cause: cause.to_owned(),
        }
    }

    pub fn location(path: &str, cause: &str) -> Self {
        Self {
            kind: InputErrorKind::Location,
            path: path.to_owned(),
            cause: cause.to_owned(),
        }
    }

    pub fn conversion(path: &str, cause: &str) -> Self {
        Self {
            kind: InputErrorKind::Conversion,
            path: path.to_owned(),
            cause: cause.to_owned(),
        }
    }

    pub fn kind(&self) -> InputErrorKind {
        self.kind
    }
}