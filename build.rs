use vergen::*;

fn main() {
    let flags = ConstantsFlags::all();
    gen(flags).unwrap_or_else(|_e| {
        println!("cargo:rustc-env=VERGEN_GIT_BRANCH=unknown");
        println!("cargo:rustc-env=VERGEN_GIT_SHA=unknown");
    });

    let mut features = vec![];
    for (k, _v) in std::env::vars() {
        match k.as_str() {
            "CARGO_FEATURE_FITS_OUTPUT" => features.push("fits-output"),
            "CARGO_FEATURE_HDF5_OUTPUT" => features.push("hdf5-output"),
            "CARGO_FEATURE_WITH_MPI" => features.push("with-mpi"),
            "CARGO_FEATURE_COMPENSATING_CHIRP" => features.push("compensating-chirp"),
            "CARGO_FEATURE_NO_RADIATION_REACTION" => features.push("no-radiation-reaction"),
            "CARGO_FEATURE_COS2_ENVELOPE_IN_3D" => features.push("cos2-envelope-in-3d"),
            _ => {}
        }
    }
    let features = features.join(",");
    println!("cargo:rustc-env=PTARMIGAN_ACTIVE_FEATURES={}", features);
}