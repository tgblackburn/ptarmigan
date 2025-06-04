use vergen::{ vergen, Config };

fn main() {
    let config = Config::default();
    vergen(config).unwrap_or_else(|_| {
        println!("cargo:rustc-env=VERGEN_GIT_BRANCH=unknown");
        println!("cargo:rustc-env=VERGEN_GIT_SHA=unknown");
    });

    let mut features = vec![];
    for (k, _v) in std::env::vars() {
        match k.as_str() {
            "CARGO_FEATURE_HDF5_OUTPUT" => features.push("hdf5-output"),
            "CARGO_FEATURE_WITH_MPI" => features.push("with-mpi"),
            "CARGO_FEATURE_COMPENSATING_CHIRP" => features.push("compensating-chirp"),
            _ => {}
        }
    }
    let features = features.join(",");
    println!("cargo:rustc-env=PTARMIGAN_ACTIVE_FEATURES={}", features);
}