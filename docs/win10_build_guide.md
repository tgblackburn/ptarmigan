# Building Ptarmigan on Windows

Start by installing the following components from the [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/):

* MSVC v143 - VS 2022 C++ x64/x86 build tools
* Windows 11 SDK

Then get [Rust](https://www.rust-lang.org/tools/install) itself.
This will suffice to build Ptarmigan without optional features (`cargo build --release`).

## MPI

In order to run Ptarmigan in parallel, you will need to install, in addition to the above:

* [Microsoft MPI v10.1.2](https://www.microsoft.com/en-us/download/details.aspx?id=100593) (both the SDK and the redistributable)
* [LLVM v14.0.6](https://github.com/llvm/llvm-project/releases/tag/llvmorg-14.0.6)

Ptarmigan can then be built with MPI support (`cargo build --release --features with-mpi`).
