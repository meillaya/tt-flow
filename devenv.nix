# The single source of truth for this project's environment.
#
# Everything the repository needs to configure, build, test, run, and render
# figures is declared here: the pinned LLVM/MLIR toolchain, the Python used by
# the tests and generators, the pinned GoogleTest for the C++ suites, the
# system libraries that the pinned TT-Metalium release links against, and the
# shell variables the CMake build reads. Do not add a second provisioning path
# (a flake, a shell.nix, or hand-installed packages); add it to this file.
{ pkgs, ... }:
let
  # Versioned LLVM set. The project is built against MLIR 20 and the CMake
  # config files below must match the compiler the tables are generated with.
  llvm = pkgs.llvmPackages_20;

  # Python for the golden, generator, and pytest suites.
  python = pkgs.python3.withPackages (ps: [ ps.numpy ps.pytest ]);

  # Tenstorrent's release debs declare these runtime dependencies: MPI,
  # hwloc, numactl, and zlib. They are normal packages on a Nix host, so the
  # shell provides them and points the loader and the linker at them.
  ttMetalSystemLibs = with pkgs; [ openmpi hwloc numactl zlib ];

  # The headers and CMake package for the pinned runtime come from the deb
  # tree under .deps, which scripts/fetch_tt_metal.sh extracts. The shell only
  # provides the build-time dependencies that tree expects to find installed.
  cmakeSearchDirs = with pkgs; [
    "${nlohmann_json}"
    "${(nlohmann_json.dev or nlohmann_json)}/lib/cmake"
    "${openmpi}"
    "${(openmpi.dev or openmpi)}/lib/cmake"
    "${hwloc}"
    "${(hwloc.dev or hwloc)}/lib/cmake"
  ];

  linkerSearchDirs = with pkgs; [
    "${openmpi}/lib"
    "${hwloc}/lib"
    "${numactl}/lib"
    "${zlib}/lib"
    "${stdenv.cc.cc.lib}/lib"
  ];
in
{
  packages = [
    llvm.clang
    llvm.llvm
    llvm.bintools
    llvm.mlir
    llvm.mlir.dev
    llvm.libllvm.dev
    llvm.tblgen
    pkgs.lit
    pkgs.cmake
    pkgs.ninja
    pkgs.bash

    # C++ unit suites.
    pkgs.gtest
    pkgs.gtest.dev

    python

    # The tt-mlir Python wheel is built for CPython 3.11 and is the only
    # released binary of the compiler, so keep an interpreter for it.
    pkgs.python311

    # Dependency fetch and project scripts.
    pkgs.curl
    pkgs.gnutar
    pkgs.zstd
    pkgs.binutils
    pkgs.gzip

    # Build-time dependency of the pinned TT-Metalium headers.
    pkgs.nlohmann_json
  ] ++ ttMetalSystemLibs;

  env.LLVM_DIR = "${llvm.libllvm.dev}/lib/cmake/llvm";
  env.MLIR_DIR = "${llvm.mlir.dev}/lib/cmake/mlir";
  env.CMAKE_PREFIX_PATH = pkgs.lib.concatStringsSep ":" cmakeSearchDirs;
  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
    ttMetalSystemLibs ++ [ pkgs.stdenv.cc.cc.lib ]
  );

  # Link flags for the generated project. The pinned Ubuntu binaries leave
  # MPI, hwloc, numa, and zlib symbols to the executable, and the rpath keeps
  # the runtime loader on the same paths.
  env.TT_METAL_SYSTEM_LINK_FLAGS = pkgs.lib.concatStringsSep " " (
    map (dir: "-L${dir}") linkerSearchDirs
    ++ map (dir: "-Wl,-rpath,${dir}") linkerSearchDirs
    ++ [ "-lmpi" "-lhwloc" "-lnuma" "-lz" ]
  );

  enterShell = ''
    export PATH="${llvm.tblgen}/bin:${python}/bin:$PATH"

    # Use the pinned clang, not the ambient gcc, for the C and C++ builds.
    export CC="${llvm.clang}/bin/clang"
    export CXX="${llvm.clang}/bin/clang++"

    # LLVM's lit target runs LLVM_EXTERNAL_LIT through a wrapper that assumes a
    # Python script. The nix `lit` on PATH is a bash wrapper, so point the build
    # at the wrapped Python entry point that ships beside it.
    export LLVM_EXTERNAL_LIT="$(dirname "$(command -v lit)")/.lit-wrapped"
  '';

  # Figure rendering is authoring work, so chromium is referenced here but kept
  # out of the default package list. The scene generator and the renderer use
  # the excalidraw CLI installed on the host; only the browser is provisioned.
  # The wrapper gives the browser a writable scratch HOME, which a restricted
  # CI container needs before Chromium will render at all.
  scripts.render-figures.exec = ''
    CHROMIUM_BIN=${pkgs.chromium}/bin/chromium \
    EXCALIDRAW_BROWSER_PATH="$PWD/scripts/chromium_sandbox_wrapper.sh" \
      bash scripts/render_figures.sh
  '';
}
