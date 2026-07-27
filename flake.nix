{
  description = "TensixFlow — nix dev environment for an out-of-tree LLVM/MLIR toolchain (LLVM 20)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # Explicit versioned LLVM set — MUST NOT use the floating default `llvmPackages`.
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          llvm = pkgs.llvmPackages_20;
        in
        {
          default = pkgs.mkShell {
            packages = [
              llvm.clang
              llvm.llvm
              llvm.bintools
              llvm.mlir
              llvm.mlir.dev
              llvm.libllvm.dev
              llvm.tblgen
              pkgs.cmake
              pkgs.ninja
              pkgs.python3
              pkgs.lit
              # GoogleTest for the C++ unit-test suite (todo 24+). `gtest.dev`
              # carries the headers + lib/cmake/GTest config and propagates the
              # `out` shared libs; `gtest` (out) is listed explicitly so the
              # runtime .so dir is on the link/runtime path regardless of
              # propagation. outputsToInstall for gtest is only "out", so the
              # dev output MUST be named explicitly for find_package(GTest).
              pkgs.gtest
              pkgs.gtest.dev
            ];

            shellHook = ''
              export LLVM_DIR="${llvm.libllvm.dev}/lib/cmake/llvm"
              export MLIR_DIR="${llvm.mlir.dev}/lib/cmake/mlir"
              export PATH="${llvm.tblgen}/bin:$PATH"
            '';
          };
        });
    };
}
