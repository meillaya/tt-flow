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
