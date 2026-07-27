{ pkgs, ... }:
let
  # Explicit versioned LLVM set — MUST NOT use the floating default `llvmPackages`.
  llvm = pkgs.llvmPackages_20;
in
{
  languages.c.enable = true;

  packages = [
    llvm.clang
    llvm.mlir
    llvm.mlir.dev
    llvm.libllvm.dev
    llvm.tblgen
    pkgs.cmake
    pkgs.ninja
    pkgs.python3
    pkgs.lit
  ];

  env.LLVM_DIR = "${llvm.libllvm.dev}/lib/cmake/llvm";
  env.MLIR_DIR = "${llvm.mlir.dev}/lib/cmake/mlir";

  enterShell = ''
    export PATH="${llvm.tblgen}/bin:$PATH"
  '';
}
