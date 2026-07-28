// RUN: split-file %s %t
// RUN: rm -rf %t/project %t/bad
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium=output-dir=%t/project %t/happy.mlir
// RUN: test -s %t/project/host.cpp
// RUN: test -s %t/project/reader.cpp
// RUN: test -s %t/project/compute.cpp
// RUN: test -s %t/project/writer.cpp
// RUN: test -s %t/project/CMakeLists.txt
// RUN: test -s %t/project/README.md
// RUN: %cxx -std=c++20 -fsyntax-only -Wall -I %S/../../kernels/handwritten_baselines/tt_metal_stubs %t/project/host.cpp
// RUN: %cxx -std=c++20 -fsyntax-only -Wall -I %S/../../kernels/handwritten_baselines/tt_metal_stubs %t/project/reader.cpp
// RUN: %cxx -std=c++20 -fsyntax-only -Wall -I %S/../../kernels/handwritten_baselines/tt_metal_stubs %t/project/compute.cpp
// RUN: %cxx -std=c++20 -fsyntax-only -Wall -I %S/../../kernels/handwritten_baselines/tt_metal_stubs %t/project/writer.cpp
// RUN: not tensixflow-opt --tensixflow-emit-metalium=output-dir=%t/bad %t/no_schedule.mlir 2>&1 | FileCheck %t/no_schedule.mlir
// RUN: not test -e %t/bad

// Project-assembler test for --tensixflow-emit-metalium=output-dir= (todo 37).
//
// Happy path: the full plan -> allocate -> lower -> emit pipeline runs with the
// output-dir option over an unfused rmsnorm -> matmul pair (planned as
// fused_rmsnorm_matmul). The pass must write ALL SIX project files, each
// non-empty (`test -s`): host.cpp, reader.cpp, compute.cpp, writer.cpp,
// CMakeLists.txt, README.md. Then all four kernels must compile `-fsyntax-only`
// against the signature-only stubs (%cxx is CMAKE_CXX_COMPILER, wired through
// lit.site.cfg.py; the stubs path is relative to this test's source dir).
//
// Failure path: running the emit pass ALONE (no plan passes) over an unplanned
// module must fail naming the missing schedule attribute AND write nothing --
// `not test -e %t/bad` asserts the output directory is never even created, so a
// validation failure can never leave a half-written project tree.
//
// The option's CLI form is `--tensixflow-emit-metalium=output-dir=<dir>` (MLIR
// 20.1.8 parses pass options as `--<pass>=<option>=<value>`; a bare `-o` is
// mlir-opt's own output-file flag). With the option UNSET, the pass keeps its
// stdout + delimiter-banner behavior (covered by emit_metalium_host.mlir,
// emit_metalium_reader_writer.mlir, emit_metalium_compute.mlir).

//--- happy.mlir
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

//--- no_schedule.mlir
// CHECK: requires a tensixflow.schedule attribute
tensixflow.func @unplanned(%x: tensor<2048x4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<2048x12288xbf16> {
  %0 = tensixflow.matmul %x, %w : tensor<2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<2048x12288xbf16>
  tensixflow.return %0 : tensor<2048x12288xbf16>
}
