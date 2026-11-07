// RUN: split-file %s %t
// RUN: rm -rf %t/project %t/rejected %t/unplanned
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium=output-dir=%t/project %t/matmul.mlir
// RUN: test -s %t/project/host.cpp
// RUN: test -s %t/project/kernels/reader.cpp
// RUN: test -s %t/project/kernels/compute.cpp
// RUN: test -s %t/project/kernels/writer.cpp
// RUN: test -s %t/project/CMakeLists.txt
// RUN: test -s %t/project/README.md
// RUN: grep -q 'find_package(TT-Metalium REQUIRED)' %t/project/CMakeLists.txt
// RUN: grep -q 'TT::Metalium' %t/project/CMakeLists.txt
// RUN: grep -q 'kernels/reader.cpp' %t/project/host.cpp
// RUN: grep -q 'kernels/compute.cpp' %t/project/host.cpp
// RUN: grep -q 'kernels/writer.cpp' %t/project/host.cpp
// RUN: grep -q 'TENSIXFLOW_RESULT' %t/project/host.cpp
// RUN: not tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium=output-dir=%t/rejected %t/fused.mlir 2>&1 | FileCheck %t/fused.mlir
// RUN: not test -e %t/rejected
// RUN: not tensixflow-opt --tensixflow-emit-metalium=output-dir=%t/unplanned %t/unplanned.mlir 2>&1 | FileCheck %t/unplanned.mlir
// RUN: not test -e %t/unplanned

// Project-assembler test for --tensixflow-emit-metalium=output-dir=.
//
// Happy path: the planning pipeline runs over a pure matmul and the pass writes
// all six project files, each non-empty. The CMakeLists.txt must link the real
// package (find_package(TT-Metalium REQUIRED), TT::Metalium); the host must
// reference the three kernel files it JIT-compiles and print the machine
// readable result line that scripts/run_generated.sh parses. Compiling and
// linking that project is the job of scripts/build_generated.sh, not of this
// lit test.
//
// Failure paths: a fused plan is rejected with the actionable message and no
// output directory; a module without a schedule attribute is rejected before any
// file is written.

//--- matmul.mlir
tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}

//--- fused.mlir
// CHECK: cannot emit a TT-Metalium program for the fused plan 'fused_rmsnorm_matmul'
// CHECK: pure matmul path only
// CHECK: tensixflow-report
tensixflow.func @rmsnorm_matmul(%x: tensor<64x64xbf16>, %gamma: tensor<64xbf16>, %w: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<64x64xbf16>, tensor<64xbf16> -> tensor<64x64xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %1 : tensor<64x64xbf16>
}

//--- unplanned.mlir
// CHECK: requires a tensixflow.schedule attribute
tensixflow.func @unplanned(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}
