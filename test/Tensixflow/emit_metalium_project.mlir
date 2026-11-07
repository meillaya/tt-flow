// RUN: split-file %s %t
// RUN: rm -rf %t/project %t/fused_project %t/rejected %t/unplanned %t/no_emission %t/bad_mapping
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
// RUN: grep -q 'split_work_to_cores(coreGrid, kOutputTileCount)' %t/project/host.cpp
// RUN: grep -q 'CreateCircularBuffer(program, allCores' %t/project/host.cpp
// RUN: grep -q 'workPerCore' %t/project/host.cpp
// RUN: grep -q 'TENSIXFLOW_RESULT' %t/project/host.cpp
// RUN: grep -q 'Realized core grid | 8 x 8' %t/project/README.md
// RUN: grep -q 'Cores used | 4' %t/project/README.md
// RUN: grep -q 'Max tiles per core | 1' %t/project/README.md
// RUN: grep -q 'DRAM read bytes | 32768' %t/project/README.md
// RUN: grep -q 'read unicast, replicated per output tile' %t/project/README.md
// RUN: grep -q 'output-tile-ranges' %t/project/README.md
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium=output-dir=%t/fused_project %t/fused.mlir
// RUN: test -s %t/fused_project/host.cpp
// RUN: test -s %t/fused_project/kernels/compute.cpp
// RUN: grep -q 'kCbNorm = 4' %t/fused_project/host.cpp
// RUN: grep -q 'mul_tiles_bcast_cols' %t/fused_project/kernels/compute.cpp
// RUN: not tensixflow-opt --tensixflow-emit-metalium=output-dir=%t/unplanned %t/unplanned.mlir 2>&1 | FileCheck %t/unplanned.mlir
// RUN: not test -e %t/unplanned
// RUN: not tensixflow-opt --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium=output-dir=%t/no_emission %t/no_emission.mlir 2>&1 | FileCheck %t/no_emission.mlir
// RUN: not test -e %t/no_emission
// RUN: not tensixflow-opt --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium=output-dir=%t/bad_mapping %t/bad_mapping.mlir 2>&1 | FileCheck %t/bad_mapping.mlir
// RUN: not test -e %t/bad_mapping

// Project-assembler test for --tensixflow-emit-metalium=output-dir=.
//
// Happy path: the planning pipeline runs over a pure matmul and the pass writes
// all six project files, each non-empty. The CMakeLists.txt must link the real
// package (find_package(TT-Metalium REQUIRED), TT::Metalium); the host must
// reference the three kernel files it JIT-compiles, realize the plan grid with
// split_work_to_cores, and print the machine readable result line that
// scripts/run_generated.sh parses. The generated README must state the realized
// grid, the work split, the projected traffic, and that the reads are unicast
// and replicated per output tile. Compiling and running that project is the job
// of scripts/build_generated.sh and scripts/run_generated.sh, not of this lit
// test.
//
// A fused plan emits the fused project into its own tree. The failure paths
// check that a module without a schedule attribute is rejected before any file
// is written, that a module with a schedule but no emission block is rejected
// naming the planning pass, and that an emission block with an unknown mapping
// is rejected naming the one supported mapping. No failure path creates an
// output directory.

//--- matmul.mlir
tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}

//--- fused.mlir
// The fused plan emits the RMSNorm prologue in front of the matmul; the run
// lines above check the fused project tree it writes.
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

//--- no_emission.mlir
// CHECK: requires a tensixflow.emission attribute; run --tensixflow-plan-noc-sharding first
module attributes {tensixflow.schedule = #tensixflow.schedule<grid 8x8, act "row-major", weight "replicate", storage "DRAM", fusion "none", cbs [0:2:2048, 1:2:2048, 16:2:2048], dram_read 0, dram_write 0, l1 0, multicast 0, saved_read 0, saved_write 0, saved_intermediate 0>} {
  tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
    tensixflow.return %0 : tensor<64x64xbf16>
  }
}

//--- bad_mapping.mlir
// CHECK: unknown emission mapping 'mystery'
// CHECK: supports 'output-tile-ranges' only
module attributes {
  tensixflow.schedule = #tensixflow.schedule<grid 8x8, act "row-major", weight "replicate", storage "DRAM", fusion "none", cbs [0:2:2048, 1:2:2048, 16:2:2048], dram_read 0, dram_write 0, l1 0, multicast 0, saved_read 0, saved_write 0, saved_intermediate 0>,
  tensixflow.emission = #tensixflow.emission<mapping "mystery", cores_used 4, max_tiles_per_core 1, dram_read 32768, dram_write 8192, fused_stage_tiles 4, fused_feasible true>
} {
  tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
    tensixflow.return %0 : tensor<64x64xbf16>
  }
}
