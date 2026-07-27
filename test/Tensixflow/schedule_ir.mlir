// RUN: split-file %s %t
// RUN: tensixflow-opt %t/positive.mlir | FileCheck %t/positive.mlir
// RUN: tensixflow-opt -verify-diagnostics %t/negative.mlir

//--- positive.mlir

// A full reader/compute/writer tile program for a fused rmsnorm+matmul kernel:
// the reader stages input tiles into in0 (c_0) over the NoC, the compute program
// runs tile rmsnorm + matmul (+ gelu epilogue) over the CBs, and the writer drains
// the out CB (c_16). CB indices follow the tt-metal c_0/c_1/c_16 convention.

// CHECK-LABEL: tensixflow.sched.kernel @fused_rmsnorm_matmul
tensixflow.sched.kernel @fused_rmsnorm_matmul {
  // CHECK: tensixflow.sched.reader {
  // CHECK: tensixflow.sched.tile_loop 64 {
  // CHECK: tensixflow.sched.cb_reserve_back 0, 1
  // CHECK: tensixflow.sched.noc_read 12345, 0, 1
  // CHECK: tensixflow.sched.cb_push_back 0, 1
  tensixflow.sched.reader {
    tensixflow.sched.tile_loop 64 {
      tensixflow.sched.cb_reserve_back 0, 1
      tensixflow.sched.noc_read 12345, 0, 1
      tensixflow.sched.cb_push_back 0, 1
    }
  }
  // CHECK: tensixflow.sched.compute {
  // CHECK: tensixflow.sched.tile_loop 64 {
  // CHECK: tensixflow.sched.cb_wait_front 0, 1
  // CHECK: tensixflow.sched.rmsnorm 0, 2, 0
  // CHECK: tensixflow.sched.matmul_tiles 0, 1, 16
  // CHECK: tensixflow.sched.gelu 16, 16
  // CHECK: tensixflow.sched.cb_pop_front 0, 1
  tensixflow.sched.compute {
    tensixflow.sched.tile_loop 64 {
      tensixflow.sched.cb_wait_front 0, 1
      tensixflow.sched.rmsnorm 0, 2, 0
      tensixflow.sched.matmul_tiles 0, 1, 16
      tensixflow.sched.gelu 16, 16
      tensixflow.sched.cb_pop_front 0, 1
    }
  }
  // CHECK: tensixflow.sched.writer {
  // CHECK: tensixflow.sched.tile_loop 64 {
  // CHECK: tensixflow.sched.cb_wait_front 16, 1
  // CHECK: tensixflow.sched.noc_write 16, 67890, 1
  // CHECK: tensixflow.sched.noc_multicast 16, 99999, 1, 8
  // CHECK: tensixflow.sched.cb_pop_front 16, 1
  tensixflow.sched.writer {
    tensixflow.sched.tile_loop 64 {
      tensixflow.sched.cb_wait_front 16, 1
      tensixflow.sched.noc_write 16, 67890, 1
      tensixflow.sched.noc_multicast 16, 99999, 1, 8
      tensixflow.sched.cb_pop_front 16, 1
    }
  }
}

//--- negative.mlir

// An ill-formed tile loop: the trip count is negative. The verifier must reject
// it. Isolated in its own split-file section because the module verifier
// short-circuits on the first failing op (todo-12 learning).
tensixflow.sched.kernel @bad_trip_count {
  tensixflow.sched.reader {
    // expected-error @below {{trip count must be non-negative}}
    tensixflow.sched.tile_loop -1 {
      tensixflow.sched.cb_reserve_back 0, 1
    }
  }
}
