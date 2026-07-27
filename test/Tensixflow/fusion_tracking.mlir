// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-fuse-rmsnorm-matmul %t/positive_rmsnorm_matmul.mlir | FileCheck %t/positive_rmsnorm_matmul.mlir
// RUN: tensixflow-opt --tensixflow-fuse-rmsnorm-matmul %t/positive_mlp.mlir | FileCheck %t/positive_mlp.mlir
// RUN: tensixflow-opt --tensixflow-fuse-rmsnorm-matmul %t/negative_no_fusion.mlir | FileCheck %t/negative_no_fusion.mlir

// --tensixflow-fuse-rmsnorm-matmul annotates each fused op with the
// intermediate tensors it eliminates: `tensixflow.eliminated_tensors` (count)
// and `tensixflow.eliminated_bytes` (element count x element byte width; bf16
// = 2 bytes). It also records a module-level `tensixflow.fusion_report`
// counter = the number of fused ops present after the rewrite. A module with
// no fusable pattern reports 0.
//
// Byte arithmetic (all bf16, 2 bytes/element):
//   rmsnorm+matmul: rmsnorm out 1x2048x4096 = 8388608 elems -> 16777216 bytes.
//   mlp: rmsnorm out (1x2048x4096) + matmul1/add1/gelu out (1x2048x12288 each)
//        + matmul2 out (1x2048x4096)
//        = 8388608 + 25165824 + 25165824 + 25165824 + 8388608 = 92274688 elems
//        -> 184549376 bytes.

//--- positive_rmsnorm_matmul.mlir

// A single-use rmsnorm -> matmul fuses to fused_rmsnorm_matmul, eliminating the
// one normalized intermediate (1x2048x4096 bf16 = 16777216 bytes). The discardable
// attrs print alphabetically (eliminated_bytes before eliminated_tensors).

// CHECK: module attributes {tensixflow.fusion_report = 1 : i64}
// CHECK-LABEL: tensixflow.func @single_use
// CHECK: tensixflow.fused_rmsnorm_matmul
// CHECK-SAME: {tensixflow.eliminated_bytes = 16777216 : i64, tensixflow.eliminated_tensors = 1 : i64}
// CHECK-NOT: tensixflow.rmsnorm
// CHECK-NOT: tensixflow.matmul
tensixflow.func @single_use(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

//--- positive_mlp.mlir

// The full MLP chain fuses to fused_mlp, eliminating five intermediates whose
// total size is 184549376 bytes (see the arithmetic in the header comment).

// CHECK: module attributes {tensixflow.fusion_report = 1 : i64}
// CHECK-LABEL: tensixflow.func @full_chain
// CHECK: tensixflow.fused_mlp
// CHECK-SAME: {tensixflow.eliminated_bytes = 184549376 : i64, tensixflow.eliminated_tensors = 5 : i64}
// CHECK-NOT: tensixflow.rmsnorm
// CHECK-NOT: tensixflow.matmul
// CHECK-NOT: tensixflow.gelu
// CHECK-NOT: tensixflow.add
tensixflow.func @full_chain(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w1: tensor<4096x12288xbf16>, %b1: tensor<1x2048x12288xbf16>, %w2: tensor<12288x4096xbf16>, %residual: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w1 : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %2 = tensixflow.add %1, %b1 : tensor<1x2048x12288xbf16>
  %3 = tensixflow.gelu %2 : tensor<1x2048x12288xbf16>
  %4 = tensixflow.matmul %3, %w2 : tensor<1x2048x12288xbf16> x tensor<12288x4096xbf16> -> tensor<1x2048x4096xbf16>
  %5 = tensixflow.add %4, %residual : tensor<1x2048x4096xbf16>
  tensixflow.return %5 : tensor<1x2048x4096xbf16>
}

//--- negative_no_fusion.mlir

// A lone gelu has no fusable pattern: no fused op is created and the module
// reports fusion_report = 0.

// CHECK: module attributes {tensixflow.fusion_report = 0 : i64}
// CHECK-LABEL: tensixflow.func @lone_gelu
// CHECK-NOT: tensixflow.fused_rmsnorm_matmul
// CHECK-NOT: tensixflow.fused_mlp
// CHECK: tensixflow.gelu
tensixflow.func @lone_gelu(%x: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  %0 = tensixflow.gelu %x : tensor<1x2048x4096xbf16>
  tensixflow.return %0 : tensor<1x2048x4096xbf16>
}
