// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-infer-tile-layout --tensixflow-fuse-rmsnorm-matmul %t/rmsnorm_matmul.mlir | FileCheck %t/rmsnorm_matmul.mlir
// RUN: tensixflow-opt --tensixflow-infer-tile-layout --tensixflow-fuse-rmsnorm-matmul %t/mlp_block.mlir | FileCheck %t/mlp_block.mlir

// Full fusion pipeline (--tensixflow-infer-tile-layout + --tensixflow-fuse-rmsnorm-matmul)
// on the two example graphs. infer-tile-layout annotates func args with DRAM tile
// layouts; fusion then collapses the chains into single fused ops with
// eliminated_tensors/eliminated_bytes attrs and a module-level fusion_report counter.
//
// Byte arithmetic (all bf16, 2 bytes/element):
//   rmsnorm_matmul: rmsnorm out 1x2048x4096 = 8388608 elems -> 16777216 bytes, 1 tensor.
//   mlp_block: rmsnorm out (1x2048x4096) + matmul1/add1/gelu out (1x2048x12288 each)
//              + matmul2 out (1x2048x4096)
//              = 8388608 + 25165824 + 25165824 + 25165824 + 8388608 = 92274688 elems
//              -> 184549376 bytes, 5 tensors.

//--- rmsnorm_matmul.mlir

// RMSNorm -> MatMul fuses to fused_rmsnorm_matmul. The DRAM attrs on func args
// prove infer-tile-layout ran; the eliminated attrs + fusion_report prove fusion ran.

// CHECK: module attributes {tensixflow.fusion_report = 1 : i64}
// CHECK-LABEL: tensixflow.func @rmsnorm_matmul
// CHECK-SAME: mem DRAM
// CHECK: %[[R:[a-z0-9]+]] = tensixflow.fused_rmsnorm_matmul
// CHECK-SAME: {tensixflow.eliminated_bytes = 16777216 : i64, tensixflow.eliminated_tensors = 1 : i64}
// CHECK-NOT: tensixflow.rmsnorm
// CHECK-NOT: tensixflow.matmul
// CHECK: tensixflow.return %[[R]] : tensor<1x2048x12288xbf16>
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

//--- mlp_block.mlir

// Full MLP chain fuses to fused_mlp. Five intermediates eliminated (184549376 bytes).

// CHECK: module attributes {tensixflow.fusion_report = 1 : i64}
// CHECK-LABEL: tensixflow.func @mlp_block
// CHECK-SAME: mem DRAM
// CHECK: %[[R:[a-z0-9]+]] = tensixflow.fused_mlp
// CHECK-SAME: {tensixflow.eliminated_bytes = 184549376 : i64, tensixflow.eliminated_tensors = 5 : i64}
// CHECK-NOT: tensixflow.rmsnorm
// CHECK-NOT: tensixflow.matmul
// CHECK-NOT: tensixflow.gelu
// CHECK-NOT: tensixflow.add
// CHECK: tensixflow.return %[[R]] : tensor<1x2048x4096xbf16>
tensixflow.func @mlp_block(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w1: tensor<4096x12288xbf16>, %b1: tensor<1x2048x12288xbf16>, %w2: tensor<12288x4096xbf16>, %residual: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w1 : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %2 = tensixflow.add %1, %b1 : tensor<1x2048x12288xbf16>
  %3 = tensixflow.gelu %2 : tensor<1x2048x12288xbf16>
  %4 = tensixflow.matmul %3, %w2 : tensor<1x2048x12288xbf16> x tensor<12288x4096xbf16> -> tensor<1x2048x4096xbf16>
  %5 = tensixflow.add %4, %residual : tensor<1x2048x4096xbf16>
  tensixflow.return %5 : tensor<1x2048x4096xbf16>
}
