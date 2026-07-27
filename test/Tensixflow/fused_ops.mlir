// RUN: split-file %s %t
// RUN: tensixflow-opt %t/positive.mlir | FileCheck %t/positive.mlir
// RUN: tensixflow-opt -verify-diagnostics %t/negative_k_mismatch.mlir
// RUN: tensixflow-opt -verify-diagnostics %t/negative_gamma.mlir

//--- positive.mlir

// CHECK-LABEL: tensixflow.func @fused_rmsnorm_matmul
// CHECK-SAME: (%[[X:.*]]: tensor<1x2048x4096xbf16>, %[[G:.*]]: tensor<4096xbf16>, %[[W:.*]]: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16>
tensixflow.func @fused_rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  // CHECK: %[[R:.*]] = tensixflow.fused_rmsnorm_matmul %[[X]], %[[G]], %[[W]] : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %0 = tensixflow.fused_rmsnorm_matmul %x, %gamma, %w : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  // CHECK: tensixflow.return %[[R]] : tensor<1x2048x12288xbf16>
  tensixflow.return %0 : tensor<1x2048x12288xbf16>
}

// CHECK-LABEL: tensixflow.func @fused_mlp
// CHECK-SAME: (%[[MX:.*]]: tensor<1x2048x4096xbf16>, %[[MG:.*]]: tensor<4096xbf16>, %[[W1:.*]]: tensor<4096x11008xbf16>, %[[B1:.*]]: tensor<11008xbf16>, %[[W2:.*]]: tensor<11008x4096xbf16>, %[[RES:.*]]: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
tensixflow.func @fused_mlp(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w1: tensor<4096x11008xbf16>, %b1: tensor<11008xbf16>, %w2: tensor<11008x4096xbf16>, %residual: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  // CHECK: %[[MR:.*]] = tensixflow.fused_mlp %[[MX]], %[[MG]], %[[W1]], %[[B1]], %[[W2]], %[[RES]] : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x11008xbf16>, tensor<11008xbf16>, tensor<11008x4096xbf16>, tensor<1x2048x4096xbf16> -> tensor<1x2048x4096xbf16>
  %0 = tensixflow.fused_mlp %x, %gamma, %w1, %b1, %w2, %residual : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x11008xbf16>, tensor<11008xbf16>, tensor<11008x4096xbf16>, tensor<1x2048x4096xbf16> -> tensor<1x2048x4096xbf16>
  // CHECK: tensixflow.return %[[MR]] : tensor<1x2048x4096xbf16>
  tensixflow.return %0 : tensor<1x2048x4096xbf16>
}

//--- negative_k_mismatch.mlir

// Mismatched contracting dimension: input K = 4096 (last dim) but weight K =
// 2048 (second-to-last dim). The verifier must reject this.
tensixflow.func @fused_rm_k_mismatch(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<2048x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  // expected-error @below {{mismatched contracting dimensions: input K = 4096 but weight K = 2048}}
  %0 = tensixflow.fused_rmsnorm_matmul %x, %gamma, %w : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<2048x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %0 : tensor<1x2048x12288xbf16>
}

//--- negative_gamma.mlir

// Gamma size mismatch: gamma is tensor<2048xbf16> but the input's last dim is
// 4096. The verifier must reject this.
tensixflow.func @fused_rm_gamma_mismatch(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<2048xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  // expected-error @below {{gamma size (2048) must match the input's last dimension (4096)}}
  %0 = tensixflow.fused_rmsnorm_matmul %x, %gamma, %w : tensor<1x2048x4096xbf16>, tensor<2048xbf16>, tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %0 : tensor<1x2048x12288xbf16>
}
