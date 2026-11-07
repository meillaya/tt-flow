// RUN: split-file %s %t
// RUN: tensixflow-opt %t/positive.mlir | FileCheck %t/positive.mlir
// RUN: tensixflow-opt -verify-diagnostics %t/negative_matmul.mlir
// RUN: tensixflow-opt -verify-diagnostics %t/negative_rmsnorm.mlir

//--- positive.mlir

// A realistic fused transformer flow: rmsnorm -> matmul -> gelu -> add.
// Each op's result type is inferred; the CHECK lines assert the exact
// inferred types rather than only the presence of the op.

// CHECK-LABEL: tensixflow.func @fused_flow
// CHECK-SAME: (%[[X:.*]]: tensor<1x2048x4096xbf16>, %[[W:.*]]: tensor<4096x12288xbf16>, %[[G:.*]]: tensor<4096xbf16>, %[[RES:.*]]: tensor<1x2048x12288xbf16>) -> tensor<1x2048x12288xbf16>
tensixflow.func @fused_flow(%x: tensor<1x2048x4096xbf16>, %w: tensor<4096x12288xbf16>,
                            %gamma: tensor<4096xbf16>, %res: tensor<1x2048x12288xbf16>)
    -> tensor<1x2048x12288xbf16> {
  // rmsnorm: result == input shape [1,2048,4096]
  // CHECK: %[[R0:.*]] = tensixflow.rmsnorm %[[X]], %[[G]] : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  // matmul: [1,2048,4096] x [4096,12288] -> [1,2048,12288]
  // CHECK: %[[R1:.*]] = tensixflow.matmul %[[R0]], %[[W]] : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  // gelu: result == input shape [1,2048,12288]
  // CHECK: %[[R2:.*]] = tensixflow.gelu %[[R1]] : tensor<1x2048x12288xbf16>
  %2 = tensixflow.gelu %1 : tensor<1x2048x12288xbf16>
  // add: result == operands shape [1,2048,12288]
  // CHECK: %[[R3:.*]] = tensixflow.add %[[R2]], %[[RES]] : tensor<1x2048x12288xbf16>
  %3 = tensixflow.add %2, %res : tensor<1x2048x12288xbf16>
  // CHECK: tensixflow.return %[[R3]] : tensor<1x2048x12288xbf16>
  tensixflow.return %3 : tensor<1x2048x12288xbf16>
}

// Rank-2 matmul inference path: [2048,4096] x [4096,12288] -> [2048,12288]

// CHECK-LABEL: tensixflow.func @matmul_rank2
// CHECK-SAME: (%[[X2:.*]]: tensor<2048x4096xbf16>, %[[W2:.*]]: tensor<4096x12288xbf16>) -> tensor<2048x12288xbf16>
tensixflow.func @matmul_rank2(%x: tensor<2048x4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<2048x12288xbf16> {
  // CHECK: %[[R:.*]] = tensixflow.matmul %[[X2]], %[[W2]] : tensor<2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<2048x12288xbf16>
  %0 = tensixflow.matmul %x, %w : tensor<2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<2048x12288xbf16>
  // CHECK: tensixflow.return %[[R]] : tensor<2048x12288xbf16>
  tensixflow.return %0 : tensor<2048x12288xbf16>
}

//--- negative_matmul.mlir

// matmul K-mismatch: lhs last dim (K) = 4096, rhs second-to-last dim (K) = 2048.
// Kept in its own split-file section: the MLIR module verifier short-circuits on
// the first failing op, so combining this with the rmsnorm case in one file would
// hide the rmsnorm diagnostic.
tensixflow.func @matmul_k_mismatch(%x: tensor<2048x4096xbf16>, %w: tensor<2048x12288xbf16>) -> tensor<2048x12288xbf16> {
  // expected-error @below {{mismatched contracting dimensions}}
  %0 = tensixflow.matmul %x, %w : tensor<2048x4096xbf16> x tensor<2048x12288xbf16> -> tensor<2048x12288xbf16>
  tensixflow.return %0 : tensor<2048x12288xbf16>
}

//--- negative_rmsnorm.mlir

// rmsnorm gamma-mismatch: gamma size 2048 != input last dim 4096.
tensixflow.func @rmsnorm_gamma_mismatch(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<2048xbf16>) -> tensor<1x2048x4096xbf16> {
  // expected-error @below {{gamma size (2048) must match the input's last dimension (4096)}}
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<2048xbf16> -> tensor<1x2048x4096xbf16>
  tensixflow.return %0 : tensor<1x2048x4096xbf16>
}
