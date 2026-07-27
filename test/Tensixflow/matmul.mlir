// RUN: split-file %s %t
// RUN: tensixflow-opt %t/positive.mlir | FileCheck %t/positive.mlir
// RUN: tensixflow-opt -verify-diagnostics %t/negative.mlir

//--- positive.mlir

// CHECK-LABEL: tensixflow.func @matmul_rank3
// CHECK-SAME: (%[[X:.*]]: tensor<1x2048x4096xbf16>, %[[W:.*]]: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16>
tensixflow.func @matmul_rank3(%x: tensor<1x2048x4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  // CHECK: %[[R:.*]] = tensixflow.matmul %[[X]], %[[W]] : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %0 = tensixflow.matmul %x, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  // CHECK: tensixflow.return %[[R]] : tensor<1x2048x12288xbf16>
  tensixflow.return %0 : tensor<1x2048x12288xbf16>
}

// CHECK-LABEL: tensixflow.func @matmul_rank2
// CHECK-SAME: (%[[X2:.*]]: tensor<2048x4096xbf16>, %[[W2:.*]]: tensor<4096x12288xbf16>) -> tensor<2048x12288xbf16>
tensixflow.func @matmul_rank2(%x: tensor<2048x4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<2048x12288xbf16> {
  // CHECK: %[[R2:.*]] = tensixflow.matmul %[[X2]], %[[W2]] : tensor<2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<2048x12288xbf16>
  %0 = tensixflow.matmul %x, %w : tensor<2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<2048x12288xbf16>
  // CHECK: tensixflow.return %[[R2]] : tensor<2048x12288xbf16>
  tensixflow.return %0 : tensor<2048x12288xbf16>
}

//--- negative.mlir

// Mismatched contracting dimension: lhs K = 4096 (last dim) but rhs K = 2048
// (second-to-last dim). The verifier must reject this.
tensixflow.func @matmul_k_mismatch(%x: tensor<2048x4096xbf16>, %w: tensor<2048x12288xbf16>) -> tensor<2048x12288xbf16> {
  // expected-error @below {{mismatched contracting dimensions}}
  %0 = tensixflow.matmul %x, %w : tensor<2048x4096xbf16> x tensor<2048x12288xbf16> -> tensor<2048x12288xbf16>
  tensixflow.return %0 : tensor<2048x12288xbf16>
}
