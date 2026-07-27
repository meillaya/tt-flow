// RUN: split-file %s %t
// RUN: tensixflow-opt %t/positive.mlir | FileCheck %t/positive.mlir
// RUN: tensixflow-opt -verify-diagnostics %t/negative.mlir

//--- positive.mlir

// CHECK-LABEL: tensixflow.func @rmsnorm_rank3
// CHECK-SAME: (%[[X:.*]]: tensor<1x2048x4096xbf16>, %[[G:.*]]: tensor<4096xbf16>) -> tensor<1x2048x4096xbf16>
tensixflow.func @rmsnorm_rank3(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>) -> tensor<1x2048x4096xbf16> {
  // CHECK: %[[R:.*]] = tensixflow.rmsnorm %[[X]], %[[G]] : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  // CHECK: tensixflow.return %[[R]] : tensor<1x2048x4096xbf16>
  tensixflow.return %0 : tensor<1x2048x4096xbf16>
}

//--- negative.mlir

// Gamma size mismatch: gamma is tensor<2048xbf16> but the input's last dim is
// 4096. The verifier must reject this.
tensixflow.func @rmsnorm_gamma_mismatch(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<2048xbf16>) -> tensor<1x2048x4096xbf16> {
  // expected-error @below {{gamma size (2048) must match the input's last dimension (4096)}}
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<2048xbf16> -> tensor<1x2048x4096xbf16>
  tensixflow.return %0 : tensor<1x2048x4096xbf16>
}
