// RUN: tensixflow-opt %s | FileCheck %s

// CHECK-LABEL: tensixflow.func @elementwise
// CHECK-SAME: (%[[X:.*]]: tensor<2048x4096xbf16>, %[[Y:.*]]: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16>
tensixflow.func @elementwise(%x: tensor<2048x4096xbf16>, %y: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  // CHECK: %[[ADD:.*]] = tensixflow.add %[[X]], %[[Y]] : tensor<2048x4096xbf16>
  %0 = tensixflow.add %x, %y : tensor<2048x4096xbf16>
  // CHECK: %[[GELU:.*]] = tensixflow.gelu %[[ADD]] : tensor<2048x4096xbf16>
  %1 = tensixflow.gelu %0 : tensor<2048x4096xbf16>
  // CHECK: tensixflow.return %[[GELU]] : tensor<2048x4096xbf16>
  tensixflow.return %1 : tensor<2048x4096xbf16>
}

// CHECK-LABEL: tensixflow.func @with_constant
tensixflow.func @with_constant() -> tensor<4x4xf32> {
  // CHECK: %[[C:.*]] = tensixflow.constant dense<1.000000e+00> : tensor<4x4xf32>
  %0 = tensixflow.constant dense<1.0> : tensor<4x4xf32>
  // CHECK: tensixflow.return %[[C]] : tensor<4x4xf32>
  tensixflow.return %0 : tensor<4x4xf32>
}
