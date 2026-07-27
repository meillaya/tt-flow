// RUN: tensixflow-opt --pass-pipeline='builtin.module(tensixflow-infer-tile-layout,tensixflow-canonicalize)' %s | FileCheck %s

// The D22 probe: both passes are no-op stubs at this stage, so the IR is
// unchanged. This proves the two-pass --pass-pipeline REGISTRATION +
// COMPOSITION works (the pipeline syntax was unverified during planning).

// CHECK-LABEL: tensixflow.func @probe
// CHECK-SAME: (%[[X:.*]]: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16>
tensixflow.func @probe(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  // CHECK: %[[G:.*]] = tensixflow.gelu %[[X]] : tensor<2048x4096xbf16>
  %0 = tensixflow.gelu %x : tensor<2048x4096xbf16>
  // CHECK: tensixflow.return %[[G]] : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}
