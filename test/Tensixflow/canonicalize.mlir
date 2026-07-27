// RUN: tensixflow-opt --canonicalize %s | FileCheck %s
// RUN: tensixflow-opt --tensixflow-canonicalize %s | FileCheck %s

// Both the standard --canonicalize pass (which picks up the dialect patterns
// via hasCanonicalizer) and the tensixflow --tensixflow-canonicalize wrapper
// apply the same identity-add-removal pattern, so one set of CHECK lines
// serves both RUN lines.

// Positive: add %x, %zero -> %x (zero on the RHS). The add must be gone and
// the return must use %x directly. CHECK-NOT spans the func body up to the
// return, so any surviving tensixflow.add fails the test (a dead zero constant
// may or may not remain; it is not asserted either way).
// CHECK-LABEL: tensixflow.func @add_zero_rhs
// CHECK-SAME: (%[[X:[a-z0-9]+]]: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16>
// CHECK-NOT: tensixflow.add
// CHECK: tensixflow.return %[[X]] : tensor<2048x4096xbf16>
tensixflow.func @add_zero_rhs(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %z = tensixflow.constant dense<0.0> : tensor<2048x4096xbf16>
  %r = tensixflow.add %x, %z : tensor<2048x4096xbf16>
  tensixflow.return %r : tensor<2048x4096xbf16>
}

// Positive: add %zero, %x -> %x (zero on the LHS; symmetric operand order).
// CHECK-LABEL: tensixflow.func @add_zero_lhs
// CHECK-SAME: (%[[XL:[a-z0-9]+]]: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16>
// CHECK-NOT: tensixflow.add
// CHECK: tensixflow.return %[[XL]] : tensor<2048x4096xbf16>
tensixflow.func @add_zero_lhs(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %z = tensixflow.constant dense<0.0> : tensor<2048x4096xbf16>
  %r = tensixflow.add %z, %x : tensor<2048x4096xbf16>
  tensixflow.return %r : tensor<2048x4096xbf16>
}

// Positive: fp32 zero is folded too (the pattern handles any float element
// type via APFloat::isZero).
// CHECK-LABEL: tensixflow.func @add_zero_f32
// CHECK-SAME: (%[[XF:[a-z0-9]+]]: tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK-NOT: tensixflow.add
// CHECK: tensixflow.return %[[XF]] : tensor<16x16xf32>
tensixflow.func @add_zero_f32(%x: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %z = tensixflow.constant dense<0.0> : tensor<16x16xf32>
  %r = tensixflow.add %x, %z : tensor<16x16xf32>
  tensixflow.return %r : tensor<16x16xf32>
}

// Negative: two non-constant operands. The add is NOT an identity and must be
// preserved unchanged.
// CHECK-LABEL: tensixflow.func @add_nonzero
// CHECK-SAME: (%[[XN:[a-z0-9]+]]: tensor<2048x4096xbf16>, %[[YN:[a-z0-9]+]]: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16>
// CHECK: %[[R:[a-z0-9]+]] = tensixflow.add %[[XN]], %[[YN]] : tensor<2048x4096xbf16>
// CHECK: tensixflow.return %[[R]] : tensor<2048x4096xbf16>
tensixflow.func @add_nonzero(%x: tensor<2048x4096xbf16>, %y: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %r = tensixflow.add %x, %y : tensor<2048x4096xbf16>
  tensixflow.return %r : tensor<2048x4096xbf16>
}

// Negative: a NON-zero splat constant is not an identity and must be preserved.
// CHECK-LABEL: tensixflow.func @add_nonzero_const
// CHECK-SAME: (%[[XC:[a-z0-9]+]]: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16>
// CHECK: tensixflow.add %[[XC]], %{{[a-z0-9]+}} : tensor<2048x4096xbf16>
tensixflow.func @add_nonzero_const(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %c = tensixflow.constant dense<1.0> : tensor<2048x4096xbf16>
  %r = tensixflow.add %x, %c : tensor<2048x4096xbf16>
  tensixflow.return %r : tensor<2048x4096xbf16>
}
