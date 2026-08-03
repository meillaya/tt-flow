// REQUIRES: ttmlir-opt

// RUN: tensixflow-opt --tensixflow-emit-ttir %s | ttmlir-opt | FileCheck %s

// Gated ttmlir-opt parse check (todo 53, D-findings: ttmlir-opt runs TTIR on
// CPU). When ttmlir-opt is on PATH (lit feature detected in lit.cfg.py), the
// TTIR text emitted by --tensixflow-emit-ttir (todo 51) is piped through a
// bare ttmlir-opt parse round-trip: it must parse and print the TTIR ops
// back. When ttmlir-opt is ABSENT (the current state; todo 50 took the
// source-study fallback and never built it), lit reports this test as SKIP
// (UNSUPPORTED), never FAIL, and exits 0. The FileCheck structural test
// test/Tensixflow/emit_ttir.mlir ALWAYS runs regardless (needs only
// tensixflow-opt + FileCheck).
//
// Input keeps to ops with study-grounded TTIR mappings (rmsnorm, matmul, add,
// fused_rmsnorm_matmul) so the emitted text is parse-clean: every func has a
// return terminator and no UNSUPPORTED marker comments. The gelu marker case
// belongs to the structural test, not the parse round-trip.

// CHECK-LABEL: func.func @rmsnorm_matmul
// CHECK: ttir.rms_norm
// CHECK: ttir.matmul
// CHECK: return
// CHECK-LABEL: func.func @just_add
// CHECK: ttir.add
// CHECK: return
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

tensixflow.func @just_add(%a: tensor<2048x4096xbf16>, %b: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %0 = tensixflow.add %a, %b : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}
