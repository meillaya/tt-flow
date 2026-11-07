// RUN: tensixflow-opt --tensixflow-emit-ttir %s | FileCheck %s

// Structural test for the UNSUPPORTED convention of --tensixflow-emit-ttir.
// This module is not a parser-gate input: the fused MLP tail has no TTIR
// mapping, so its result stays unbound and the emitted function has no return
// terminator. The parser gate input is emit_ttir_ttmlir_parse.mlir.
//
// The marker colon is escaped on the CHECK lines so lit does not read them as
// UNSUPPORTED directives.

// Partial fused MLP: rms_norm, matmul, broadcast bias add, and gelu map, but
// the tail (matmul w2 and residual add) has no TTIR mapping. The fused result
// stays unbound, so the return is marked transitively.
// CHECK-LABEL: func.func @partially_supported_mlp
// CHECK: "ttir.rms_norm"
// CHECK: "ttir.matmul"
// CHECK: "ttir.add"
// CHECK: "ttir.gelu"
// CHECK-NEXT: // UNSUPPORTED{{:}} tensixflow.fused_mlp tail (matmul w2, residual add) has no TTIR mapping
// CHECK-NEXT: // UNSUPPORTED{{:}} tensixflow.return has no TTIR mapping (operand is unsupported)
// CHECK-NEXT: }
tensixflow.func @partially_supported_mlp(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w1: tensor<4096x12288xbf16>, %b1: tensor<1x2048x12288xbf16>, %w2: tensor<12288x4096xbf16>, %residual: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  %0 = tensixflow.fused_mlp %x, %gamma, %w1, %b1, %w2, %residual : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x12288xbf16>, tensor<1x2048x12288xbf16>, tensor<12288x4096xbf16>, tensor<1x2048x4096xbf16> -> tensor<1x2048x4096xbf16>
  tensixflow.return %0 : tensor<1x2048x4096xbf16>
}
