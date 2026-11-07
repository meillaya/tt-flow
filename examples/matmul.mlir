// Pure matmul codegen quickstart: bf16 operands, every dimension divisible by
// the 32x32 tile so the tile layout and CB depths align without padding.
// Run:
//   tensixflow-opt --tensixflow-device=n150 \
//     --tensixflow-infer-tile-layout \
//     --tensixflow-plan-noc-sharding \
//     --tensixflow-allocate-circular-buffers \
//     examples/matmul.mlir
// to attach the tile layout and the projected schedule, then serialize the plan:
//   tensixflow-opt ... examples/matmul.mlir | \
//     tensixflow-translate --tensixflow-to-json - > plan.json
// and render the offline report:
//   tensixflow-report --in plan.json --out report.html
tensixflow.func @matmul(%a: tensor<256x512xbf16>, %b: tensor<512x1024xbf16>) -> tensor<256x1024xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<256x512xbf16> x tensor<512x1024xbf16> -> tensor<256x1024xbf16>
  tensixflow.return %0 : tensor<256x1024xbf16>
}
