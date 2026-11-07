// Small pure matmul used by the emitted-project check: M=128, N=128, K=64, so
// 4 x 4 output tiles and 2 K tiles. A 4 x 4 tile problem uses 16 of the plan
// grid's 64 cores, which exercises the split_work_to_cores path where the grid
// is larger than the output-tile count.
tensixflow.func @matmul_small(%a: tensor<128x64xbf16>, %b: tensor<64x128xbf16>) -> tensor<128x128xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<128x64xbf16> x tensor<64x128xbf16> -> tensor<128x128xbf16>
  tensixflow.return %0 : tensor<128x128xbf16>
}
