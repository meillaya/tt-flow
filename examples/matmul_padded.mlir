// Pure matmul with dimensions that are not multiples of the 32x32 tile. The
// host pads each operand tile with zeros and the reference checks the same
// padded product, so this example exercises the tilize padding path that every
// tile-aligned example skips.
tensixflow.func @matmul_padded(%a: tensor<100x50xbf16>, %b: tensor<50x100xbf16>) -> tensor<100x100xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<100x50xbf16> x tensor<50x100xbf16> -> tensor<100x100xbf16>
  tensixflow.return %0 : tensor<100x100xbf16>
}
