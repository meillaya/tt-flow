# Hardware

TensixFlow models four Tenstorrent cards. `spec/devices.json` records every value in this document with a source id, and `tools/gen_device_spec.py` turns that file into the C++ header and the Python module the build reads. Change a value in the JSON, then run the generator; the header and the module are output.

The scheduler models a device as `chips x compute grid`, so the modeled core count is 64 for n150, 112 for n300, and 120 for p100a and p150a.

## Compute

| Device | Architecture | Chips | Compute grid | Dispatch | Tensix per die / after harvest | Clock | Matrix unit MACs per instruction | FLOPs per cycle per core |
|---|---|---|---|---|---|---|---|---|
| n150 | Wormhole B0 | 1 | 8x8 | row | 80 / 72 | 1.0 GHz | 2,048 | 4,096 |
| n300 | Wormhole B0 | 2 | 8x7 per chip | row | 80 / 64 per die | 1.0 GHz | 2,048 | 4,096 |
| p100a | Blackhole | 1 | 12x10 | column | 140 / 120 | 1.35 GHz | 2,048 | 4,096 |
| p150a | Blackhole | 1 | 12x10 | column | 140 / 120 | 1.35 GHz | 2,048 | 4,096 |

The matrix unit needs one instruction per fidelity phase. The cost model assumes one phase, which matches the bf16 matmul path.

## Memory

| Device | L1 usable per core | L1 nominal per core | SRAM | DRAM type | DRAM capacity | DRAM bandwidth |
|---|---|---|---|---|---|---|
| n150 | 1,499,136 B (1,464 KiB) | 1,572,864 B (1,536 KiB) | 113,246,208 B (108 MiB) | GDDR6 | 12,884,901,888 B (12 GiB) | 288,000,000,000 B/s (288 GB/s) |
| n300 | 1,499,136 B (1,464 KiB) per chip | 1,572,864 B (1,536 KiB) per chip | 201,326,592 B (192 MiB) | GDDR6 | 25,769,803,776 B (24 GiB), board aggregate | 576,000,000,000 B/s (576 GB/s), board aggregate |
| p100a | 1,572,864 B (1,536 KiB) | 1,572,864 B (1,536 KiB) | 188,743,680 B (180 MiB) | GDDR6 | 30,064,771,072 B (28 GiB) | 448,000,000,000 B/s (448 GB/s) |
| p150a | 1,572,864 B (1,536 KiB) | 1,572,864 B (1,536 KiB) | 188,743,680 B (180 MiB) | GDDR6 | 34,359,738,368 B (32 GiB) | 512,000,000,000 B/s (512 GB/s) |

The vendor pages state DRAM capacity in GB. `spec/devices.json` records the equivalent GiB byte count to keep the arithmetic in binary units, and each device note records the vendor spelling.

The NoC block is the same on all four devices: two independent NoCs, 32-byte flits, an 8,192-byte maximum packet payload, and 32 bytes per cycle per direction into L1. The cost model uses the 32 bytes per cycle figure for its NoC bound.

## Board

| Device | PCIe | Power | Ethernet |
|---|---|---|---|
| n150 | 4.0 x16 | 160 W | 2x QSFP-DD 200G active, 2x Warp 100 bridge |
| n300 | 4.0 x16 | 300 W | 2x QSFP-DD 200G active, 2x Warp 100 bridge, internal 200G chip-to-chip |
| p100a | 5.0 x16 | 300 W | none |
| p150a | 5.0 x16 | 300 W | 4x QSFP-DD 800G passive |

## Tensix counts

Wormhole has 80 functional Tensix workers on a 10x12 NoC coordinate grid. The n150 harvests one row, which leaves 72 workers, and tt-metal reserves a row for dispatch, which leaves the 8x8 compute grid. The n300 harvests two rows per die, which leaves 64 workers and an 8x7 compute grid per chip.

Blackhole has three published counts and `spec/devices.json` records all three:

- 140 functional Tensix workers on the 17x12 NoC coordinate grid.
- A 13x10 unharvested compute grid, which is 130 cores.
- 120 Tensix cores on the p100a and p150a product pages.

The scheduler models the 120-core product figure through the 1x-harvested 12x10 column-dispatch grid. The two other numbers stay in the device note rather than being dropped.

## Identifying names

`--tensixflow-device` accepts the device key or an alias.

| Key | Aliases |
|---|---|
| n150 | n150d, n150s |
| n300 | n300d, n300s |
| p100a | bh_p100a |
| p150a | bh_p150a |

The default device is n150. `spec/devices.json` sets that default, and an unknown name fails with the list of keys.

## Sources

Every source was retrieved on 2026-11-06. The source id in the last column matches the ids in `spec/devices.json`, where each numeric group names its source.

| Source | URL | Supports |
|---|---|---|
| tt-aibs-wormhole | https://docs.tenstorrent.com/aibs/wormhole/index.html | Wormhole card specifications: 80 Tensix on the die, n150 72 Tensix at 1 GHz with 12 GB GDDR6 at 288 GB/s and 160 W, n300 two ASICs with 64 Tensix each and 24 GB at 576 GB/s and 300 W, PCIe 4.0 x16, 2x QSFP-DD 200G, Ubuntu 22.04 |
| tt-aibs-blackhole | https://docs.tenstorrent.com/aibs/blackhole/index.html | Blackhole card specifications: 120 Tensix, up to 1.35 GHz, p100a 28 GB GDDR6 at 448 GB/s, p150a 32 GB at 512 GB/s, 300 W, PCIe 5.0 x16, p150a 4x QSFP-DD 800G, 180 MB SRAM, Ubuntu 22.04 |
| tt-metal-wormhole-soc-desc | https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml | Wormhole physical description: 10x12 NoC coordinate grid, 80 functional Tensix workers, WORMHOLE_B0, 1,499,136-byte worker L1 |
| tt-metal-wormhole-core-desc | https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/core_descriptors/wormhole_b0_80_arch.yaml | Wormhole compute grids and dispatch: nebula_x1 row dispatch 8x8 grid for n150, nebula_x2 row dispatch 8x7 grid for n300 |
| tt-metal-blackhole-soc-desc | https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/soc_descriptors/blackhole_140_arch.yaml | Blackhole physical description: 17x12 NoC coordinate grid, 140 functional Tensix workers, BLACKHOLE, 1,572,864-byte worker L1 |
| tt-metal-blackhole-core-desc | https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/core_descriptors/blackhole_140_arch.yaml | Blackhole compute grids and column dispatch: unharvested 13x10, 1x harvested 12x10, 2x harvested 11x10 |
| tt-isa-l1 | https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/L1.md | L1 size per Tensix tile: 1,464 KiB on Wormhole, 1,536 KiB on Blackhole. Each NoC reads and writes 256 bits per cycle against L1 |
| tt-isa-matrix-unit | https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md | Matrix unit throughput: 4.096 TFLOP/s per core at 1 GHz and one fidelity phase, one to four fidelity phases, and 72/64/80 Tensix tiles per n150/n300/Galaxy part |
| tt-isa-noc | https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/NoC/README.md | NoC structure: two physically separate NoCs, 256-bit flits (32 bytes), and packets of one header flit plus up to 256 data flits (8,192 data bytes) |
| tt-metal-install | https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md | Toolchain minimums: Ubuntu 22.04, tt-kmd 2.8.0 or newer, firmware 19.8.1, tt-smi 5.0.0 or newer, IOMMU required |

The Blackhole ISA set available at retrieval time has no NoC page, so the two Blackhole entries take their NoC block from the Wormhole ISA page. tt-metal configures noc0 and noc1 on Blackhole, which matches the two-NoC value.
