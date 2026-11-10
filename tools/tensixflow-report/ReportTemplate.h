//===- ReportTemplate.h -----------------------------------------*- C++ -*-===//
//
// The self-contained HTML template for the TensixFlow visualizer. Pure template
// data: the inline CSS, the static section containers (the six required
// anchors), and the vanilla inline JS that statically renders the plan into
// them. There are no external resources here: no https?:// URLs, no
// <script src>, no CDN, no @import, no url(), no animation and no transition.
// The JS runs once on load and builds the DOM, with no event handlers and no
// interactivity beyond static layout. The plan JSON is embedded inline by the
// tool as `const PLAN = {...}` before kJs runs.
//
// Every rendered number is a projected static estimate. The device block in the
// plan names the hardware the projection assumes; the JS prints that name so a
// reader can never mistake a projection for a measurement.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_REPORTTEMPLATE_H
#define TENSIXFLOW_REPORTTEMPLATE_H

namespace mlir {
namespace tensixflow {
namespace report {

// Inline CSS (dark, static). No url(), @import, animation or transition.
constexpr const char kCss[] = R"tfcss(
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #0d1117; color: #c9d1d9; line-height: 1.5; }
header { padding: 24px 32px; background: #161b22; border-bottom: 1px solid #30363d; }
header h1 { margin: 0 0 4px; font-size: 20px; color: #58a6ff; }
header .sub { color: #8b949e; font-size: 13px; }
main { padding: 24px 32px; display: grid; gap: 20px; max-width: 1200px; margin: 0 auto; }
.panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 18px 20px; }
.panel h2 { margin: 0 0 12px; font-size: 14px; color: #58a6ff; text-transform: uppercase; letter-spacing: 0.05em; }
.cap { color: #8b949e; font-size: 12px; margin: 0 0 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #21262d; }
th { color: #8b949e; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.grid { display: grid; gap: 4px; }
.core { background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 8px 0; text-align: center; font-size: 11px; color: #8b949e; }
.bar { background: #21262d; border-radius: 4px; height: 18px; overflow: hidden; border: 1px solid #30363d; }
.bar > span { display: block; height: 100%; background: #3fb950; }
.bar > span.warn { background: #d29922; }
.bar > span.hot { background: #f85149; }
.badge { display: inline-block; padding: 3px 12px; border-radius: 12px; font-size: 12px; font-weight: 700; text-transform: uppercase; }
.badge.compute { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }
.badge.dram { background: #d2992233; color: #d29922; border: 1px solid #d29922; }
.badge.noc { background: #a371f733; color: #bc8cff; border: 1px solid #a371f7; }
.badge.balanced { background: #3fb95033; color: #3fb950; border: 1px solid #3fb950; }
ul.suggestions { margin: 8px 0 0; padding-left: 20px; }
ul.suggestions li { margin: 4px 0; }
.metric { display: inline-block; margin-right: 28px; }
.metric .v { font-size: 18px; color: #c9d1d9; font-weight: 700; }
.metric .k { font-size: 11px; color: #8b949e; text-transform: uppercase; }
.lane { display: grid; grid-template-columns: 90px 1fr; align-items: center; gap: 8px; margin: 6px 0; }
.lane .name { font-size: 12px; color: #8b949e; }
.lane .track { background: #21262d; border-radius: 4px; height: 26px; position: relative; border: 1px solid #30363d; }
.lane .block { position: absolute; top: 2px; bottom: 2px; border-radius: 3px; font-size: 10px; color: #0d1117; padding: 3px 6px; font-weight: 700; white-space: nowrap; }
.flow { font-size: 12px; color: #8b949e; margin: 10px 0 0; }
.muted { color: #8b949e; }
)tfcss";

// The seven section containers. The id attributes are the required anchors
// (#core-grid #shards #noc-routes #emitted-path #cb-usage #timeline
// #bottlenecks); the static markup keeps the anchors present even before the JS
// runs, and the JS fills each section's .body div from the embedded PLAN.
constexpr const char kSections[] = R"tfhtml(
<header>
  <h1>TensixFlow Plan Report</h1>
  <div class="sub" id="subtitle">projected static estimate</div>
</header>
<main>
  <section class="panel" id="core-grid"><h2>Core Grid</h2><div class="body"></div></section>
  <section class="panel" id="shards"><h2>Tensor Shards</h2><div class="body"></div></section>
  <section class="panel" id="noc-routes"><h2>NoC Routes</h2><div class="body"></div></section>
  <section class="panel" id="emitted-path"><h2>Projected Emitted Path</h2><div class="body"></div></section>
  <section class="panel" id="cb-usage"><h2>Circular Buffer Usage</h2><div class="body"></div></section>
  <section class="panel" id="timeline"><h2>Kernel Pipeline Timeline</h2><div class="body"></div></section>
  <section class="panel" id="bottlenecks"><h2>Projected Bottlenecks</h2><div class="body"></div></section>
</main>
)tfhtml";

// Vanilla inline JS (no framework). Statically renders the embedded PLAN into
// the seven section bodies on load. No event handlers, no fetch, no import, no
// eval, no animation. BUDGET is the device's usable L1 bytes per core from the
// plan's device block; occupancy is projected against that ceiling, not against
// a hand-written number.
constexpr const char kJs[] = R"tfjs(
(function () {
  var plan = (typeof PLAN !== 'undefined') ? PLAN : null;
  if (!plan) { return; }
  var device = plan.device;
  var BUDGET = device.l1_usable_bytes;
  function body(id) { return document.querySelector('#' + id + ' .body'); }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function fmt(n) { return Number(n).toLocaleString('en-US'); }

  var st = document.getElementById('subtitle');
  if (st) { st.textContent = 'projected | device: ' + device.name + ' (' + device.arch + ') | graph: ' + plan.graph_id + ' | ' + plan.layout.activation_sharding + ' activations | ' + plan.layout.weight_strategy + ' weights'; }

  var gy = plan.shard_grid.grid_y, gx = plan.shard_grid.grid_x;
  var h = '<p class="cap">projected ' + gy + ' x ' + gx + ' Tensix compute grid on ' + esc(device.name) + ' (' + fmt(gy * gx) + ' cores per die, ' + fmt(device.l1_usable_bytes) + ' bytes usable L1 per core)</p>';
  h += '<div class="grid" style="grid-template-columns:repeat(' + gx + ',1fr)">';
  for (var y = 0; y < gy; y++) { for (var x = 0; x < gx; x++) { h += '<div class="core">' + y + ',' + x + '</div>'; } }
  h += '</div>';
  body('core-grid').innerHTML = h;

  h = '<table><thead><tr><th>tensor</th><th>shape</th><th>dtype</th><th>placement</th></tr></thead><tbody>';
  for (var i = 0; i < plan.tensors.length; i++) {
    var t = plan.tensors[i];
    h += '<tr><td>' + esc(t.name) + '</td><td>' + esc(t.shape.join(' x ')) + '</td><td>' + esc(t.dtype) + '</td><td>' + esc(t.placement) + '</td></tr>';
  }
  h += '</tbody></table>';
  h += '<p class="flow">projected activation sharding: ' + esc(plan.layout.activation_sharding) + ' | projected weight strategy: ' + esc(plan.layout.weight_strategy) + '</p>';
  body('shards').innerHTML = h;

  var noc = plan.noc;
  h = '<table><thead><tr><th>projected route</th><th class="num">bytes</th></tr></thead><tbody>';
  h += '<tr><td>DRAM read</td><td class="num">' + fmt(noc.dram_read_bytes) + '</td></tr>';
  h += '<tr><td>DRAM write</td><td class="num">' + fmt(noc.dram_write_bytes) + '</td></tr>';
  h += '<tr><td>L1 (on-chip)</td><td class="num">' + fmt(noc.l1_bytes) + '</td></tr>';
  h += '<tr><td>NoC multicast</td><td class="num">' + fmt(noc.multicast_bytes) + '</td></tr>';
  h += '</tbody></table>';
  h += '<p class="flow">projected DRAM &#8594; cores (read) &#8594; L1 &#8594; compute &#8594; DRAM (write)';
  if (noc.multicast_bytes > 0) { h += ' | projected multicast: weight tiles fanned out across rows via row master nodes'; }
  h += '</p>';
  body('noc-routes').innerHTML = h;

  var em = plan.emission;
  var mappingCap = em.mapping === 'blocked-reuse'
    ? 'the emitted program assigns each core a 2D block of output tiles, stages the block\'s activation rows and weight columns in L1 once, multicasts the weight column block down the used row blocks, and accumulates every output tile from those resident buffers ('
    : 'the emitted program assigns output tiles to cores in contiguous row-major ranges (';
  h = '<p class="cap">' + mappingCap + esc(em.mapping) + ')</p>';
  h += '<table><thead><tr><th>projected emitted-path item</th><th class="num">value</th></tr></thead><tbody>';
  h += '<tr><td>cores used</td><td class="num">' + fmt(em.cores_used) + '</td></tr>';
  h += '<tr><td>max output tiles per core</td><td class="num">' + fmt(em.max_tiles_per_core) + '</td></tr>';
  h += '<tr><td>DRAM read bytes</td><td class="num">' + fmt(em.dram_read_bytes) + '</td></tr>';
  h += '<tr><td>DRAM write bytes</td><td class="num">' + fmt(em.dram_write_bytes) + '</td></tr>';
  h += '<tr><td>fused stage tiles</td><td class="num">' + fmt(em.fused_stage_tiles) + '</td></tr>';
  h += '<tr><td>fused variant feasible</td><td class="num">' + (em.fused_feasible ? 'yes' : 'no') + '</td></tr>';
  h += '</tbody></table>';
  h += em.mapping === 'blocked-reuse'
    ? '<p class="flow">projected emitted-path reads come from the 2D block assignment: the activation row block is read once per used column block, and the weight column block is read once and multicast down the used row blocks. The activation replication across columns remains, so the read figure is an upper bound, not a measured count.</p>'
    : '<p class="flow">projected emitted-path reads are unicast and replicated per output tile: each core reads the A and B tiles of every output tile it owns, a tile in K is re-read per output row, and no reuse across output tiles is modeled. The read figure is an upper bound, not a measured count.</p>';
  body('emitted-path').innerHTML = h;

  var cbs = plan.cb_specs, total = 0;
  h = '<table><thead><tr><th>CB index</th><th class="num">depth (tiles)</th><th class="num">tile bytes</th><th class="num">bytes</th></tr></thead><tbody>';
  for (i = 0; i < cbs.length; i++) {
    var b = cbs[i].depth * cbs[i].tile_bytes; total += b;
    h += '<tr><td>c_' + cbs[i].index + '</td><td class="num">' + fmt(cbs[i].depth) + '</td><td class="num">' + fmt(cbs[i].tile_bytes) + '</td><td class="num">' + fmt(b) + '</td></tr>';
  }
  h += '</tbody></table>';
  var occ = BUDGET > 0 ? total / BUDGET : 0, pct = Math.round(occ * 100);
  var cls = occ > 1 ? 'hot' : (occ > 0.85 ? 'warn' : '');
  h += '<p class="cap">projected per-core L1 occupancy: ' + fmt(total) + ' / ' + fmt(BUDGET) + ' bytes usable L1 (' + pct + '%)</p>';
  h += '<div class="bar"><span class="' + cls + '" style="width:' + Math.min(pct, 100) + '%"></span></div>';
  body('cb-usage').innerHTML = h;

  h = '<p class="cap">projected reader / compute / writer stages overlap across the tile loop (static schedule schematic)</p>';
  var lanes = [
    { name: 'reader', color: '#58a6ff', label: 'noc_read in0/in1', left: 0, width: 55 },
    { name: 'compute', color: '#3fb950', label: 'rmsnorm + matmul_tiles', left: 20, width: 60 },
    { name: 'writer', color: '#d29922', label: 'noc_write out', left: 45, width: 55 }
  ];
  for (i = 0; i < lanes.length; i++) {
    var ln = lanes[i];
    h += '<div class="lane"><div class="name">' + ln.name + '</div><div class="track">';
    h += '<div class="block" style="left:' + ln.left + '%;width:' + ln.width + '%;background:' + ln.color + '">' + ln.label + '</div>';
    h += '</div></div>';
  }
  body('timeline').innerHTML = h;

  var perf = plan.perf;
  h = '<p class="cap">projected binding constraint on ' + esc(device.name) + '</p>';
  h += '<p><span class="badge ' + esc(perf.bottleneck) + '">' + esc(perf.bottleneck) + '</span></p>';
  h += '<p><span class="metric"><span class="v">' + fmt(perf.est_cycles) + '</span> <span class="k">projected cycles</span></span>';
  h += '<span class="metric"><span class="v">' + Math.round(perf.compute_util * 100) + '%</span> <span class="k">projected compute util</span></span></p>';
  h += '<p class="cap">projected suggested optimizations</p><ul class="suggestions">';
  for (i = 0; i < perf.suggestions.length; i++) { h += '<li>' + esc(perf.suggestions[i]) + '</li>'; }
  if (perf.suggestions.length === 0) { h += '<li class="muted">(none)</li>'; }
  h += '</ul>';
  body('bottlenecks').innerHTML = h;
})();
)tfjs";

} // namespace report
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_REPORTTEMPLATE_H
