const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '../..');
const INDEX = path.join(ROOT, 'artifacts/results/result_index.json');
const OUT = path.join(ROOT, 'paper/figures/editable/gsa_benchmark_and_method.pptx');

if (!fs.existsSync(INDEX)) throw new Error(`Missing ${INDEX}; run python eval/build_result_index.py`);
const index = JSON.parse(fs.readFileSync(INDEX, 'utf8'));

const C = {
  ink: '202124', muted: '5F6368', blue: '0072B2', teal: '009E73', red: 'D55E00',
  amber: 'E69F00', light: 'F5F6F7', rule: 'C7CBD1', white: 'FFFFFF', paleBlue: 'EAF4FA',
  paleTeal: 'E9F6F1', paleRed: 'FBEFEA', paleAmber: 'FFF6DF', darkGray: '3C4043'
};
const family = { G: C.blue, S: C.teal, A: C.red };

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'GSA Forge';
pptx.subject = 'Editable publication figures';
pptx.title = 'GSA benchmark and routed MOPD';
pptx.company = 'GSA Forge';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Arial', bodyFontFace: 'Arial', lang: 'en-US'
};
pptx.defineSlideMaster({
  title: 'MASTER',
  background: { color: C.white },
  objects: [
    { line: { x: 0.45, y: 7.12, w: 12.43, h: 0, line: { color: C.rule, width: 0.7 } } },
    { text: { text: 'GSA Forge | editable source', options: { x: 0.48, y: 7.18, w: 3.0, h: 0.18, fontFace: 'Arial', fontSize: 7.5, color: C.muted, margin: 0 } } }
  ],
  slideNumber: { x: 12.45, y: 7.16, w: 0.35, h: 0.2, fontFace: 'Arial', fontSize: 7.5, color: C.muted, align: 'right', margin: 0 }
});

function addTitle(slide, title, subtitle) {
  slide.addText(title, { x: 0.48, y: 0.28, w: 8.8, h: 0.36, fontFace: 'Arial', fontSize: 21, bold: true, color: C.ink, margin: 0, breakLine: false });
  if (subtitle) slide.addText(subtitle, { x: 0.5, y: 0.72, w: 11.8, h: 0.28, fontFace: 'Arial', fontSize: 9.5, color: C.muted, margin: 0 });
}

function addRect(slide, x, y, w, h, fill, line = C.rule, radius = 0.04) {
  const shape = radius ? pptx.ShapeType.roundRect : pptx.ShapeType.rect;
  slide.addShape(shape, { x, y, w, h, rectRadius: radius, fill: { color: fill }, line: { color: line, width: 0.8 } });
}

function addLabel(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h, fontFace: 'Arial', fontSize: opts.size || 10, color: opts.color || C.ink,
    bold: !!opts.bold, align: opts.align || 'left', valign: opts.valign || 'mid',
    margin: opts.margin === undefined ? 0.06 : opts.margin, breakLine: false, fit: 'shrink'
  });
}

function arrow(slide, x, y, w, h, color = C.darkGray, dash = 'solid') {
  slide.addShape(pptx.ShapeType.line, { x, y, w, h, line: { color, width: 1.6, beginArrowType: 'none', endArrowType: 'triangle', dash } });
}

// Slide 1: a compact, evidence-aware benchmark table.
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'Benchmark results under one audited protocol', 'Only gsa-eval-v7 cells matching the current scorer fingerprint and task budget are admitted; withheld is not zero.');
  const benches = index.benchmarks;
  const runs = index.runs.filter(r => r.include_in_main_table);
  const x0 = 0.5, y0 = 1.28, modelW = 3.18, colW = 1.82, headH = 0.84, rowH = 0.78;
  addRect(slide, x0, y0, modelW, headH, C.darkGray, C.darkGray, 0);
  addLabel(slide, 'Model', x0 + 0.15, y0, modelW - 0.25, headH, { size: 11, color: C.white, bold: true });
  const benchColors = [C.blue, C.teal, C.teal, C.amber, C.red];
  benches.forEach((bench, i) => {
    const x = x0 + modelW + i * colW;
    addRect(slide, x, y0, colW, headH, benchColors[i], benchColors[i], 0);
    addLabel(slide, bench.display_name, x + 0.05, y0 + 0.03, colW - 0.1, headH - 0.06, { size: 9.5, color: C.white, bold: true, align: 'center' });
  });
  runs.forEach((run, r) => {
    const y = y0 + headH + r * rowH;
    const fill = r % 2 ? C.white : C.light;
    addRect(slide, x0, y, modelW, rowH, fill, C.rule, 0);
    addLabel(slide, run.display_name, x0 + 0.15, y, modelW - 0.25, rowH, { size: 10.5, bold: r >= 2 });
    benches.forEach((bench, i) => {
      const x = x0 + modelW + i * colW;
      const cell = run.results[bench.id];
      addRect(slide, x, y, colW, rowH, fill, C.rule, 0);
      let text = '--', color = C.muted, bold = false;
      if (cell && cell.eligible) { text = cell.value.toFixed(2); color = C.ink; bold = true; }
      else if (cell && cell.exists && (cell.reasons || []).includes('trust_gate_withheld')) { text = 'WITHHELD'; color = C.red; bold = true; }
      addLabel(slide, text, x + 0.04, y, colW - 0.08, rowH, { size: text === 'WITHHELD' ? 8.7 : 12.5, color, bold, align: 'center' });
    });
  });
  const fy = y0 + headH + runs.length * rowH + 0.35;
  addRect(slide, 0.55, fy, 4.0, 0.76, C.paleBlue, C.blue);
  addLabel(slide, 'Public transfer', 0.72, fy + 0.08, 1.35, 0.23, { size: 9, color: C.blue, bold: true });
  addLabel(slide, 'E1 vs base: -3.50 / -3.17 / -38.00 points.', 0.72, fy + 0.31, 3.58, 0.32, { size: 8.5, color: C.darkGray });
  addRect(slide, 4.7, fy, 3.55, 0.76, C.paleAmber, C.amber);
  addLabel(slide, 'Protocol-shift stress', 4.87, fy + 0.08, 1.55, 0.23, { size: 9, color: C.amber, bold: true });
  addLabel(slide, 'E1 vs base: -41.08 points; keep separate.', 4.87, fy + 0.31, 3.14, 0.32, { size: 8.5, color: C.darkGray });
  addRect(slide, 8.4, fy, 4.35, 0.76, C.paleRed, C.red);
  addLabel(slide, 'Internal benchmark scope', 8.57, fy + 0.08, 1.78, 0.23, { size: 9, color: C.red, bold: true });
  addLabel(slide, 'GSA-G: E1 vs base -9.01; S/A remain blocked.', 8.57, fy + 0.31, 3.95, 0.32, { size: 8.5, color: C.darkGray });
}

// Slide 2: routed MOPD method, entirely in native shapes.
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'Dependency-routed multi-teacher on-policy distillation', 'The prediction family selects a specialist; every teacher scores the exact prefix sampled by the student.');
  addRect(slide, 0.55, 1.35, 2.1, 1.02, C.light, C.rule);
  addLabel(slide, 'G/S/A example', 0.75, 1.48, 1.7, 0.24, { size: 11, bold: true });
  addLabel(slide, 'image + task\nprediction family\nupstream state', 0.75, 1.76, 1.7, 0.48, { size: 9, color: C.darkGray, valign: 'top' });

  addRect(slide, 3.15, 1.28, 2.35, 1.18, C.darkGray, C.darkGray);
  addLabel(slide, 'Student policy', 3.38, 1.48, 1.89, 0.26, { size: 13, color: C.white, bold: true, align: 'center' });
  addLabel(slide, 'sample y ~ pi_theta(.|x)', 3.38, 1.82, 1.89, 0.28, { size: 9.5, color: C.white, align: 'center' });
  addLabel(slide, 'on-policy rollout', 3.38, 2.11, 1.89, 0.19, { size: 8.5, color: 'DADCE0', align: 'center' });
  arrow(slide, 2.66, 1.86, 0.46, 0, C.darkGray);

  addRect(slide, 6.05, 1.35, 1.55, 1.02, C.light, C.rule);
  addLabel(slide, 'Router', 6.26, 1.52, 1.13, 0.25, { size: 12, bold: true, align: 'center' });
  addLabel(slide, 'capability\nfamily', 6.26, 1.82, 1.13, 0.37, { size: 9, color: C.darkGray, align: 'center' });
  arrow(slide, 5.52, 1.86, 0.51, 0, C.darkGray);

  const ty = [1.08, 1.83, 2.58];
  ['G', 'S', 'A'].forEach((f, i) => {
    const fills = { G: C.paleBlue, S: C.paleTeal, A: C.paleRed };
    addRect(slide, 8.2, ty[i], 2.25, 0.58, fills[f], family[f]);
    addLabel(slide, `${f} specialist LoRA`, 8.42, ty[i], 1.82, 0.58, { size: 10.5, color: family[f], bold: true, align: 'center' });
  });
  addRect(slide, 10.73, 1.08, 1.92, 2.08, C.light, C.rule);
  addLabel(slide, 'Shared frozen\nteacher backbone', 10.97, 1.35, 1.44, 0.55, { size: 11.5, bold: true, align: 'center' });
  addLabel(slide, 'same image\nsame prompt\nsame generated prefix', 10.97, 2.05, 1.44, 0.72, { size: 9, color: C.darkGray, align: 'center' });
  arrow(slide, 7.62, 1.86, 0.55, -0.47, C.blue);
  arrow(slide, 7.62, 1.86, 0.55, 0.26, C.teal);
  arrow(slide, 7.62, 1.86, 0.55, 1.01, C.red);
  ty.forEach(y => arrow(slide, 10.46, y + 0.29, 0.25, 0, C.darkGray));

  addRect(slide, 5.08, 3.55, 3.22, 0.86, C.white, C.darkGray);
  addLabel(slide, 'Generalized JSD on student tokens', 5.34, 3.66, 2.7, 0.28, { size: 11.5, bold: true, align: 'center' });
  addLabel(slide, 'JSD_beta(pi_teacher, pi_student)', 5.34, 3.98, 2.7, 0.23, { size: 9.2, color: C.darkGray, align: 'center' });
  arrow(slide, 11.68, 3.18, -3.35, 0.74, C.darkGray);
  arrow(slide, 5.06, 3.98, -1.0, -1.48, C.darkGray);

  addLabel(slide, 'Balanced exposure', 0.58, 4.8, 1.75, 0.27, { size: 11, bold: true });
  const bx = [2.25, 3.68, 5.11];
  ['G batch', 'S batch', 'A batch'].forEach((t, i) => {
    addRect(slide, bx[i], 4.69, 1.2, 0.52, [C.paleBlue, C.paleTeal, C.paleRed][i], [C.blue, C.teal, C.red][i]);
    addLabel(slide, t, bx[i], 4.69, 1.2, 0.52, { size: 9.5, color: [C.blue, C.teal, C.red][i], bold: true, align: 'center' });
    if (i < 2) arrow(slide, bx[i] + 1.21, 4.95, 0.2, 0, C.rule);
  });
  addLabel(slide, 'round-robin, family-homogeneous global batches', 6.57, 4.72, 3.1, 0.43, { size: 9.3, color: C.darkGray });

  addLabel(slide, 'Cascade context', 0.58, 5.58, 1.75, 0.27, { size: 11, bold: true });
  ['Ground', 'Spatial', 'Act'].forEach((t, i) => {
    const x = 2.25 + i * 2.08;
    addRect(slide, x, 5.46, 1.55, 0.58, [C.paleBlue, C.paleTeal, C.paleRed][i], [C.blue, C.teal, C.red][i]);
    addLabel(slide, `${['G', 'S', 'A'][i]} | ${t}`, x, 5.46, 1.55, 0.58, { size: 10, color: [C.blue, C.teal, C.red][i], bold: true, align: 'center' });
    if (i < 2) arrow(slide, x + 1.58, 5.75, 0.45, 0, C.darkGray);
  });
  addLabel(slide, 'Dependency context is preserved from grounding to spatial reasoning to action.', 8.72, 5.42, 3.75, 0.72, { size: 9.2, color: C.darkGray });
}

// Slide 3: benchmark evidence pipeline and current scope.
{
  const slide = pptx.addSlide('MASTER');
  addTitle(slide, 'Benchmark evidence pipeline', 'A score enters the paper only after label provenance, split, leakage, and evaluation-code checks.');
  const stages = [
    ['Human GT', 'bbox + point\ncurrent G scope', C.blue, C.paleBlue],
    ['Stable identity', 'image SHA-256\npath-independent split', C.darkGray, C.light],
    ['Leakage audit', 'image bytes +\nnormalized text', C.teal, C.paleTeal],
    ['Unified eval', 'same prompts, subset,\nparser and scorer', C.amber, C.paleAmber],
    ['Result gate', 'schema + code hash +\nparse/truncation limits', C.red, C.paleRed]
  ];
  stages.forEach((s, i) => {
    const x = 0.55 + i * 2.54;
    addRect(slide, x, 1.55, 2.12, 1.22, s[3], s[2]);
    addLabel(slide, s[0], x + 0.16, 1.69, 1.8, 0.28, { size: 11.5, color: s[2], bold: true, align: 'center' });
    addLabel(slide, s[1], x + 0.16, 2.02, 1.8, 0.54, { size: 8.8, color: C.darkGray, align: 'center' });
    if (i < stages.length - 1) arrow(slide, x + 2.14, 2.16, 0.36, 0, C.rule);
  });

  addLabel(slide, 'Current internal evidence', 0.58, 3.35, 3.0, 0.3, { size: 13, bold: true });
  const scopes = [
    ['G', 'bbox / point', 'Human GT available', C.blue, C.paleBlue],
    ['S', 'depth / relations', 'Human GT pending', C.teal, C.paleTeal],
    ['A', 'affordance / DoF / action', 'Human GT pending', C.red, C.paleRed]
  ];
  scopes.forEach((s, i) => {
    const y = 3.82 + i * 0.76;
    addRect(slide, 0.65, y, 0.62, 0.52, s[4], s[3]);
    addLabel(slide, s[0], 0.65, y, 0.62, 0.52, { size: 13, color: s[3], bold: true, align: 'center' });
    addLabel(slide, s[1], 1.48, y, 2.65, 0.52, { size: 10.5, bold: true });
    addLabel(slide, s[2], 4.25, y, 2.25, 0.52, { size: 9.5, color: i === 0 ? C.teal : C.red, bold: true });
  });

  addRect(slide, 7.25, 3.52, 5.45, 2.6, C.light, C.rule);
  addLabel(slide, 'What the table can claim now', 7.55, 3.73, 4.85, 0.32, { size: 13, bold: true });
  addLabel(slide,
    '1  Public spatial transfer under one protocol\n2  Object-aligned RefSpatial subgroup\n3  Grounding-only internal human-GT benchmark\n4  Explicit withheld cells when format reliability fails',
    7.55, 4.18, 4.65, 1.42, { size: 10.3, color: C.darkGray, valign: 'top', margin: 0.02 });
  addLabel(slide, 'Not yet claimable: a complete eight-form G/S/A benchmark mean.', 7.55, 5.67, 4.65, 0.28, { size: 9.5, color: C.red, bold: true });
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
pptx.writeFile({ fileName: OUT });
console.log(OUT);
