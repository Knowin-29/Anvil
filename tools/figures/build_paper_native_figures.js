const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '../..');
const OUT = path.join(ROOT, 'paper/figures/editable/gsa_paper_native.pptx');
const C = {
  ink: '202124', muted: '5F6368', blue: '0072B2', green: '009E73', red: 'D55E00',
  amber: 'E69F00', rule: 'B8BDC5', light: 'F5F6F7', white: 'FFFFFF',
  paleBlue: 'EAF4FA', paleGreen: 'E9F6F1', paleRed: 'FBEFEA', paleAmber: 'FFF6DF'
};

const pptx = new pptxgen();
pptx.defineLayout({ name: 'PAPER', width: 13.8, height: 5.4 });
pptx.layout = 'PAPER';
pptx.author = 'GSA-Forge';
pptx.subject = 'Editable paper-native figures';
pptx.title = 'GSA-Forge paper figures';
pptx.lang = 'en-US';
pptx.theme = { headFontFace: 'Arial', bodyFontFace: 'Arial', lang: 'en-US' };

function rect(slide, x, y, w, h, fill, line = C.rule, width = 1) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: fill }, line: { color: line, width } });
}

function text(slide, value, x, y, w, h, options = {}) {
  slide.addText(value, {
    x, y, w, h, margin: options.margin === undefined ? 0.05 : options.margin,
    fontFace: 'Arial', fontSize: options.size || 14, color: options.color || C.ink,
    bold: !!options.bold, align: options.align || 'left', valign: options.valign || 'mid',
    breakLine: false, fit: 'shrink'
  });
}

function arrow(slide, x, y, w, h, color = C.muted) {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h, line: { color, width: 1.5, beginArrowType: 'none', endArrowType: 'triangle' }
  });
}

function title(slide, main, sub) {
  text(slide, main, 0.35, 0.16, 8.5, 0.34, { size: 21, bold: true, margin: 0 });
  text(slide, sub, 0.37, 0.54, 12.9, 0.25, { size: 11, color: C.muted, margin: 0 });
}

// Figure 1: scientific teaser linking the benchmark and method.
{
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  title(slide, 'Ground, relate, act: one dependency, two interventions',
    'GSA-Forge aligns how embodied capability is measured with how specialist knowledge is transferred.');

  const stages = [
    ['G | Grounding', 'Which entities and geometry\nare present?', C.blue, C.paleBlue],
    ['S | Spatial reasoning', 'How are grounded entities\nrelated?', C.green, C.paleGreen],
    ['A | Action', 'Where and how should the\nagent interact?', C.red, C.paleRed]
  ];
  stages.forEach((s, i) => {
    const x = 0.42 + i * 4.54;
    rect(slide, x, 1.03, 3.48, 1.08, s[3], s[2], 1.2);
    text(slide, s[0], x + 0.14, 1.18, 3.20, 0.28, { size: 16, color: s[2], bold: true, align: 'center' });
    text(slide, s[1], x + 0.14, 1.54, 3.20, 0.42, { size: 11.5, color: C.ink, align: 'center' });
    if (i < stages.length - 1) arrow(slide, x + 3.52, 1.57, 0.96, 0, C.rule);
  });

  text(slide, 'upstream state', 3.78, 1.27, 1.03, 0.22, { size: 10.5, color: C.muted, align: 'center' });
  text(slide, 'upstream state', 8.31, 1.27, 1.03, 0.22, { size: 10.5, color: C.muted, align: 'center' });

  rect(slide, 0.42, 2.62, 6.22, 2.32, C.white, C.blue, 1.2);
  text(slide, 'GSA-Bench | measure the dependency', 0.70, 2.82, 5.66, 0.30,
    { size: 16, color: C.blue, bold: true });
  text(slide, 'Level-local evaluation', 0.72, 3.30, 2.42, 0.26, { size: 13.5, bold: true });
  text(slide, 'Human upstream context isolates\nwhere capability fails.', 0.72, 3.60, 2.48, 0.61,
    { size: 11.5, color: C.muted, valign: 'top' });
  text(slide, 'Chained evaluation', 3.52, 3.30, 2.42, 0.26, { size: 13.5, bold: true });
  text(slide, 'Model context reveals downstream\nerror propagation.', 3.52, 3.60, 2.48, 0.61,
    { size: 11.5, color: C.muted, valign: 'top' });
  rect(slide, 1.25, 4.36, 4.55, 0.38, C.paleBlue, C.blue, 1.0);
  text(slide, 'dependency gap = oracle-context score - predicted-context score',
    1.35, 4.36, 4.35, 0.38, { size: 10.7, color: C.blue, bold: true, align: 'center' });

  rect(slide, 7.16, 2.62, 6.22, 2.32, C.white, C.red, 1.2);
  text(slide, 'GSA-MOPD | learn along the dependency', 7.44, 2.82, 5.66, 0.30,
    { size: 16, color: C.red, bold: true });
  text(slide, 'Capability routing', 7.46, 3.30, 2.42, 0.26, { size: 13.5, bold: true });
  text(slide, 'G, S, or A specialist scores the\nstudent\'s own rollout.', 7.46, 3.60, 2.48, 0.61,
    { size: 11.5, color: C.muted, valign: 'top' });
  text(slide, 'Dependency preservation', 10.26, 3.30, 2.56, 0.26, { size: 13.5, bold: true });
  text(slide, 'Spatial and action learning retain\ntheir required upstream state.', 10.26, 3.60, 2.54, 0.61,
    { size: 11.5, color: C.muted, valign: 'top' });
  rect(slide, 7.98, 4.36, 4.55, 0.38, C.paleRed, C.red, 1.0);
  text(slide, 'balanced exposure + routed feedback + hierarchy context',
    8.08, 4.36, 4.35, 0.38, { size: 10.7, color: C.red, bold: true, align: 'center' });
}

// Figure 2: method diagram. Every component maps to the formal method text.
{
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  title(slide, 'Dependency-routed multi-teacher on-policy distillation',
    'The prediction family selects a specialist; each teacher scores the exact prefix sampled by the student.');

  rect(slide, 0.38, 1.17, 1.82, 1.00, C.light);
  text(slide, 'G/S/A example', 0.52, 1.30, 1.54, 0.25, { size: 14, bold: true, align: 'center' });
  text(slide, 'image + task\nprediction family\nupstream state', 0.55, 1.58, 1.48, 0.48, { size: 11.5, align: 'center' });

  rect(slide, 2.82, 1.10, 2.15, 1.13, C.ink, C.ink);
  text(slide, 'Student policy', 3.05, 1.29, 1.70, 0.27, { size: 17, color: C.white, bold: true, align: 'center' });
  text(slide, 'sample y ~ pi_theta(.|x)\non-policy rollout', 3.05, 1.65, 1.70, 0.42, { size: 11.5, color: C.white, align: 'center' });
  arrow(slide, 2.22, 1.66, 0.56, 0);

  rect(slide, 5.62, 1.17, 1.55, 1.00, C.light);
  text(slide, 'Router', 5.82, 1.36, 1.15, 0.27, { size: 16, bold: true, align: 'center' });
  text(slide, 'capability family', 5.82, 1.68, 1.15, 0.29, { size: 11.5, align: 'center' });
  arrow(slide, 4.99, 1.66, 0.59, 0);

  const teacherRows = [
    ['G specialist LoRA', 0.91, C.blue, C.paleBlue],
    ['S specialist LoRA', 1.55, C.green, C.paleGreen],
    ['A specialist LoRA', 2.19, C.red, C.paleRed]
  ];
  teacherRows.forEach((r) => {
    rect(slide, 8.02, r[1], 2.20, 0.46, r[3], r[2], 1.1);
    text(slide, r[0], 8.15, r[1], 1.94, 0.46, { size: 13, color: r[2], bold: true, align: 'center' });
  });
  arrow(slide, 7.19, 1.66, 0.79, -0.52, C.blue);
  arrow(slide, 7.19, 1.66, 0.79, 0.12, C.green);
  arrow(slide, 7.19, 1.66, 0.79, 0.76, C.red);

  rect(slide, 10.85, 1.00, 2.48, 1.84, C.light);
  text(slide, 'Shared frozen\nteacher backbone', 11.12, 1.28, 1.94, 0.54, { size: 16, bold: true, align: 'center' });
  text(slide, 'same image\nsame prompt\nsame generated prefix', 11.14, 1.96, 1.90, 0.60, { size: 11.5, align: 'center' });
  teacherRows.forEach((r) => arrow(slide, 10.24, r[1] + 0.23, 0.57, 0, C.rule));

  rect(slide, 4.55, 2.92, 4.25, 0.78, C.white, C.ink, 1.2);
  text(slide, 'Generalized JSD on student tokens', 4.85, 3.04, 3.65, 0.25,
    { size: 16, bold: true, align: 'center' });
  text(slide, 'teacher and student share the sampled prefix', 4.85, 3.34, 3.65, 0.20,
    { size: 11.5, color: C.muted, align: 'center' });
  arrow(slide, 12.08, 2.86, -3.24, 0.43);
  arrow(slide, 4.52, 3.29, -0.66, -1.02);

  text(slide, 'Balanced exposure', 0.42, 4.15, 1.72, 0.28, { size: 14, bold: true });
  [['G batch', C.blue, C.paleBlue], ['S batch', C.green, C.paleGreen], ['A batch', C.red, C.paleRed]].forEach((r, i) => {
    const x = 2.17 + i * 1.42;
    rect(slide, x, 4.05, 1.16, 0.46, r[2], r[1], 1.0);
    text(slide, r[0], x, 4.05, 1.16, 0.46, { size: 12, color: r[1], bold: true, align: 'center' });
  });
  text(slide, 'round-robin family-homogeneous global batches', 6.57, 4.07, 3.05, 0.38,
    { size: 11.5, color: C.muted });

  text(slide, 'Cascade context', 0.42, 4.82, 1.72, 0.28, { size: 14, bold: true });
  [['G | ground', C.blue, C.paleBlue], ['S | spatial', C.green, C.paleGreen], ['A | act', C.red, C.paleRed]].forEach((r, i) => {
    const x = 2.17 + i * 1.92;
    rect(slide, x, 4.72, 1.48, 0.46, r[2], r[1], 1.0);
    text(slide, r[0], x, 4.72, 1.48, 0.46, { size: 12.5, color: r[1], bold: true, align: 'center' });
    if (i < 2) arrow(slide, x + 1.51, 4.95, 0.36, 0, C.rule);
  });
  text(slide, 'Dependency context is preserved from grounding to spatial reasoning to action.',
    8.18, 4.72, 5.15, 0.46, { size: 11.5, color: C.muted, align: 'right' });
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
pptx.writeFile({ fileName: OUT });
console.log(OUT);
