const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '../..');
const INPUT = path.join(ROOT, 'paper/figures/fig_showcase_selection.json');
const OUT = path.join(ROOT, 'paper/figures/editable/gsa_showcase.pptx');
const records = JSON.parse(fs.readFileSync(INPUT, 'utf8'));
const C = {
  ink: '202124', muted: '5F6368', blue: '0072B2', green: '009E73', red: 'D55E00',
  amber: 'E69F00', magenta: 'CC79A7', cyan: '56B4E9', white: 'FFFFFF', rule: 'B8BDC5'
};
const palette = [C.blue, C.amber, C.green, C.magenta, C.red, C.cyan];

const pptx = new pptxgen();
pptx.defineLayout({ name: 'SHOWCASE', width: 13.8, height: 6.1 });
pptx.layout = 'SHOWCASE';
pptx.author = 'GSA-Forge';
pptx.subject = 'Editable eight-form source-clean showcase';
pptx.title = 'GSA-Forge eight output interfaces';
pptx.lang = 'en-US';
pptx.theme = { headFontFace: 'Arial', bodyFontFace: 'Arial', lang: 'en-US' };

function text(slide, value, x, y, w, h, options = {}) {
  slide.addText(value, {
    x, y, w, h, margin: options.margin === undefined ? 0.04 : options.margin,
    fontFace: 'Arial', fontSize: options.size || 12, color: options.color || C.ink,
    bold: !!options.bold, align: options.align || 'left', valign: options.valign || 'mid',
    fit: 'shrink', breakLine: false,
    fill: options.fill ? { color: options.fill, transparency: options.transparency || 0 } : undefined,
    line: options.line ? { color: options.line, width: 0.8 } : undefined
  });
}

function contain(iw, ih, x, y, w, h) {
  const scale = Math.min(w / iw, h / ih);
  const dw = iw * scale;
  const dh = ih * scale;
  return { x: x + (w - dw) / 2, y: y + (h - dh) / 2, w: dw, h: dh };
}

function pos(image, point) {
  return { x: image.x + point[0] / 1000 * image.w, y: image.y + point[1] / 1000 * image.h };
}

function box(image, b) {
  const p1 = pos(image, [b[0], b[1]]);
  const p2 = pos(image, [b[2], b[3]]);
  return { x: p1.x, y: p1.y, w: p2.x - p1.x, h: p2.y - p1.y };
}

function rect(slide, b, color, width = 1.8, transparency = 100) {
  slide.addShape(pptx.ShapeType.rect, {
    x: b.x, y: b.y, w: Math.max(b.w, 0.01), h: Math.max(b.h, 0.01),
    fill: { color, transparency }, line: { color, width }
  });
}

function line(slide, x, y, w, h, color = C.red, width = 1.8) {
  slide.addShape(pptx.ShapeType.line, { x, y, w, h, line: { color, width } });
}

function dot(slide, point, color = C.red, radius = 0.10) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x: point.x - radius, y: point.y - radius, w: radius * 2, h: radius * 2,
    fill: { color }, line: { color: C.white, width: 1.2 }
  });
}

function short(value, limit = 26) {
  const s = String(value || '');
  return s.length <= limit ? s : s.slice(0, limit - 3) + '...';
}

function addOverlay(slide, record, image) {
  const items = Array.isArray(record.payload) ? record.payload : [record.payload];
  if (record.form === 'bbox') {
    const b = box(image, items[0].bbox_2d);
    rect(slide, b, C.red);
    text(slide, short(items[0].label), b.x, Math.max(image.y, b.y - 0.22), Math.max(0.65, b.w), 0.21,
      { size: 9.5, color: C.white, bold: true, fill: C.red, margin: 0.02 });
  } else if (record.form === 'detection') {
    items.forEach((item, i) => {
      const color = palette[i % palette.length];
      const b = box(image, item.bbox_2d);
      rect(slide, b, color, 1.5);
      text(slide, short(item.label, 12), b.x, Math.max(image.y, b.y - 0.17), Math.max(0.42, b.w), 0.16,
        { size: 7.5, color: C.white, bold: true, fill: color, margin: 0.01 });
    });
  } else if (record.form === 'mask') {
    const polygon = items[0].polygon;
    polygon.forEach((point, i) => {
      const a = pos(image, point);
      const b = pos(image, polygon[(i + 1) % polygon.length]);
      line(slide, a.x, a.y, b.x - a.x, b.y - a.y, C.amber, 2.1);
    });
    text(slide, short(items[0].label), image.x + 0.08, image.y + image.h - 0.28, 0.92, 0.21,
      { size: 9.5, color: C.white, bold: true, fill: C.amber, margin: 0.02 });
  } else if (record.form === 'point') {
    const p = pos(image, items[0].point_2d);
    dot(slide, p, C.red, 0.085);
    line(slide, p.x - 0.18, p.y, 0.36, 0, C.red, 1.5);
    line(slide, p.x, p.y - 0.18, 0, 0.36, C.red, 1.5);
    text(slide, short(items[0].label), Math.min(p.x + 0.10, image.x + image.w - 0.95), p.y - 0.15, 0.90, 0.22,
      { size: 8.5, color: C.white, bold: true, fill: C.red, margin: 0.02 });
  } else if (record.form === 'depth') {
    items.slice(0, 4).forEach((item, i) => {
      text(slide, `${item.rank}  ${short(item.label, 16)}  (${Number(item.relative_depth).toFixed(1)})`,
        image.x + 0.06, image.y + image.h - 0.22 * (4 - i), Math.min(1.62, image.w - 0.12), 0.20,
        { size: 8, color: C.white, bold: true, fill: palette[i], transparency: 5, margin: 0.02 });
    });
  } else if (record.form === 'affordance') {
    const b = box(image, items[0].bbox_2d);
    rect(slide, b, C.red, 2.0, 82);
    dot(slide, pos(image, items[0].contact_point_2d), C.red, 0.085);
    text(slide, `${short(items[0].affordance, 10)} @ ${short(items[0].label, 14)}`,
      image.x + 0.06, image.y + image.h - 0.26, 1.28, 0.21,
      { size: 8.5, color: C.white, bold: true, fill: C.red, margin: 0.02 });
  } else if (record.form === 'dof') {
    const item = items[0];
    const p = pos(image, item.grasp_point_2d);
    const axis = item.jaw_axis_2d;
    const norm = Math.hypot(axis[0], axis[1]) || 1;
    const ux = axis[0] / norm;
    const uy = axis[1] / norm;
    const len = 0.24;
    line(slide, p.x - ux * len, p.y - uy * len, ux * len * 2, uy * len * 2, C.red, 2.3);
    line(slide, p.x - ux * len - uy * 0.08, p.y - uy * len + ux * 0.08, uy * 0.16, -ux * 0.16, C.red, 3.0);
    line(slide, p.x + ux * len - uy * 0.08, p.y + uy * len + ux * 0.08, uy * 0.16, -ux * 0.16, C.red, 3.0);
    dot(slide, p, C.red, 0.065);
    text(slide, `${short(item.label, 14)} | ${item.approach}, wrist ${item.wrist_deg} deg`,
      image.x + 0.06, image.y + image.h - 0.26, 1.72, 0.21,
      { size: 8.2, color: C.white, bold: true, fill: C.red, margin: 0.02 });
  } else if (record.form === 'action_cot') {
    const item = items[0];
    if (item.target_point_2d) dot(slide, pos(image, item.target_point_2d), C.red, 0.095);
    const lines = [`goal: ${short(item.goal, 28)}`]
      .concat((item.steps || []).slice(0, 3).map((step, i) => `${i + 1}. ${short(step, 30)}`));
    text(slide, lines.join('\n'), image.x + 0.05, image.y + 0.06, Math.min(2.05, image.w - 0.10), 0.92,
      { size: 8.2, color: C.white, fill: '000000', transparency: 25, valign: 'top', margin: 0.04 });
  }
}

const labels = {
  bbox: 'bbox: one referent, one box', detection: 'detection: object set',
  mask: 'mask: polygon outline', point: 'point: location without extent',
  depth: 'depth: near-to-far ordering', affordance: 'affordance: sub-part and contact',
  dof: 'DoF: grasp point, jaw axis, approach', action_cot: 'action plan: ordered grounded steps'
};

const slide = pptx.addSlide();
slide.background = { color: C.white };
text(slide, 'Eight output interfaces in the source-clean generation pilot', 0.28, 0.12, 10.8, 0.34,
  { size: 20, bold: true, margin: 0 });
text(slide, 'All labels and overlays remain editable; examples are teacher-generated candidates, not benchmark ground truth.',
  0.30, 0.49, 12.7, 0.22, { size: 10.5, color: C.muted, margin: 0 });

records.forEach((record, index) => {
  const col = index % 4;
  const row = Math.floor(index / 4);
  const x = 0.28 + col * 3.38;
  const y = 0.86 + row * 2.58;
  const w = 3.08;
  const h = 2.13;
  text(slide, labels[record.form], x, y, w, 0.25, { size: 10.5, bold: true, margin: 0 });
  slide.addShape(pptx.ShapeType.rect, {
    x, y: y + 0.30, w, h, fill: { color: 'F1F3F4' }, line: { color: C.rule, width: 0.8 }
  });
  const image = contain(record.image_width, record.image_height, x + 0.02, y + 0.32, w - 0.04, h - 0.04);
  slide.addImage({ path: record.image_path, x: image.x, y: image.y, w: image.w, h: image.h });
  addOverlay(slide, record, image);
});

fs.mkdirSync(path.dirname(OUT), { recursive: true });
pptx.writeFile({ fileName: OUT });
console.log(OUT);
