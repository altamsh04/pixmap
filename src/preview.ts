import fs from 'node:fs';
import sharp from 'sharp';

// ── Terminal detection ──────────────────────────────────────────────

type Protocol = 'iterm' | 'kitty' | 'sixel' | 'halfblock';

function detectProtocol(): Protocol {
  const term = process.env.TERM_PROGRAM ?? '';
  const termEnv = process.env.TERM ?? '';

  // iTerm2, WezTerm, Hyper, Tabby, mintty all support iTerm2 inline images
  if (
    term === 'iTerm.app' ||
    term === 'WezTerm' ||
    term === 'Hyper' ||
    term === 'Tabby' ||
    process.env.WEZTERM_PANE != null ||
    process.env.MINTTY_SHORTCUT != null
  ) {
    return 'iterm';
  }

  // Kitty
  if (term === 'kitty' || process.env.KITTY_PID != null) {
    return 'kitty';
  }

  // VS Code terminal supports iTerm2 protocol
  if (term === 'vscode') {
    return 'iterm';
  }

  // Sixel support (xterm with sixel, foot, mlterm)
  if (termEnv.includes('xterm') && process.env.SIXEL_SUPPORT === '1') {
    return 'sixel';
  }

  return 'halfblock';
}

/** Get usable terminal width */
function getTermCols(): number {
  return (process.stdout.columns || 120) - 4;
}

// ── iTerm2 inline image protocol ────────────────────────────────────

async function renderIterm(imagePath: string, widthCols: number, heightRows?: number): Promise<string> {
  // Resize with sharp first to control dimensions
  const meta = await sharp(imagePath).metadata();
  const origW = meta.width ?? 400;
  const origH = meta.height ?? 400;

  // Each terminal column ≈ 8px, each row ≈ 16px (common defaults)
  // We target pixel dimensions for good quality
  const targetPxW = widthCols * 8;
  let targetPxH = Math.round(targetPxW * (origH / origW));

  if (heightRows) {
    const maxPxH = heightRows * 16;
    if (targetPxH > maxPxH) {
      targetPxH = maxPxH;
    }
  }

  const buf = await sharp(imagePath)
    .resize(targetPxW, targetPxH, { fit: 'inside', kernel: 'lanczos3' })
    .png()
    .toBuffer();

  const b64 = buf.toString('base64');
  const args = `inline=1;width=${widthCols};preserveAspectRatio=1`;
  // OSC 1337 ; File=[args]:[base64 data] ST
  return `  \x1b]1337;File=${args}:${b64}\x07`;
}

// ── Kitty graphics protocol ─────────────────────────────────────────

async function renderKitty(imagePath: string, widthCols: number, heightRows?: number): Promise<string> {
  const meta = await sharp(imagePath).metadata();
  const origW = meta.width ?? 400;
  const origH = meta.height ?? 400;

  const targetPxW = widthCols * 8;
  let targetPxH = Math.round(targetPxW * (origH / origW));

  if (heightRows) {
    const maxPxH = heightRows * 16;
    if (targetPxH > maxPxH) targetPxH = maxPxH;
  }

  const buf = await sharp(imagePath)
    .resize(targetPxW, targetPxH, { fit: 'inside', kernel: 'lanczos3' })
    .png()
    .toBuffer();

  const b64 = buf.toString('base64');
  const chunks: string[] = [];
  const CHUNK_SIZE = 4096;

  for (let i = 0; i < b64.length; i += CHUNK_SIZE) {
    const chunk = b64.slice(i, i + CHUNK_SIZE);
    const more = i + CHUNK_SIZE < b64.length ? 1 : 0;
    if (i === 0) {
      chunks.push(`\x1b_Ga=T,f=100,c=${widthCols},m=${more};${chunk}\x1b\\`);
    } else {
      chunks.push(`\x1b_Gm=${more};${chunk}\x1b\\`);
    }
  }

  return '  ' + chunks.join('');
}

// ── Half-block fallback (true-color ANSI) ───────────────────────────

const UPPER_HALF = '\u2580';
const RESET = '\x1b[0m';

function fg(r: number, g: number, b: number): string {
  return `\x1b[38;2;${r};${g};${b}m`;
}

function bgc(r: number, g: number, b: number): string {
  return `\x1b[48;2;${r};${g};${b}m`;
}

type Pixel = [number, number, number]; // r, g, b

async function loadPixels(
  imagePath: string,
  cols: number,
  maxTermRows?: number,
): Promise<{ pixels: Pixel[][]; width: number; height: number }> {
  const meta = await sharp(imagePath).metadata();
  const origW = meta.width ?? cols;
  const origH = meta.height ?? cols;
  const aspect = origH / origW;

  let pxW = cols;
  let pxH = Math.round(cols * aspect);

  if (maxTermRows && pxH > maxTermRows * 2) {
    pxH = maxTermRows * 2;
    pxW = Math.round(pxH / aspect);
  }

  pxW = Math.max(pxW, 2);
  pxH = Math.max(pxH, 2);
  // Ensure even height for half-block pairing
  if (pxH % 2 !== 0) pxH += 1;

  const { data, info } = await sharp(imagePath)
    .resize(pxW, pxH, { fit: 'fill', kernel: 'lanczos3' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixels: Pixel[][] = [];
  for (let y = 0; y < info.height; y++) {
    const row: Pixel[] = [];
    for (let x = 0; x < info.width; x++) {
      const i = (y * info.width + x) * 3;
      row.push([data[i], data[i + 1], data[i + 2]]);
    }
    pixels.push(row);
  }

  return { pixels, width: info.width, height: info.height };
}

function pixelsToAnsi(pixels: Pixel[][], indent = 2): string {
  const pad = ' '.repeat(indent);
  const lines: string[] = [];

  for (let y = 0; y < pixels.length; y += 2) {
    let line = pad;
    const topRow = pixels[y];
    const botRow = y + 1 < pixels.length ? pixels[y + 1] : null;

    for (let x = 0; x < topRow.length; x++) {
      const [tr, tg, tb] = topRow[x];
      const [br, bg, bb] = botRow ? botRow[x] : [0, 0, 0];
      line += `${fg(tr, tg, tb)}${bgc(br, bg, bb)}${UPPER_HALF}`;
    }
    line += RESET;
    lines.push(line);
  }

  return lines.join('\n');
}

async function renderHalfblock(imagePath: string, cols: number, maxTermRows?: number): Promise<string> {
  const { pixels } = await loadPixels(imagePath, cols, maxTermRows);
  return pixelsToAnsi(pixels);
}

// ── Public API ──────────────────────────────────────────────────────

const protocol = detectProtocol();

/** Render a single image to terminal string */
export async function renderImage(
  imagePath: string,
  cols?: number,
  maxTermRows?: number,
): Promise<string> {
  const w = cols ?? getTermCols();

  switch (protocol) {
    case 'iterm':
      return renderIterm(imagePath, w, maxTermRows);
    case 'kitty':
      return renderKitty(imagePath, w, maxTermRows);
    default:
      return renderHalfblock(imagePath, w, maxTermRows);
  }
}

type ImagePanel = {
  label: string;
  imagePath: string;
};

/** Render multiple images side-by-side in a row with labels on top */
export async function renderImageRow(
  panels: ImagePanel[],
  panelCols?: number,
  maxPanelRows = 24,
  gap = 2,
): Promise<string> {
  const termW = getTermCols();
  const cols = panelCols ?? Math.floor((termW - gap * (panels.length - 1)) / panels.length);

  // For inline-image protocols, render each image stacked (side-by-side isn't
  // possible with protocol images). Label + image + gap, repeated.
  if (protocol === 'iterm' || protocol === 'kitty') {
    const parts: string[] = [];
    for (const panel of panels) {
      parts.push(`  ${panel.label}`);
      const img = await renderImage(panel.imagePath, cols, maxPanelRows);
      parts.push(img);
      parts.push('');
    }
    return parts.join('\n');
  }

  // Half-block: true side-by-side rendering
  const allData = await Promise.all(
    panels.map((p) => loadPixels(p.imagePath, cols, maxPanelRows)),
  );

  const lines: string[] = [];
  const indent = '  ';
  const gapStr = ' '.repeat(gap);

  // Labels
  let labelLine = indent;
  for (let i = 0; i < panels.length; i++) {
    const label = panels[i].label;
    const padded = label.length > cols ? label.slice(0, cols - 1) + '\u2026' : label.padEnd(cols);
    labelLine += padded;
    if (i < panels.length - 1) labelLine += gapStr;
  }
  lines.push(labelLine);

  const maxPxRows = Math.max(...allData.map((d) => d.height));

  for (let y = 0; y < maxPxRows; y += 2) {
    let line = indent;

    for (let gi = 0; gi < allData.length; gi++) {
      const { pixels, width } = allData[gi];

      for (let x = 0; x < width; x++) {
        const [tr, tg, tb] = y < pixels.length ? pixels[y][x] : [0, 0, 0];
        const [br, bg, bb] = y + 1 < pixels.length ? pixels[y + 1][x] : [0, 0, 0];
        line += `${fg(tr, tg, tb)}${bgc(br, bg, bb)}${UPPER_HALF}`;
      }
      line += RESET;

      // Pad shorter panels
      const padN = cols - width;
      if (padN > 0) line += ' '.repeat(padN);
      if (gi < allData.length - 1) line += gapStr;
    }

    lines.push(line);
  }

  return lines.join('\n');
}
