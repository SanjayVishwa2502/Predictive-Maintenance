import type React from 'react';

function downloadDataUrl(dataUrl: string, filename: string) {
  const link = document.createElement('a');
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function ensureSvgNamespaces(svgText: string) {
  let out = svgText;
  if (!out.includes('xmlns="http://www.w3.org/2000/svg"')) {
    out = out.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  if (!out.includes('xmlns:xlink="http://www.w3.org/1999/xlink"')) {
    out = out.replace('<svg', '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
  }
  return out;
}

export async function exportFirstSvgInContainerAsPng(
  containerRef: React.RefObject<HTMLElement | null>,
  filename: string,
  opts?: { backgroundColor?: string }
) {
  const container = containerRef.current;
  if (!container) return;

  const svg = container.querySelector('svg');
  if (!svg) return;

  const rect = svg.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));

  const serializer = new XMLSerializer();
  const raw = serializer.serializeToString(svg);
  const svgText = ensureSvgNamespaces(raw);

  const blob = new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);

  try {
    const image = new Image();
    const loaded = new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error('Failed to load SVG image'));
    });

    image.src = url;
    await loaded;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bg = opts?.backgroundColor;
    if (bg) {
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);
    }

    ctx.drawImage(image, 0, 0, width, height);
    const dataUrl = canvas.toDataURL('image/png');
    downloadDataUrl(dataUrl, filename);
  } finally {
    URL.revokeObjectURL(url);
  }
}
