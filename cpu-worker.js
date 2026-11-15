/**
 * cpu-worker.js
 * GPU（WebGPU / WebGL2）非対応環境およびテクスチャ上限超過時のCPUフォールバック用 Web Worker。
 * メインスレッドをブロックせずに高精細化・シャープネス・明度コントラスト・リサイズ・モザイクを実行する。
 * 処理の高速化のため、LUT事前計算・チャンネルループ展開・スクラッチバッファ再利用を行う。
 */

'use strict';

const MAX_OUTPUT_DIMENSION = 4096;

/**
 * 中間調保護ウェイト LUT (0..255)。
 * ハイライトやシャドウの白飛び・黒つぶれを抑え、中間調（128付近）にマイクロコントラストを集中させる。
 * 式: max(0.15, 1.0 - |v - 128| / 140)
 */
const MIDTONE_WEIGHT_LUT = (() => {
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const w = 1.0 - Math.abs(i - 128) / 140;
    lut[i] = w > 0.15 ? w : 0.15;
  }
  return lut;
})();

/** Worker内で再利用するスクラッチバッファ（毎回の巨大配列確保を防ぐ） */
let scratchBuffer = null;

/** OffscreenCanvas 再利用インスタンス */
let workCanvas = null;
let workCtx = null;
let stepCanvas = null;
let stepCtx = null;

/**
 * 必要バイト長のスクラッチバッファを取得または拡張する。
 * @param {number} length 必要な配列長 (width * height * 4)
 * @returns {Uint8ClampedArray}
 */
function getScratchBuffer(length) {
  if (!scratchBuffer || scratchBuffer.length !== length) {
    scratchBuffer = new Uint8ClampedArray(length);
  }
  return scratchBuffer;
}

/**
 * OffscreenCanvas 2D コンテキストが利用可能かを確認し、リサイズを実行する。
 * @param {Uint8ClampedArray} srcPixels 元画像ピクセル (RGBA)
 * @param {number} srcW 元画像幅
 * @param {number} srcH 元画像高さ
 * @param {number} targetW 出力幅
 * @param {number} targetH 出力高さ
 * @param {number} superRes 高解像度化強度 (0..100)
 * @returns {Uint8ClampedArray} リサイズ後のRGBAピクセル配列
 */
function resizePixels(srcPixels, srcW, srcH, targetW, targetH, superRes) {
  if (srcW === targetW && srcH === targetH && !(superRes >= 25 && targetW >= srcW)) {
    return new Uint8ClampedArray(srcPixels);
  }

  if (typeof OffscreenCanvas !== 'undefined') {
    if (!workCanvas) {
      workCanvas = new OffscreenCanvas(1, 1);
      workCtx = workCanvas.getContext('2d', { willReadFrequently: true });
    }
    const srcCanvas = new OffscreenCanvas(srcW, srcH);
    const srcCtx = srcCanvas.getContext('2d');
    const srcImageData = new ImageData(new Uint8ClampedArray(srcPixels), srcW, srcH);
    srcCtx.putImageData(srcImageData, 0, 0);

    workCanvas.width = targetW;
    workCanvas.height = targetH;
    workCtx.imageSmoothingEnabled = true;
    workCtx.imageSmoothingQuality = 'high';
    workCtx.clearRect(0, 0, targetW, targetH);

    // 密度補間: 拡大または等倍で超解像強度が25以上のとき、一度オーバーサンプリングしてから目標サイズに落とし輪郭を滑らかにする
    if (superRes >= 25 && targetW >= srcW) {
      const boostFactor = 1.0 + (superRes / 100) * 0.35;
      const midW = Math.min(MAX_OUTPUT_DIMENSION, Math.round(targetW * boostFactor));
      const midH = Math.min(MAX_OUTPUT_DIMENSION, Math.round(targetH * boostFactor));

      if (!stepCanvas) {
        stepCanvas = new OffscreenCanvas(1, 1);
        stepCtx = stepCanvas.getContext('2d');
      }
      stepCanvas.width = midW;
      stepCanvas.height = midH;
      stepCtx.imageSmoothingEnabled = true;
      stepCtx.imageSmoothingQuality = 'high';
      stepCtx.clearRect(0, 0, midW, midH);
      stepCtx.drawImage(srcCanvas, 0, 0, midW, midH);

      workCtx.drawImage(stepCanvas, 0, 0, targetW, targetH);
      // メモリ解放のため stepCanvas は小さく戻す
      stepCanvas.width = 1;
      stepCanvas.height = 1;
    } else {
      workCtx.drawImage(srcCanvas, 0, 0, targetW, targetH);
    }

    srcCanvas.width = 0;
    srcCanvas.height = 0;
    return workCtx.getImageData(0, 0, targetW, targetH).data;
  }

  // OffscreenCanvas 非対応環境向けの純JSバイリニア補間フォールバック
  return resizeBilinearFallback(srcPixels, srcW, srcH, targetW, targetH);
}

/**
 * OffscreenCanvas 非対応環境向けのバイリニア補間フォールバック。
 * @param {Uint8ClampedArray} src
 * @param {number} srcW
 * @param {number} srcH
 * @param {number} dstW
 * @param {number} dstH
 * @returns {Uint8ClampedArray}
 */
function resizeBilinearFallback(src, srcW, srcH, dstW, dstH) {
  const dst = new Uint8ClampedArray(dstW * dstH * 4);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    const fy = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.max(0, Math.min(srcH - 1, Math.floor(fy)));
    const y1 = Math.max(0, Math.min(srcH - 1, y0 + 1));
    const wy = fy - y0;
    const wyInv = 1 - wy;
    const row0 = y0 * srcW * 4;
    const row1 = y1 * srcW * 4;
    const dstRow = y * dstW * 4;

    for (let x = 0; x < dstW; x++) {
      const fx = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.max(0, Math.min(srcW - 1, Math.floor(fx)));
      const x1 = Math.max(0, Math.min(srcW - 1, x0 + 1));
      const wx = fx - x0;
      const wxInv = 1 - wx;

      const w00 = wxInv * wyInv;
      const w10 = wx * wyInv;
      const w01 = wxInv * wy;
      const w11 = wx * wy;

      const i00 = row0 + (x0 << 2);
      const i10 = row0 + (x1 << 2);
      const i01 = row1 + (x0 << 2);
      const i11 = row1 + (x1 << 2);
      const dIdx = dstRow + (x << 2);

      dst[dIdx] = src[i00] * w00 + src[i10] * w10 + src[i01] * w01 + src[i11] * w11;
      dst[dIdx + 1] = src[i00 + 1] * w00 + src[i10 + 1] * w10 + src[i01 + 1] * w01 + src[i11 + 1] * w11;
      dst[dIdx + 2] = src[i00 + 2] * w00 + src[i10 + 2] * w10 + src[i01 + 2] * w01 + src[i11 + 2] * w11;
      dst[dIdx + 3] = src[i00 + 3] * w00 + src[i10 + 3] * w10 + src[i01 + 3] * w01 + src[i11 + 3] * w11;
    }
  }
  return dst;
}

/**
 * 明るさ・コントラストを事前計算LUTで一括適用する。
 * コントラスト定数 259 は標準的な [-255, 255] 線形スケーリング式に基づく。
 * @param {Uint8ClampedArray} data ピクセル配列 (in-place更新)
 * @param {number} brightness -100..100
 * @param {number} contrast -100..100
 */
function applyBrightnessContrastLUT(data, brightness, contrast) {
  const lut = new Uint8ClampedArray(256);
  const cFactor = (259 * (contrast + 255)) / (255 * (259 - contrast));
  const bOffset = (brightness / 100) * 128;

  for (let i = 0; i < 256; i++) {
    lut[i] = cFactor * (i + bOffset - 128) + 128;
  }

  const len = data.length;
  for (let i = 0; i < len; i += 4) {
    if (data[i + 3] === 0) continue; // 完全透明ピクセルは色値を変更しない
    data[i] = lut[data[i]];
    data[i + 1] = lut[data[i + 1]];
    data[i + 2] = lut[data[i + 2]];
  }
}

/**
 * 3x3ガウシアン近似ローカル平均との差分を用いた高解像度・高精細化処理。
 * 透過領域境界の黒ずみフリンジを防ぐため、透明近傍画素は中心色で代替し、外周1pxもクランプ処理する。
 * @param {Uint8ClampedArray} data ピクセル配列 (in-place更新)
 * @param {number} width
 * @param {number} height
 * @param {number} superRes 0..100
 * @param {number} detail 0..100
 */
function applyHighDefinitionClarity(data, width, height, superRes, detail) {
  const src = getScratchBuffer(data.length);
  src.set(data);

  const srGain = (superRes / 100) * 0.65;
  const clarityGain = (detail / 100) * 1.15;
  const stride = width * 4;

  // 1ピクセルの処理ヘルパー（境界または透過ピクセル混在時用）
  const processBoundaryOrAlphaPixel = (x, y) => {
    const idx = y * stride + (x << 2);
    const centerA = src[idx + 3];
    if (centerA === 0) return;

    const ym1 = y > 0 ? y - 1 : 0;
    const yp1 = y < height - 1 ? y + 1 : height - 1;
    const xm1 = x > 0 ? x - 1 : 0;
    const xp1 = x < width - 1 ? x + 1 : width - 1;

    const rowM1 = ym1 * stride;
    const row0 = y * stride;
    const rowP1 = yp1 * stride;

    const i00 = rowM1 + (xm1 << 2);
    const i10 = rowM1 + (x << 2);
    const i20 = rowM1 + (xp1 << 2);
    const i01 = row0 + (xm1 << 2);
    const i21 = row0 + (xp1 << 2);
    const i02 = rowP1 + (xm1 << 2);
    const i12 = rowP1 + (x << 2);
    const i22 = rowP1 + (xp1 << 2);

    const cR = src[idx], cG = src[idx + 1], cB = src[idx + 2];

    // 近傍が完全透明な場合は中心画素色で代替し黒ずみフリンジを防止
    const r00 = src[i00 + 3] > 0 ? src[i00] : cR;
    const r10 = src[i10 + 3] > 0 ? src[i10] : cR;
    const r20 = src[i20 + 3] > 0 ? src[i20] : cR;
    const r01 = src[i01 + 3] > 0 ? src[i01] : cR;
    const r21 = src[i21 + 3] > 0 ? src[i21] : cR;
    const r02 = src[i02 + 3] > 0 ? src[i02] : cR;
    const r12 = src[i12 + 3] > 0 ? src[i12] : cR;
    const r22 = src[i22 + 3] > 0 ? src[i22] : cR;

    const g00 = src[i00 + 3] > 0 ? src[i00 + 1] : cG;
    const g10 = src[i10 + 3] > 0 ? src[i10 + 1] : cG;
    const g20 = src[i20 + 3] > 0 ? src[i20 + 1] : cG;
    const g01 = src[i01 + 3] > 0 ? src[i01 + 1] : cG;
    const g21 = src[i21 + 3] > 0 ? src[i21 + 1] : cG;
    const g02 = src[i02 + 3] > 0 ? src[i02 + 1] : cG;
    const g12 = src[i12 + 3] > 0 ? src[i12 + 1] : cG;
    const g22 = src[i22 + 3] > 0 ? src[i22 + 1] : cG;

    const b00 = src[i00 + 3] > 0 ? src[i00 + 2] : cB;
    const b10 = src[i10 + 3] > 0 ? src[i10 + 2] : cB;
    const b20 = src[i20 + 3] > 0 ? src[i20 + 2] : cB;
    const b01 = src[i01 + 3] > 0 ? src[i01 + 2] : cB;
    const b21 = src[i21 + 3] > 0 ? src[i21 + 2] : cB;
    const b02 = src[i02 + 3] > 0 ? src[i02 + 2] : cB;
    const b12 = src[i12 + 3] > 0 ? src[i12 + 2] : cB;
    const b22 = src[i22 + 3] > 0 ? src[i22 + 2] : cB;

    const meanR = (r00 + r20 + r02 + r22 + ((r10 + r12 + r01 + r21) << 1) + (cR << 2)) >> 4;
    const meanG = (g00 + g20 + g02 + g22 + ((g10 + g12 + g01 + g21) << 1) + (cG << 2)) >> 4;
    const meanB = (b00 + b20 + b02 + b22 + ((b10 + b12 + b01 + b21) << 1) + (cB << 2)) >> 4;

    const diffR = cR - meanR;
    const absR = diffR < 0 ? -diffR : diffR;
    let deltaR = 0;
    if (superRes > 0 && absR >= 2) deltaR += diffR * srGain;
    if (detail > 0 && absR >= 4) deltaR += diffR * clarityGain * MIDTONE_WEIGHT_LUT[cR];
    if (deltaR !== 0) data[idx] = cR + deltaR;

    const diffG = cG - meanG;
    const absG = diffG < 0 ? -diffG : diffG;
    let deltaG = 0;
    if (superRes > 0 && absG >= 2) deltaG += diffG * srGain;
    if (detail > 0 && absG >= 4) deltaG += diffG * clarityGain * MIDTONE_WEIGHT_LUT[cG];
    if (deltaG !== 0) data[idx + 1] = cG + deltaG;

    const diffB = cB - meanB;
    const absB = diffB < 0 ? -diffB : diffB;
    let deltaB = 0;
    if (superRes > 0 && absB >= 2) deltaB += diffB * srGain;
    if (detail > 0 && absB >= 4) deltaB += diffB * clarityGain * MIDTONE_WEIGHT_LUT[cB];
    if (deltaB !== 0) data[idx + 2] = cB + deltaB;
  };

  // 内側領域の高速処理（RGB展開 + アルファ一括判定）
  for (let y = 1; y < height - 1; y++) {
    const rowOffset = y * stride;
    for (let x = 1; x < width - 1; x++) {
      const idx = rowOffset + (x << 2);
      const cA = src[idx + 3];
      if (cA === 0) continue;

      const tl = idx - stride - 4;
      const t  = idx - stride;
      const tr = idx - stride + 4;
      const l  = idx - 4;
      const r  = idx + 4;
      const bl = idx + stride - 4;
      const b  = idx + stride;
      const br = idx + stride + 4;

      // 周囲に透明ピクセルが1つでも含まれる場合は安全な境界処理へ委譲
      if (
        src[tl + 3] === 0 || src[t + 3] === 0 || src[tr + 3] === 0 ||
        src[l + 3] === 0  || src[r + 3] === 0 ||
        src[bl + 3] === 0 || src[b + 3] === 0 || src[br + 3] === 0
      ) {
        processBoundaryOrAlphaPixel(x, y);
        continue;
      }

      // R
      const cR = src[idx];
      const meanR = (src[tl] + src[tr] + src[bl] + src[br] + ((src[t] + src[b] + src[l] + src[r]) << 1) + (cR << 2)) >> 4;
      const diffR = cR - meanR;
      const absR = diffR < 0 ? -diffR : diffR;
      let deltaR = 0;
      if (superRes > 0 && absR >= 2) deltaR += diffR * srGain;
      if (detail > 0 && absR >= 4) deltaR += diffR * clarityGain * MIDTONE_WEIGHT_LUT[cR];
      if (deltaR !== 0) data[idx] = cR + deltaR;

      // G
      const cG = src[idx + 1];
      const meanG = (src[tl + 1] + src[tr + 1] + src[bl + 1] + src[br + 1] + ((src[t + 1] + src[b + 1] + src[l + 1] + src[r + 1]) << 1) + (cG << 2)) >> 4;
      const diffG = cG - meanG;
      const absG = diffG < 0 ? -diffG : diffG;
      let deltaG = 0;
      if (superRes > 0 && absG >= 2) deltaG += diffG * srGain;
      if (detail > 0 && absG >= 4) deltaG += diffG * clarityGain * MIDTONE_WEIGHT_LUT[cG];
      if (deltaG !== 0) data[idx + 1] = cG + deltaG;

      // B
      const cB = src[idx + 2];
      const meanB = (src[tl + 2] + src[tr + 2] + src[bl + 2] + src[br + 2] + ((src[t + 2] + src[b + 2] + src[l + 2] + src[r + 2]) << 1) + (cB << 2)) >> 4;
      const diffB = cB - meanB;
      const absB = diffB < 0 ? -diffB : diffB;
      let deltaB = 0;
      if (superRes > 0 && absB >= 2) deltaB += diffB * srGain;
      if (detail > 0 && absB >= 4) deltaB += diffB * clarityGain * MIDTONE_WEIGHT_LUT[cB];
      if (deltaB !== 0) data[idx + 2] = cB + deltaB;
    }
  }

  // 外周1px枠のクランプ処理（上下端・左右端）
  for (let x = 0; x < width; x++) {
    processBoundaryOrAlphaPixel(x, 0);
    if (height > 1) processBoundaryOrAlphaPixel(x, height - 1);
  }
  for (let y = 1; y < height - 1; y++) {
    processBoundaryOrAlphaPixel(0, y);
    if (width > 1) processBoundaryOrAlphaPixel(width - 1, y);
  }
}

/**
 * 10段階シャープネス処理（8近傍ラプラシアンハイパスフィルタ）。
 * 微小ノイズ増幅を防ぐため、ハイパス振幅が ±2 を超えたエッジのみを強調する。
 * @param {Uint8ClampedArray} data ピクセル配列 (in-place更新)
 * @param {number} width
 * @param {number} height
 * @param {number} level 1..10
 */
function applySharpness10Levels(data, width, height, level) {
  const src = getScratchBuffer(data.length);
  src.set(data);

  const amountDiv6 = (level * 0.18 + (level * level) * 0.002) / 6;
  const stride = width * 4;

  const processBoundaryOrAlphaPixel = (x, y) => {
    const idx = y * stride + (x << 2);
    if (src[idx + 3] === 0) return;

    const ym1 = y > 0 ? y - 1 : 0;
    const yp1 = y < height - 1 ? y + 1 : height - 1;
    const xm1 = x > 0 ? x - 1 : 0;
    const xp1 = x < width - 1 ? x + 1 : width - 1;

    const rowM1 = ym1 * stride;
    const row0 = y * stride;
    const rowP1 = yp1 * stride;

    const i00 = rowM1 + (xm1 << 2);
    const i10 = rowM1 + (x << 2);
    const i20 = rowM1 + (xp1 << 2);
    const i01 = row0 + (xm1 << 2);
    const i21 = row0 + (xp1 << 2);
    const i02 = rowP1 + (xm1 << 2);
    const i12 = rowP1 + (x << 2);
    const i22 = rowP1 + (xp1 << 2);

    const cR = src[idx], cG = src[idx + 1], cB = src[idx + 2];

    const nR =
      (src[i10 + 3] > 0 ? src[i10] : cR) + (src[i12 + 3] > 0 ? src[i12] : cR) +
      (src[i01 + 3] > 0 ? src[i01] : cR) + (src[i21 + 3] > 0 ? src[i21] : cR) +
      (((src[i00 + 3] > 0 ? src[i00] : cR) + (src[i20 + 3] > 0 ? src[i20] : cR) +
        (src[i02 + 3] > 0 ? src[i02] : cR) + (src[i22 + 3] > 0 ? src[i22] : cR)) >> 1);
    const hpR6 = cR * 6 - nR;
    if (hpR6 > 12 || hpR6 < -12) data[idx] = cR + hpR6 * amountDiv6;

    const nG =
      (src[i10 + 3] > 0 ? src[i10 + 1] : cG) + (src[i12 + 3] > 0 ? src[i12 + 1] : cG) +
      (src[i01 + 3] > 0 ? src[i01 + 1] : cG) + (src[i21 + 3] > 0 ? src[i21 + 1] : cG) +
      (((src[i00 + 3] > 0 ? src[i00 + 1] : cG) + (src[i20 + 3] > 0 ? src[i20 + 1] : cG) +
        (src[i02 + 3] > 0 ? src[i02 + 1] : cG) + (src[i22 + 3] > 0 ? src[i22 + 1] : cG)) >> 1);
    const hpG6 = cG * 6 - nG;
    if (hpG6 > 12 || hpG6 < -12) data[idx + 1] = cG + hpG6 * amountDiv6;

    const nB =
      (src[i10 + 3] > 0 ? src[i10 + 2] : cB) + (src[i12 + 3] > 0 ? src[i12 + 2] : cB) +
      (src[i01 + 3] > 0 ? src[i01 + 2] : cB) + (src[i21 + 3] > 0 ? src[i21 + 2] : cB) +
      (((src[i00 + 3] > 0 ? src[i00 + 2] : cB) + (src[i20 + 3] > 0 ? src[i20 + 2] : cB) +
        (src[i02 + 3] > 0 ? src[i02 + 2] : cB) + (src[i22 + 3] > 0 ? src[i22 + 2] : cB)) >> 1);
    const hpB6 = cB * 6 - nB;
    if (hpB6 > 12 || hpB6 < -12) data[idx + 2] = cB + hpB6 * amountDiv6;
  };

  for (let y = 1; y < height - 1; y++) {
    const rowOffset = y * stride;
    for (let x = 1; x < width - 1; x++) {
      const idx = rowOffset + (x << 2);
      if (src[idx + 3] === 0) continue;

      const tl = idx - stride - 4;
      const t  = idx - stride;
      const tr = idx - stride + 4;
      const l  = idx - 4;
      const r  = idx + 4;
      const bl = idx + stride - 4;
      const b  = idx + stride;
      const br = idx + stride + 4;

      if (
        src[tl + 3] === 0 || src[t + 3] === 0 || src[tr + 3] === 0 ||
        src[l + 3] === 0  || src[r + 3] === 0 ||
        src[bl + 3] === 0 || src[b + 3] === 0 || src[br + 3] === 0
      ) {
        processBoundaryOrAlphaPixel(x, y);
        continue;
      }

      // highPass = (center * 6 - neighbors) / 6; |highPass| > 2 ⇔ |center * 6 - neighbors| > 12
      const cR = src[idx];
      const nR = src[t] + src[b] + src[l] + src[r] + ((src[tl] + src[tr] + src[bl] + src[br]) >> 1);
      const hpR6 = cR * 6 - nR;
      if (hpR6 > 12 || hpR6 < -12) data[idx] = cR + hpR6 * amountDiv6;

      const cG = src[idx + 1];
      const nG = src[t + 1] + src[b + 1] + src[l + 1] + src[r + 1] + ((src[tl + 1] + src[tr + 1] + src[bl + 1] + src[br + 1]) >> 1);
      const hpG6 = cG * 6 - nG;
      if (hpG6 > 12 || hpG6 < -12) data[idx + 1] = cG + hpG6 * amountDiv6;

      const cB = src[idx + 2];
      const nB = src[t + 2] + src[b + 2] + src[l + 2] + src[r + 2] + ((src[tl + 2] + src[tr + 2] + src[bl + 2] + src[br + 2]) >> 1);
      const hpB6 = cB * 6 - nB;
      if (hpB6 > 12 || hpB6 < -12) data[idx + 2] = cB + hpB6 * amountDiv6;
    }
  }

  for (let x = 0; x < width; x++) {
    processBoundaryOrAlphaPixel(x, 0);
    if (height > 1) processBoundaryOrAlphaPixel(x, height - 1);
  }
  for (let y = 1; y < height - 1; y++) {
    processBoundaryOrAlphaPixel(0, y);
    if (width > 1) processBoundaryOrAlphaPixel(width - 1, y);
  }
}

/**
 * 正規化座標で指定された複数矩形領域へブロック平均モザイクを適用する。
 * @param {Uint8ClampedArray} data ピクセル配列 (in-place更新)
 * @param {number} width
 * @param {number} height
 * @param {number} blockSize モザイク粗さ (px)
 * @param {Array<{x:number,y:number,w:number,h:number}>} regions
 */
function applyRegionalMosaic(data, width, height, blockSize, regions) {
  const step = Math.max(2, Math.round(blockSize));
  const stride = width * 4;

  for (const region of regions) {
    const startX = Math.max(0, Math.floor(region.x * width));
    const startY = Math.max(0, Math.floor(region.y * height));
    const endX = Math.min(width, Math.ceil((region.x + region.w) * width));
    const endY = Math.min(height, Math.ceil((region.y + region.h) * height));

    for (let y = startY; y < endY; y += step) {
      const bMaxY = Math.min(endY, y + step);
      for (let x = startX; x < endX; x += step) {
        const bMaxX = Math.min(endX, x + step);

        let sumR = 0;
        let sumG = 0;
        let sumB = 0;
        let sumA = 0;
        let opaqueCount = 0;
        let count = 0;

        for (let by = y; by < bMaxY; by++) {
          const rowBase = by * stride;
          for (let bx = x; bx < bMaxX; bx++) {
            const idx = rowBase + (bx << 2);
            const a = data[idx + 3];
            sumA += a;
            if (a > 0) {
              sumR += data[idx];
              sumG += data[idx + 1];
              sumB += data[idx + 2];
              opaqueCount++;
            }
            count++;
          }
        }

        if (count === 0) continue;
        const avgA = (sumA / count) | 0;
        const avgR = opaqueCount > 0 ? ((sumR / opaqueCount) | 0) : 0;
        const avgG = opaqueCount > 0 ? ((sumG / opaqueCount) | 0) : 0;
        const avgB = opaqueCount > 0 ? ((sumB / opaqueCount) | 0) : 0;

        for (let by = y; by < bMaxY; by++) {
          const rowBase = by * stride;
          for (let bx = x; bx < bMaxX; bx++) {
            const idx = rowBase + (bx << 2);
            data[idx] = avgR;
            data[idx + 1] = avgG;
            data[idx + 2] = avgB;
            data[idx + 3] = avgA;
          }
        }
      }
    }
  }
}

self.onmessage = (event) => {
  const { jobId, sourceBuffer, srcW, srcH, targetW, targetH, params, mosaicRegions } = event.data;
  try {
    const srcPixels = new Uint8ClampedArray(sourceBuffer);
    const outPixels = resizePixels(srcPixels, srcW, srcH, targetW, targetH, params.superResolution);

    if (params.brightness !== 0 || params.contrast !== 0) {
      applyBrightnessContrastLUT(outPixels, params.brightness, params.contrast);
    }

    if (params.superResolution > 0 || params.detail > 0) {
      applyHighDefinitionClarity(outPixels, targetW, targetH, params.superResolution, params.detail);
    }

    if (params.sharpness >= 1) {
      applySharpness10Levels(outPixels, targetW, targetH, params.sharpness);
    }

    if (mosaicRegions && mosaicRegions.length > 0 && params.mosaicSize >= 2) {
      applyRegionalMosaic(outPixels, targetW, targetH, params.mosaicSize, mosaicRegions);
    }

    // 結果バッファの所有権をメインスレッドへゼロコピー移譲する
    const resultBuffer = outPixels.buffer;
    self.postMessage(
      { ok: true, jobId, width: targetW, height: targetH, buffer: resultBuffer },
      [resultBuffer]
    );
  } catch (err) {
    self.postMessage({
      ok: false,
      jobId,
      error: err instanceof Error ? err.message : String(err)
    });
  }
};
