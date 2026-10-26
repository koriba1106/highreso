/**
 * engine.js
 * 画像処理コアエンジン（Swin2SR ディープラーニング推論 + 高精度数式ピクセルパイプライン）
 * タイムアウト保護・オンデマンドAIロード・メモリ再利用・フォールバック機構搭載
 */

export const PRESETS = Object.freeze({
  ai_hd: {
    name: '① くっきり高精細',
    superResolution: 50,
    detail: 40,
    sharpness: 4,
    scale: 1.0,
    brightness: 3,
    contrast: 10
  },
  doc_clear: {
    name: '② 書類・文字鮮明化',
    superResolution: 65,
    detail: 60,
    sharpness: 8,
    scale: 1.0,
    brightness: 12,
    contrast: 38
  },
  portrait: {
    name: '③ ポートレート美肌',
    superResolution: 25,
    detail: 5,
    sharpness: 1,
    scale: 1.0,
    brightness: 10,
    contrast: -8
  },
  vivid: {
    name: '④ 風景・ビビッド',
    superResolution: 35,
    detail: 55,
    sharpness: 5,
    scale: 1.0,
    brightness: 5,
    contrast: 24
  },
  dark_fix: {
    name: '⑤ 暗所・逆光補正',
    superResolution: 30,
    detail: 30,
    sharpness: 3,
    scale: 1.0,
    brightness: 35,
    contrast: 15
  },
  sns_up: {
    name: '⑥ SNS用 2倍拡大',
    superResolution: 80,
    detail: 35,
    sharpness: 6,
    scale: 2.0,
    brightness: 2,
    contrast: 8
  },
  retro: {
    name: '⑦ シネマ・落ち着き',
    superResolution: 0,
    detail: 20,
    sharpness: 0,
    scale: 1.0,
    brightness: -8,
    contrast: -16
  },
  light_save: {
    name: '⑧ 軽量・縮小(0.5倍)',
    superResolution: 20,
    detail: 15,
    sharpness: 3,
    scale: 0.5,
    brightness: 0,
    contrast: 5
  }
});

const MAX_FILE_SIZE_BYTES = 35 * 1024 * 1024;
const MAX_OUTPUT_DIMENSION = 4096;
const MAX_SWIN2SR_INPUT_EDGE = 512;
const SWIN2SR_MODEL_ID = 'Xenova/swin2SR-classical-sr-x2-64';

// 永久待機（ハングアップ）を防ぐ汎用タイムアウトラッパー
function withTimeout(promise, ms, timeoutMessage = 'Timeout') {
  let timerId;
  const timeoutPromise = new Promise((_, reject) => {
    timerId = setTimeout(() => reject(new Error(timeoutMessage)), ms);
  });
  return Promise.race([promise, timeoutPromise]).finally(() => clearTimeout(timerId));
}

export class ImageCoreEngine {
  /**
   * @param {HTMLCanvasElement} targetCanvas 描画先メインキャンバス
   */
  constructor(targetCanvas) {
    if (!(targetCanvas instanceof HTMLCanvasElement)) {
      throw new Error('有効なHTMLCanvasElementが必要です');
    }
    this.canvas = targetCanvas;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

    this.rawSourceCanvas = document.createElement('canvas');
    this.rawSourceCtx = this.rawSourceCanvas.getContext('2d', { willReadFrequently: true });

    this.sourceCanvas = document.createElement('canvas');
    this.sourceCtx = this.sourceCanvas.getContext('2d', { willReadFrequently: true });

    this.workCanvas = document.createElement('canvas');
    this.workCtx = this.workCanvas.getContext('2d', { willReadFrequently: true });

    this.stepCanvas = document.createElement('canvas');
    this.stepCtx = this.stepCanvas.getContext('2d', { willReadFrequently: true });

    this._scratchBuffer = null;
    this._lastProcessedImageData = null;

    this.originalWidth = 0;
    this.originalHeight = 0;
    this.imageLoaded = false;
    this.usingAIUpscaledBase = false;

    this.mosaicRegions = [];

    this._isProcessing = false;
    this._pendingParams = null;

    this._transformersModule = null;
    this._isV3Runtime = false;
    this._swin2srPipeline = null;
    this._modelLoadPromise = null;
    // 'unloaded' | 'loading' | 'ready' | 'declined' | 'unavailable'
    this._swin2srState = 'unloaded';

    this.runtimeInfo = {
      isWebGPU: false,
      label: 'WASM High-Precision Ready'
    };
  }

  /**
   * ハードウェア性能（WebGPU対応有無）のみを1.5秒タイムアウト付きで安全に検出
   * ※外部ネットワーク通信を一切行わないため、起動時にハングアップしません
   */
  async detectHardwareCapabilities() {
    let adapterAvailable = false;
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await withTimeout(navigator.gpu.requestAdapter(), 1500, 'WebGPU timeout');
        adapterAvailable = Boolean(adapter);
      } catch (_) {
        adapterAvailable = false;
      }
    }

    this.runtimeInfo.isWebGPU = adapterAvailable;
    this.runtimeInfo.label = adapterAvailable
      ? 'WebGPU / WASM Ready'
      : 'WASM High-Precision Ready';

    return this.runtimeInfo;
  }

  getSwin2SRState() {
    return this._swin2srState;
  }

  isUsingAIUpscaledBase() {
    return this.usingAIUpscaledBase;
  }

  declineSwin2SRDownload() {
    if (this._swin2srState !== 'ready') {
      this._swin2srState = 'declined';
    }
  }

  /**
   * ユーザーが「Yes」を押した後に呼び出されるSwin2SRランタイム＆モデルのダウンロード処理
   */
  async prepareSwin2SRModel(onProgress) {
    if (this._swin2srPipeline && this._swin2srState === 'ready') {
      return true;
    }
    if (this._modelLoadPromise) {
      return this._modelLoadPromise;
    }

    this._swin2srState = 'loading';

    this._modelLoadPromise = (async () => {
      try {
        // 1. Transformers.js ランタイムの動的取得（v3公式CDN → v2フォールバック、15秒タイムアウト）
        if (!this._transformersModule) {
          try {
            const modV3 = await withTimeout(
              import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2'),
              15000,
              'CDN接続タイムアウト'
            );
            this._transformersModule = modV3;
            this._isV3Runtime = true;
          } catch (_) {
            const modV2 = await withTimeout(
              import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js'),
              15000,
              'CDN接続タイムアウト'
            );
            this._transformersModule = modV2;
            this._isV3Runtime = false;
          }

          if (this._transformersModule && this._transformersModule.env) {
            this._transformersModule.env.allowLocalModels = false;
          }
        }

        if (typeof onProgress === 'function') {
          onProgress({ status: 'runtime' });
        }

        // 2. Swin2SR モデルのダウンロード＆WASMパイプライン初期化
        const pipelineOptions = this._isV3Runtime
          ? {
              dtype: 'q8',
              device: 'wasm',
              progress_callback: (p) => { if (typeof onProgress === 'function') onProgress(p); }
            }
          : {
              quantized: true,
              progress_callback: (p) => { if (typeof onProgress === 'function') onProgress(p); }
            };

        this._swin2srPipeline = await this._transformersModule.pipeline(
          'image-to-image',
          SWIN2SR_MODEL_ID,
          pipelineOptions
        );

        this._swin2srState = 'ready';
        return true;
      } catch (err) {
        this._swin2srState = 'unavailable';
        throw err;
      } finally {
        this._modelLoadPromise = null;
      }
    })();

    return this._modelLoadPromise;
  }

  /**
   * 準備済みのSwin2SRパイプラインを用いて現在読み込まれている画像を超解像推論する
   */
  async runSwin2SRInference() {
    if (!this.imageLoaded) {
      throw new Error('画像が読み込まれていません');
    }
    if (!this._swin2srPipeline || this._swin2srState !== 'ready') {
      throw new Error('Swin2SRモデルが準備されていません');
    }

    // ブラウザのメモリ溢れ(OOM)を防ぐため入力長辺を安全サイズに収める
    let inW = this.rawSourceCanvas.width;
    let inH = this.rawSourceCanvas.height;
    if (inW > MAX_SWIN2SR_INPUT_EDGE || inH > MAX_SWIN2SR_INPUT_EDGE) {
      const ratio = Math.min(MAX_SWIN2SR_INPUT_EDGE / inW, MAX_SWIN2SR_INPUT_EDGE / inH);
      inW = Math.max(16, Math.round(inW * ratio));
      inH = Math.max(16, Math.round(inH * ratio));
    }

    const inputCanvas = document.createElement('canvas');
    inputCanvas.width = inW;
    inputCanvas.height = inH;
    const inCtx = inputCanvas.getContext('2d');
    inCtx.imageSmoothingEnabled = true;
    inCtx.imageSmoothingQuality = 'high';
    inCtx.drawImage(this.rawSourceCanvas, 0, 0, inW, inH);
    const dataUrl = inputCanvas.toDataURL('image/png');

    await new Promise(r => setTimeout(r, 40));
    const rawResult = await this._swin2srPipeline(dataUrl);
    const outputRawImage = Array.isArray(rawResult) ? rawResult[0] : rawResult;

    const outW = outputRawImage.width;
    const outH = outputRawImage.height;
    const channels = outputRawImage.channels || Math.round(outputRawImage.data.length / (outW * outH));
    const outImageData = new ImageData(outW, outH);
    const dst = outImageData.data;
    const src = outputRawImage.data;

    if (channels === 4) {
      dst.set(src);
    } else if (channels === 3) {
      let sIdx = 0;
      for (let dIdx = 0; dIdx < dst.length; dIdx += 4) {
        dst[dIdx] = src[sIdx++];
        dst[dIdx + 1] = src[sIdx++];
        dst[dIdx + 2] = src[sIdx++];
        dst[dIdx + 3] = 255;
      }
    } else {
      let sIdx = 0;
      for (let dIdx = 0; dIdx < dst.length; dIdx += 4) {
        const v = src[sIdx++];
        dst[dIdx] = v;
        dst[dIdx + 1] = v;
        dst[dIdx + 2] = v;
        dst[dIdx + 3] = 255;
      }
    }

    this.sourceCanvas.width = outW;
    this.sourceCanvas.height = outH;
    this.sourceCtx.putImageData(outImageData, 0, 0);
    this.originalWidth = outW;
    this.originalHeight = outH;
    this.usingAIUpscaledBase = true;

    inputCanvas.width = 0;
    inputCanvas.height = 0;

    return { width: outW, height: outH };
  }

  resetToOriginalSource() {
    if (!this.imageLoaded) return;
    const w = this.rawSourceCanvas.width;
    const h = this.rawSourceCanvas.height;
    this.originalWidth = w;
    this.originalHeight = h;
    this.sourceCanvas.width = w;
    this.sourceCanvas.height = h;
    this.sourceCtx.clearRect(0, 0, w, h);
    this.sourceCtx.drawImage(this.rawSourceCanvas, 0, 0);
    this.usingAIUpscaledBase = false;
  }

  async _verifyMagicBytes(file) {
    const slice = file.slice(0, 12);
    const buffer = await slice.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    if (bytes.length < 4) return false;

    if (bytes[0] === 0xFF && bytes[1] === 0xD8 && bytes[2] === 0xFF) return true;
    if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4E && bytes[3] === 0x47) return true;
    if (
      bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
      bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50
    ) return true;
    if (bytes[0] === 0x42 && bytes[1] === 0x4D) return true;

    return false;
  }

  async loadImage(file) {
    if (!file || !(file instanceof Blob)) {
      throw new Error('ファイルが選択されていません');
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      throw new Error('ファイルサイズが大きすぎます（上限 35MB）');
    }

    const isValidSignature = await this._verifyMagicBytes(file);
    if (!isValidSignature) {
      throw new Error('ファイル形式が不正、または破損しています（JPEG/PNG/WebP/BMPのみ対応）');
    }

    let bitmap = null;
    try {
      if ('createImageBitmap' in window) {
        bitmap = await createImageBitmap(file);
      } else {
        bitmap = await this._loadImageFallback(file);
      }

      if (bitmap.width < 4 || bitmap.height < 4) {
        throw new Error('画像サイズが小さすぎます（最小 4×4 px）');
      }

      let w = bitmap.width;
      let h = bitmap.height;
      if (w > MAX_OUTPUT_DIMENSION || h > MAX_OUTPUT_DIMENSION) {
        const ratio = Math.min(MAX_OUTPUT_DIMENSION / w, MAX_OUTPUT_DIMENSION / h);
        w = Math.round(w * ratio);
        h = Math.round(h * ratio);
      }

      this.originalWidth = w;
      this.originalHeight = h;

      this.rawSourceCanvas.width = w;
      this.rawSourceCanvas.height = h;
      this.rawSourceCtx.clearRect(0, 0, w, h);
      this.rawSourceCtx.drawImage(bitmap, 0, 0, w, h);

      this.sourceCanvas.width = w;
      this.sourceCanvas.height = h;
      this.sourceCtx.clearRect(0, 0, w, h);
      this.sourceCtx.drawImage(bitmap, 0, 0, w, h);

      this.imageLoaded = true;
      this.usingAIUpscaledBase = false;
      this.mosaicRegions = [];

      return { width: w, height: h };
    } finally {
      if (bitmap && typeof bitmap.close === 'function') {
        bitmap.close();
      }
    }
  }

  _loadImageFallback(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('画像のデコードに失敗しました'));
      };
      img.src = url;
    });
  }

  hasImage() {
    return this.imageLoaded;
  }

  _getScratchBuffer(byteLength) {
    if (!this._scratchBuffer || this._scratchBuffer.length !== byteLength) {
      this._scratchBuffer = new Uint8ClampedArray(byteLength);
    }
    return this._scratchBuffer;
  }

  addMosaicRegion(normRect) {
    const x = Math.max(0, Math.min(1, Number(normRect.x) || 0));
    const y = Math.max(0, Math.min(1, Number(normRect.y) || 0));
    const w = Math.max(0, Math.min(1 - x, Number(normRect.w) || 0));
    const h = Math.max(0, Math.min(1 - y, Number(normRect.h) || 0));
    if (w > 0.005 && h > 0.005) {
      this.mosaicRegions.push({ x, y, w, h });
    }
  }

  undoMosaicRegion() {
    this.mosaicRegions.pop();
  }

  clearMosaicRegions() {
    this.mosaicRegions = [];
  }

  getMosaicRegions() {
    return this.mosaicRegions;
  }

  renderOriginalPreview() {
    if (!this.imageLoaded) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.save();
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.drawImage(this.rawSourceCanvas, 0, 0, w, h);
    this.ctx.restore();
  }

  restoreProcessedPreview() {
    if (!this.imageLoaded || !this._lastProcessedImageData) return;
    this.ctx.putImageData(this._lastProcessedImageData, 0, 0);
  }

  async process(rawParams = {}) {
    if (!this.imageLoaded) {
      throw new Error('画像が読み込まれていません');
    }

    if (this._isProcessing) {
      this._pendingParams = rawParams;
      return null;
    }

    this._isProcessing = true;
    try {
      let currentParams = rawParams;
      let result = null;
      while (currentParams) {
        this._pendingParams = null;
        result = this._executePipelineSync(currentParams);
        currentParams = this._pendingParams;
      }
      return result;
    } finally {
      this._isProcessing = false;
    }
  }

  _executePipelineSync(rawParams) {
    const scale = Math.min(3.0, Math.max(0.1, Math.round((Number(rawParams.scale) || 1.0) * 10) / 10));
    const superRes = Math.min(100, Math.max(0, Math.round(Number(rawParams.superResolution) || 0)));
    const detail = Math.min(100, Math.max(0, Math.round(Number(rawParams.detail) || 0)));
    const sharpness = Math.min(10, Math.max(0, Math.round(Number(rawParams.sharpness) || 0)));
    const brightness = Math.min(100, Math.max(-100, Math.round(Number(rawParams.brightness) || 0)));
    const contrast = Math.min(100, Math.max(-100, Math.round(Number(rawParams.contrast) || 0)));
    const mosaicSize = Math.min(100, Math.max(2, Math.round(Number(rawParams.mosaicSize) || 15)));

    let targetW = Math.max(1, Math.round(this.originalWidth * scale));
    let targetH = Math.max(1, Math.round(this.originalHeight * scale));

    if (targetW > MAX_OUTPUT_DIMENSION || targetH > MAX_OUTPUT_DIMENSION) {
      const limitRatio = Math.min(MAX_OUTPUT_DIMENSION / targetW, MAX_OUTPUT_DIMENSION / targetH);
      targetW = Math.max(1, Math.round(targetW * limitRatio));
      targetH = Math.max(1, Math.round(targetH * limitRatio));
    }

    this._stepResizeAndSuperRes(targetW, targetH, superRes);

    const imageData = this.workCtx.getImageData(0, 0, targetW, targetH);
    const data = imageData.data;

    if (brightness !== 0 || contrast !== 0) {
      this._applyBrightnessContrastLUT(data, brightness, contrast);
    }

    if (superRes > 0 || detail > 0) {
      this._applyHighDefinitionClarity(data, targetW, targetH, superRes, detail);
    }

    if (sharpness >= 1) {
      this._applySharpness10Levels(data, targetW, targetH, sharpness);
    }

    if (this.mosaicRegions.length > 0 && mosaicSize >= 2) {
      this._applyRegionalMosaic(data, targetW, targetH, mosaicSize);
    }

    this.canvas.width = targetW;
    this.canvas.height = targetH;
    this.ctx.putImageData(imageData, 0, 0);
    this._lastProcessedImageData = imageData;

    return { width: targetW, height: targetH };
  }

  _stepResizeAndSuperRes(targetW, targetH, superRes) {
    this.workCanvas.width = targetW;
    this.workCanvas.height = targetH;
    this.workCtx.imageSmoothingEnabled = true;
    this.workCtx.imageSmoothingQuality = 'high';

    if (superRes >= 25 && targetW >= this.originalWidth) {
      const boostFactor = 1.0 + (superRes / 100) * 0.35;
      const midW = Math.min(MAX_OUTPUT_DIMENSION, Math.round(targetW * boostFactor));
      const midH = Math.min(MAX_OUTPUT_DIMENSION, Math.round(targetH * boostFactor));

      this.stepCanvas.width = midW;
      this.stepCanvas.height = midH;
      this.stepCtx.imageSmoothingEnabled = true;
      this.stepCtx.imageSmoothingQuality = 'high';
      this.stepCtx.clearRect(0, 0, midW, midH);
      this.stepCtx.drawImage(this.sourceCanvas, 0, 0, midW, midH);

      this.workCtx.clearRect(0, 0, targetW, targetH);
      this.workCtx.drawImage(this.stepCanvas, 0, 0, targetW, targetH);
    } else {
      this.workCtx.clearRect(0, 0, targetW, targetH);
      this.workCtx.drawImage(this.sourceCanvas, 0, 0, targetW, targetH);
    }
  }

  _applyBrightnessContrastLUT(data, brightness, contrast) {
    const lut = new Uint8ClampedArray(256);
    const cFactor = (259 * (contrast + 255)) / (255 * (259 - contrast));
    const bOffset = (brightness / 100) * 128;

    for (let i = 0; i < 256; i++) {
      lut[i] = cFactor * (i + bOffset - 128) + 128;
    }

    const len = data.length;
    for (let i = 0; i < len; i += 4) {
      data[i] = lut[data[i]];
      data[i + 1] = lut[data[i + 1]];
      data[i + 2] = lut[data[i + 2]];
    }
  }

  _applyHighDefinitionClarity(data, width, height, superRes, detail) {
    const src = this._getScratchBuffer(data.length);
    src.set(data);

    const srGain = (superRes / 100) * 0.65;
    const clarityGain = (detail / 100) * 1.15;
    const stride = width * 4;

    for (let y = 1; y < height - 1; y++) {
      const rowOffset = y * stride;
      for (let x = 1; x < width - 1; x++) {
        const idx = rowOffset + (x << 2);

        for (let c = 0; c < 3; c++) {
          const p = idx + c;
          const center = src[p];

          const localMean = (
            src[p - stride - 4] + src[p - stride + 4] +
            src[p + stride - 4] + src[p + stride + 4] +
            (src[p - stride] + src[p + stride] + src[p - 4] + src[p + 4]) * 2 +
            center * 4
          ) >> 4;

          const diff = center - localMean;
          const absDiff = diff < 0 ? -diff : diff;

          const midtoneWeight = 1.0 - Math.abs(center - 128) / 140;
          const safeWeight = midtoneWeight > 0.15 ? midtoneWeight : 0.15;

          let delta = 0;
          if (superRes > 0 && absDiff >= 2) {
            delta += diff * srGain;
          }
          if (detail > 0 && absDiff >= 4) {
            delta += diff * clarityGain * safeWeight;
          }

          if (delta !== 0) {
            const val = center + delta;
            data[p] = val < 0 ? 0 : (val > 255 ? 255 : val);
          }
        }
      }
    }
  }

  _applySharpness10Levels(data, width, height, level) {
    const src = this._getScratchBuffer(data.length);
    src.set(data);

    const amount = level * 0.18 + (level * level) * 0.002;
    const stride = width * 4;

    for (let y = 1; y < height - 1; y++) {
      const rowOffset = y * stride;
      for (let x = 1; x < width - 1; x++) {
        const idx = rowOffset + (x << 2);

        for (let c = 0; c < 3; c++) {
          const p = idx + c;
          const center = src[p];

          const neighbors =
            src[p - stride] +
            src[p + stride] +
            src[p - 4] +
            src[p + 4] +
            ((src[p - stride - 4] + src[p - stride + 4] + src[p + stride - 4] + src[p + stride + 4]) >> 1);

          const highPass = (center * 6 - neighbors) / 6;

          if (highPass > 2 || highPass < -2) {
            const out = center + highPass * amount;
            data[p] = out < 0 ? 0 : (out > 255 ? 255 : out);
          }
        }
      }
    }
  }

  _applyRegionalMosaic(data, width, height, blockSize) {
    const step = Math.max(2, Math.round(blockSize));

    for (const region of this.mosaicRegions) {
      const startX = Math.max(0, Math.floor(region.x * width));
      const startY = Math.max(0, Math.floor(region.y * height));
      const endX = Math.min(width, Math.ceil((region.x + region.w) * width));
      const endY = Math.min(height, Math.ceil((region.y + region.h) * height));

      for (let y = startY; y < endY; y += step) {
        const bMaxY = Math.min(endY, y + step);
        for (let x = startX; x < endX; x += step) {
          const bMaxX = Math.min(endX, x + step);

          let sumR = 0, sumG = 0, sumB = 0, count = 0;

          for (let by = y; by < bMaxY; by++) {
            const rowBase = by * width * 4;
            for (let bx = x; bx < bMaxX; bx++) {
              const idx = rowBase + (bx << 2);
              sumR += data[idx];
              sumG += data[idx + 1];
              sumB += data[idx + 2];
              count++;
            }
          }

          if (count === 0) continue;
          const avgR = (sumR / count) | 0;
          const avgG = (sumG / count) | 0;
          const avgB = (sumB / count) | 0;

          for (let by = y; by < bMaxY; by++) {
            const rowBase = by * width * 4;
            for (let bx = x; bx < bMaxX; bx++) {
              const idx = rowBase + (bx << 2);
              data[idx] = avgR;
              data[idx + 1] = avgG;
              data[idx + 2] = avgB;
            }
          }
        }
      }
    }
  }

  exportImageBlobUrl(mimeType = 'image/png') {
    return new Promise((resolve, reject) => {
      if (!this.imageLoaded) {
        reject(new Error('画像がありません'));
        return;
      }
      if (this._lastProcessedImageData) {
        this.ctx.putImageData(this._lastProcessedImageData, 0, 0);
      }
      this.canvas.toBlob(blob => {
        if (!blob) {
          reject(new Error('画像の書き出しに失敗しました'));
          return;
        }
        resolve(URL.createObjectURL(blob));
      }, mimeType, 0.95);
    });
  }
}