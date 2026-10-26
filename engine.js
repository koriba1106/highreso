/**
 * engine.js
 * 画像処理コアエンジン（Swin2SR 推論 + トリミング + 高精度数式ピクセルパイプライン）
 * 初回同意設定の永続化記憶 (localStorage) ＆ 非破壊トリミング・メモリ再利用設計
 */

// 8種類の厳選ワンタッチ・プリセット定義
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

const MAX_FILE_SIZE_BYTES = 35 * 1024 * 1024; // 35MB上限
const MAX_OUTPUT_DIMENSION = 4096; // 最大4096px
const MAX_SWIN2SR_INPUT_EDGE = 640; // Swin2SR推論時のメモリ保護上限
const SWIN2SR_MODEL_ID = 'Xenova/swin2SR-classical-sr-x2-64';
const CONSENT_STORAGE_KEY = 'proimage_studio_swin2sr_consent_v1';

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

    // 1. 読み込んだ直後の純粋な元画像（全リセット・トリミング解除のマスター）
    this.rawSourceCanvas = document.createElement('canvas');
    this.rawSourceCtx = this.rawSourceCanvas.getContext('2d', { willReadFrequently: true });

    // 2. パイプラインの起点となるアクティブベース画像（トリミングやSwin2SRが適用された状態）
    this.sourceCanvas = document.createElement('canvas');
    this.sourceCtx = this.sourceCanvas.getContext('2d', { willReadFrequently: true });

    // 3. ワーク用オフスクリーンバッファ
    this.workCanvas = document.createElement('canvas');
    this.workCtx = this.workCanvas.getContext('2d', { willReadFrequently: true });

    // 4. 超解像・段階補間用の一時キャンバス
    this.stepCanvas = document.createElement('canvas');
    this.stepCtx = this.stepCanvas.getContext('2d', { willReadFrequently: true });

    this._scratchBuffer = null;
    this._lastProcessedImageData = null;

    this.originalWidth = 0;
    this.originalHeight = 0;
    this.imageLoaded = false;
    this.usingAIUpscaledBase = false;
    this.croppedState = false;

    this.mosaicRegions = [];

    this._isProcessing = false;
    this._pendingParams = null;

    this._transformersModule = null;
    this._swin2srPipeline = null;
    // 'unloaded' | 'loading' | 'ready' | 'declined' | 'unavailable'
    this._swin2srState = 'unloaded';

    this.runtimeInfo = {
      isWebGPU: false,
      ortLoaded: false,
      label: '初期化中...'
    };
  }

  // ====== 初回ダウンロード確認 (Yes/No) の記憶機能 (localStorage 安全ラッパー) ======

  /**
   * 記憶されている同意設定を取得 ('yes' | 'no' | null)
   */
  getSavedConsentPreference() {
    try {
      const val = window.localStorage.getItem(CONSENT_STORAGE_KEY);
      if (val === 'yes' || val === 'no') return val;
      return null;
    } catch (_) {
      return null;
    }
  }

  /**
   * 同意設定を記憶 ('yes' | 'no')
   */
  saveConsentPreference(pref) {
    try {
      if (pref === 'yes' || pref === 'no') {
        window.localStorage.setItem(CONSENT_STORAGE_KEY, pref);
      }
    } catch (_) {
      // プライベートブラウジング等でlocalStorageが無効な場合は無視してメモリ動作を継続
    }
  }

  /**
   * 記憶した同意設定を消去
   */
  clearConsentPreference() {
    try {
      window.localStorage.removeItem(CONSENT_STORAGE_KEY);
    } catch (_) {}
  }

  /**
   * 初回起動時にWebGPU / WASM環境を検出し、タイムアウト保護付きでAIランタイムを準備する
   */
  async initializeRuntime() {
    const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
    let adapterAvailable = false;

    if (hasWebGPU) {
      try {
        const adapter = await Promise.race([
          navigator.gpu.requestAdapter(),
          new Promise(resolve => setTimeout(() => resolve(null), 2000))
        ]);
        adapterAvailable = Boolean(adapter);
      } catch (_) {
        adapterAvailable = false;
      }
    }

    try {
      const cdnUrl = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';
      const mod = await Promise.race([
        import(cdnUrl),
        new Promise((_, reject) => setTimeout(() => reject(new Error('CDN Timeout')), 6000))
      ]);
      if (mod && mod.env) {
        mod.env.allowLocalModels = false;
        if (mod.env.backends && mod.env.backends.onnx && mod.env.backends.onnx.wasm) {
          mod.env.backends.onnx.wasm.numThreads = 1;
        }
        this._transformersModule = mod;
        this.runtimeInfo.ortLoaded = true;
      } else {
        this._swin2srState = 'unavailable';
      }
    } catch (_) {
      this.runtimeInfo.ortLoaded = false;
      this._swin2srState = 'unavailable';
    }

    this.runtimeInfo.isWebGPU = adapterAvailable;
    this.runtimeInfo.label = adapterAvailable
      ? 'WebGPU / AI Engine Ready'
      : 'WASM High-Precision Ready';

    return this.runtimeInfo;
  }

  getSwin2SRState() {
    return this._swin2srState;
  }

  isUsingAIUpscaledBase() {
    return this.usingAIUpscaledBase;
  }

  isCropped() {
    return this.croppedState;
  }

  getBaseDimensions() {
    return {
      width: this.originalWidth,
      height: this.originalHeight
    };
  }

  declineSwin2SRDownload(remember = true) {
    if (this._swin2srState !== 'ready') {
      this._swin2srState = 'declined';
    }
    if (remember) {
      this.saveConsentPreference('no');
    } else {
      this.clearConsentPreference();
    }
  }

  /**
   * Swin2SRモデルをダウンロード＆初期化する
   */
  async downloadSwin2SRModel(onProgress) {
    if (this._swin2srPipeline) {
      this._swin2srState = 'ready';
      return true;
    }
    if (!this._transformersModule || typeof this._transformersModule.pipeline !== 'function') {
      this._swin2srState = 'unavailable';
      throw new Error('AI推論ランタイムがロードされていません');
    }

    this._swin2srState = 'loading';
    try {
      this._swin2srPipeline = await this._transformersModule.pipeline(
        'image-to-image',
        SWIN2SR_MODEL_ID,
        {
          quantized: true,
          progress_callback: (prog) => {
            if (typeof onProgress === 'function') onProgress(prog);
          }
        }
      );
      this._swin2srState = 'ready';
      return true;
    } catch (err) {
      this._swin2srState = 'unavailable';
      throw new Error('Swin2SRモデルのダウンロードに失敗しました');
    }
  }

  /**
   * 現在のアクティブベース画像 (sourceCanvas) に対してSwin2SR超解像推論を実行する
   * ※トリミング済みの場合は、トリミングされた領域に対して高効率にAI推論が掛かります
   */
  async runSwin2SRSuperResolution(onProgress) {
    if (!this.imageLoaded) {
      throw new Error('画像が読み込まれていません');
    }
    if (!this._swin2srPipeline) {
      await this.downloadSwin2SRModel(onProgress);
    }

    if (typeof onProgress === 'function') {
      onProgress({ status: 'inference' });
    }

    let inW = this.sourceCanvas.width;
    let inH = this.sourceCanvas.height;
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
    inCtx.drawImage(this.sourceCanvas, 0, 0, inW, inH);
    const dataUrl = inputCanvas.toDataURL('image/png');

    await new Promise(r => setTimeout(r, 30));
    const outputRawImage = await this._swin2srPipeline(dataUrl);

    const outW = outputRawImage.width;
    const outH = outputRawImage.height;
    const channels = outputRawImage.channels || (outputRawImage.data.length / (outW * outH));
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

  // ====== トリミング（切り抜き）機能 ======

  /**
   * 正規化座標 (0.0 ~ 1.0) を受け取り、現在のアクティブベース画像を切り抜く
   */
  applyCropNormalized(normRect) {
    if (!this.imageLoaded) return;
    const srcW = this.sourceCanvas.width;
    const srcH = this.sourceCanvas.height;

    const cx = Math.max(0, Math.min(srcW - 4, Math.round(normRect.x * srcW)));
    const cy = Math.max(0, Math.min(srcH - 4, Math.round(normRect.y * srcH)));
    const cw = Math.max(4, Math.min(srcW - cx, Math.round(normRect.w * srcW)));
    const ch = Math.max(4, Math.min(srcH - cy, Math.round(normRect.h * srcH)));

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = cw;
    tempCanvas.height = ch;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.drawImage(this.sourceCanvas, cx, cy, cw, ch, 0, 0, cw, ch);

    this.sourceCanvas.width = cw;
    this.sourceCanvas.height = ch;
    this.sourceCtx.clearRect(0, 0, cw, ch);
    this.sourceCtx.drawImage(tempCanvas, 0, 0);

    this.originalWidth = cw;
    this.originalHeight = ch;
    this.croppedState = true;
    // トリミング後は座標系が変わるため既存のモザイク領域をクリア
    this.mosaicRegions = [];

    tempCanvas.width = 0;
    tempCanvas.height = 0;
  }

  /**
   * トリミングを解除し、読み込んだ直後の元画像に戻す
   */
  resetCrop() {
    this.resetToOriginalSource();
  }

  /**
   * トリミング・Swin2SR適用状態を解除し、読み込んだ直後の元画像に戻す
   */
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
    this.croppedState = false;
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
      this.croppedState = false;
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
    // トリミング済みの場合は切り抜き後の未加工ベース画像、未トリミングなら初期画像を表示
    this.ctx.drawImage(this.sourceCanvas, 0, 0, w, h);
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