/**
 * engine.js
 * 画像処理コアエンジン（Swin2SR ディープラーニング推論 + 高精度数式ピクセルパイプライン）
 * 初回同意に基づくオンデマンドAIモデル取得・フォールバック・省メモリ再利用設計
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
// Swin2SRモデル推論時のブラウザメモリクラッシュ(OOM)を防ぐ安全入力上限（長辺640px → 出力1280px）
const MAX_SWIN2SR_INPUT_EDGE = 640;
const SWIN2SR_MODEL_ID = 'Xenova/swin2SR-classical-sr-x2-64';

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

    // ユーザーが読み込んだ純粋な元画像（リセット・比較用）
    this.rawSourceCanvas = document.createElement('canvas');
    this.rawSourceCtx = this.rawSourceCanvas.getContext('2d', { willReadFrequently: true });

    // パイプラインの起点となるベース画像（通常は元画像、Swin2SR実行後はAI超解像画像）
    this.sourceCanvas = document.createElement('canvas');
    this.sourceCtx = this.sourceCanvas.getContext('2d', { willReadFrequently: true });

    // ワーク用オフスクリーンバッファ
    this.workCanvas = document.createElement('canvas');
    this.workCtx = this.workCanvas.getContext('2d', { willReadFrequently: true });

    // 超解像・段階補間用の一時キャンバス
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

    // Transformers.js モジュール参照とSwin2SRパイプライン状態
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

  /**
   * 初回起動時にWebGPU / WASM環境を検出し、Transformers.js ランタイム本体（軽量スクリプト）のみを準備
   * ※Swin2SRモデル本体（約15〜30MB）はユーザーのYes同意があるまで一切ダウンロードしません
   */
  async initializeRuntime() {
    const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
    let adapterAvailable = false;

    if (hasWebGPU) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        adapterAvailable = Boolean(adapter);
      } catch (_) {
        adapterAvailable = false;
      }
    }

    try {
      const cdnUrl = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';
      const mod = await import(cdnUrl);
      if (mod && mod.env) {
        mod.env.allowLocalModels = false;
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

  /**
   * Swin2SRモデルの現在の利用可否状態を返す
   */
  getSwin2SRState() {
    return this._swin2srState;
  }

  isUsingAIUpscaledBase() {
    return this.usingAIUpscaledBase;
  }

  /**
   * ユーザーがSwin2SRのダウンロードを「No（拒否）」した際の状態更新
   */
  declineSwin2SRDownload() {
    if (this._swin2srState !== 'ready') {
      this._swin2srState = 'declined';
    }
  }

  /**
   * ユーザーが「Yes」を選択した際にSwin2SRモデルをダウンロードし、超解像推論を実行する
   * @param {Function} onProgress 進捗コールバック
   */
  async runSwin2SRSuperResolution(onProgress) {
    if (!this.imageLoaded) {
      throw new Error('画像が読み込まれていません');
    }
    if (!this._transformersModule || typeof this._transformersModule.pipeline !== 'function') {
      this._swin2srState = 'unavailable';
      throw new Error('AI推論ランタイムが読み込まれていません');
    }

    // パイプライン未構築の場合はダウンロード＆初期化
    if (!this._swin2srPipeline) {
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
      } catch (err) {
        this._swin2srState = 'unavailable';
        throw new Error('モデルのダウンロードまたは初期化に失敗しました');
      }
    }

    if (typeof onProgress === 'function') {
      onProgress({ status: 'inference' });
    }

    // メモリ保護: 元画像が大きすぎる場合はSwin2SRの安全入力サイズに収めてから推論にかける
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

    // UIスレッドを1フレーム解放してから推論実行
    await new Promise(r => setTimeout(r, 30));
    const outputRawImage = await this._swin2srPipeline(dataUrl);

    // RawImage (RGB または RGBA) を Canvas ImageData に変換
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

    // ベース画像をSwin2SR推論結果に差し替え
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

  /**
   * Swin2SR適用状態を解除し、読み込んだ直後の元画像に戻す
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
  }

  /**
   * ファイルのバイナリヘッダー（マジックバイト）検査
   */
  async _verifyMagicBytes(file) {
    const slice = file.slice(0, 12);
    const buffer = await slice.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    if (bytes.length < 4) return false;

    // JPEG: FF D8 FF
    if (bytes[0] === 0xFF && bytes[1] === 0xD8 && bytes[2] === 0xFF) return true;
    // PNG: 89 50 4E 47
    if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4E && bytes[3] === 0x47) return true;
    // WebP: "RIFF" .... "WEBP"
    if (
      bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
      bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50
    ) return true;
    // BMP: "BM"
    if (bytes[0] === 0x42 && bytes[1] === 0x4D) return true;

    return false;
  }

  /**
   * 画像ファイルの厳格な検証とデコード読み込み
   */
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

      // 元画像の保存
      this.rawSourceCanvas.width = w;
      this.rawSourceCanvas.height = h;
      this.rawSourceCtx.clearRect(0, 0, w, h);
      this.rawSourceCtx.drawImage(bitmap, 0, 0, w, h);

      // パイプライン起点キャンバスへコピー
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

  // ====== モザイク領域管理 ======
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

  // ====== Before / After 比較プレビュー機能 ======
  renderOriginalPreview() {
    if (!this.imageLoaded) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.save();
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
    this.ctx.clearRect(0, 0, w, h);
    // 常に最初期の純粋な元画像と比較できるようにする
    this.ctx.drawImage(this.rawSourceCanvas, 0, 0, w, h);
    this.ctx.restore();
  }

  restoreProcessedPreview() {
    if (!this.imageLoaded || !this._lastProcessedImageData) return;
    this.ctx.putImageData(this._lastProcessedImageData, 0, 0);
  }

  /**
   * 画像処理パイプラインの統合実行（非同期排他ロック付き）
   */
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
    const mosaicSize = Math.min(100, Math.max(2, Math.round(Number(rawParams.mosaicSize) || 15));

    let targetW = Math.max(1, Math.round(this.originalWidth * scale));
    let targetH = Math.max(1, Math.round(this.originalHeight * scale));

    if (targetW > MAX_OUTPUT_DIMENSION || targetH > MAX_OUTPUT_DIMENSION) {
      const limitRatio = Math.min(MAX_OUTPUT_DIMENSION / targetW, MAX_OUTPUT_DIMENSION / targetH);
      targetW = Math.max(1, Math.round(targetW * limitRatio));
      targetH = Math.max(1, Math.round(targetH * limitRatio));
    }

    // 1. サイズ変更と高解像度ステップ補間 (機能① & 機能③)
    this._stepResizeAndSuperRes(targetW, targetH, superRes);

    const imageData = this.workCtx.getImageData(0, 0, targetW, targetH);
    const data = imageData.data;

    // 2. 明るさ・コントラスト調整 (機能④)
    if (brightness !== 0 || contrast !== 0) {
      this._applyBrightnessContrastLUT(data, brightness, contrast);
    }

    // 3. 高解像度ディテール復元 + 高精細化マイクロコントラスト強調 (機能① & 機能②)
    if (superRes > 0 || detail > 0) {
      this._applyHighDefinitionClarity(data, targetW, targetH, superRes, detail);
    }

    // 4. シャープネス 1〜10段階 (3x3 ラプラシアン・ハイパス畳み込み)
    if (sharpness >= 1) {
      this._applySharpness10Levels(data, targetW, targetH, sharpness);
    }

    // 5. 範囲指定モザイク処理 (機能⑤)
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