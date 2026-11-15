/**
 * engine.js
 * 画像処理コアエンジン（非破壊資産管理 + GPU優先パイプライン + Swin2SR v3 AI超解像）
 *
 * 処理バックエンド優先順位:
 *   - 通常画像処理: WebGPU (WGSL) → WebGL2 (GLSL ES 3.00) → Web Worker CPU (最適化LUT/アンローリング)
 *   - Swin2SR AI超解像 (@huggingface/transformers v3): WebGPU (fp16/fp32) → WASM (q8)
 */

import { GPURenderer } from './gpu-renderer.js';

/** 8種類の厳選ワンタッチ・プリセット定義 */
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

export const MAX_FILE_SIZE_BYTES = 35 * 1024 * 1024; // 35MB上限
export const MAX_OUTPUT_DIMENSION = 4096; // 出力・内部保持の最大長辺
export const MAX_SWIN2SR_INPUT_EDGE = 640; // Swin2SR ブラウザ推論時の安全入力上限
const SWIN2SR_MODEL_ID = 'Xenova/swin2SR-classical-sr-x2-64';
const CONSENT_STORAGE_KEY = 'proimage_studio_swin2sr_consent_v1';
const LOCAL_TRANSFORMERS_URL = './libs/transformers.min.js';
const CDN_TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';

export class ImageCoreEngine {
  /**
   * @param {HTMLCanvasElement} targetCanvas プレビューおよび書き出し用のメイン2Dキャンバス
   */
  constructor(targetCanvas) {
    if (!(targetCanvas instanceof HTMLCanvasElement)) {
      throw new Error('有効なHTMLCanvasElementが必要です');
    }

    this.canvas = targetCanvas;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

    // 1. 読み込んだ直後の純粋な元画像（最大4096px・比較表示や全リセットのマスター）
    this.rawCanvas = document.createElement('canvas');
    this.rawCtx = this.rawCanvas.getContext('2d', { willReadFrequently: true });

    // 2. トリミング適用後の一時ベース画像（未トリミング時はサイズ0でメモリを消費しない）
    this.croppedCanvas = document.createElement('canvas');
    this.croppedCtx = this.croppedCanvas.getContext('2d', { willReadFrequently: true });

    // 3. Swin2SR AI超解像適用後のベース画像（未実行時はサイズ0でメモリを消費しない）
    this.aiBaseCanvas = document.createElement('canvas');
    this.aiBaseCtx = this.aiBaseCanvas.getContext('2d', { willReadFrequently: true });

    this.imageLoaded = false;
    this.rawWidth = 0;
    this.rawHeight = 0;
    this.sourceVersion = 1; // ベース画像変更時にインクリメント（GPUテクスチャ再転送判定用）

    /** @type {{x:number, y:number, w:number, h:number} | null} 元画像に対する正規化トリミング矩形 */
    this.cropNormRect = null;
    this.hasAIUpscale = false;
    this.aiEnabled = false;

    /** @type {Array<{x:number, y:number, w:number, h:number}>} アクティブベースに対する正規化モザイク領域 */
    this.mosaicRegions = [];

    // 非同期パイプライン制御（世代管理＆最新パラメータ追従）
    this._isProcessing = false;
    this._pendingParams = null;
    this._imageGeneration = 0;
    this._aiJobSeq = 0;

    // GPUレンダラー ＆ CPUフォールバック Worker
    this._gpuRenderer = new GPURenderer();
    this._worker = null;
    this._workerJobSeq = 0;
    this._workerCallbacks = new Map();

    // Swin2SR ランタイム状態
    this._transformersModule = null;
    this._swin2srPipeline = null;
    this._swin2srLoadPromise = null;
    /** @type {'unloaded' | 'loading' | 'ready' | 'declined' | 'unavailable'} */
    this._swin2srState = 'unloaded';
    this._swin2srActiveDtype = null;

    this.runtimeInfo = {
      renderBackend: 'cpu-worker',
      aiBackend: 'unavailable',
      webgpuF16: false,
      ortLoaded: false,
      label: 'エンジン初期化中...'
    };
  }

  // ====== 初回ダウンロード確認 (Yes/No) の永続化 ======

  /**
   * 記憶されている同意設定を取得 ('yes' | 'no' | null)
   * @returns {'yes'|'no'|null}
   */
  getSavedConsentPreference() {
    try {
      const val = window.localStorage.getItem(CONSENT_STORAGE_KEY);
      return (val === 'yes' || val === 'no') ? val : null;
    } catch (_) {
      return null;
    }
  }

  /**
   * 同意設定を記憶 ('yes' | 'no')
   * @param {'yes'|'no'} pref
   */
  saveConsentPreference(pref) {
    try {
      if (pref === 'yes' || pref === 'no') {
        window.localStorage.setItem(CONSENT_STORAGE_KEY, pref);
      }
    } catch (_) {
      // プライベートブラウジング等で無効な場合はメモリ動作のみ継続
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

  // ====== ランタイム初期化 (GPUプローブ + CPU Worker + Transformers.js v3) ======

  /**
   * GPUパイプライン（WebGPU/WebGL2）、CPUフォールバックWorker、および Transformers.js v3 を初期化する。
   * @returns {Promise<object>} runtimeInfo
   */
  async initializeRuntime() {
    const gpuCaps = await this._gpuRenderer.init();
    this._initWorker();

    if (gpuCaps.backend === 'webgpu') {
      this.runtimeInfo.renderBackend = 'webgpu';
    } else if (gpuCaps.backend === 'webgl2') {
      this.runtimeInfo.renderBackend = 'webgl2';
    } else {
      this.runtimeInfo.renderBackend = 'cpu-worker';
    }
    this.runtimeInfo.webgpuF16 = gpuCaps.supportsShaderF16;

    await this._loadTransformersRuntime();
    this._updateRuntimeLabel();
    return this.runtimeInfo;
  }

  /**
   * ローカル libs/ を最優先し、失敗時のみ CDN から Transformers.js v3 を読み込む。
   */
  async _loadTransformersRuntime() {
    if (this._transformersModule) return true;

    let mod = null;
    try {
      mod = await import(LOCAL_TRANSFORMERS_URL);
    } catch (_) {
      try {
        mod = await Promise.race([
          import(CDN_TRANSFORMERS_URL),
          new Promise((_, reject) => setTimeout(() => reject(new Error('CDN Timeout')), 8000))
        ]);
      } catch (_) {
        mod = null;
      }
    }

    if (mod && mod.env && typeof mod.pipeline === 'function') {
      mod.env.allowLocalModels = false;
      if (mod.env.backends?.onnx?.wasm) {
        // ローカル配置された v3 互換 JSEP / WASM バイナリのディレクトリを指定
        mod.env.backends.onnx.wasm.wasmPaths = new URL('./libs/', import.meta.url).href;
        // SharedArrayBuffer が使えない通常環境では明示的に numThreads = 1 にしてフォールバック遅延を防ぐ
        const canMultiThread = typeof self !== 'undefined' && Boolean(self.crossOriginIsolated);
        mod.env.backends.onnx.wasm.numThreads = canMultiThread
          ? Math.min(4, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2)
          : 1;
      }
      this._transformersModule = mod;
      this.runtimeInfo.ortLoaded = true;
      this.runtimeInfo.aiBackend = this.runtimeInfo.renderBackend === 'webgpu' ? 'webgpu' : 'wasm';
      return true;
    }

    this.runtimeInfo.ortLoaded = false;
    this.runtimeInfo.aiBackend = 'unavailable';
    this._swin2srState = 'unavailable';
    return false;
  }

  _updateRuntimeLabel() {
    const renderMap = {
      webgpu: 'GPU (WebGPU)',
      webgl2: 'GPU (WebGL2)',
      'cpu-worker': 'CPU (Worker)'
    };
    const rLabel = renderMap[this.runtimeInfo.renderBackend] || 'CPU (Worker)';
    const aiTag = this.runtimeInfo.aiBackend === 'webgpu'
      ? 'AI: WebGPU'
      : (this.runtimeInfo.aiBackend === 'wasm' ? 'AI: WASM' : 'AI: 待機');
    this.runtimeInfo.label = `${rLabel} / ${aiTag}`;
  }

  _initWorker() {
    if (this._worker) return;
    try {
      this._worker = new Worker('./cpu-worker.js');
      this._worker.onmessage = (event) => {
        const msg = event.data;
        const cb = this._workerCallbacks.get(msg.jobId);
        if (!cb) return;
        this._workerCallbacks.delete(msg.jobId);
        if (msg.ok) {
          cb.resolve(msg);
        } else {
          cb.reject(new Error(msg.error || 'Worker処理エラー'));
        }
      };
      this._worker.onerror = (err) => {
        for (const [, cb] of this._workerCallbacks) {
          cb.reject(new Error(err.message || 'Worker実行エラー'));
        }
        this._workerCallbacks.clear();
        try { this._worker.terminate(); } catch (_) {}
        this._worker = null;
      };
    } catch (_) {
      this._worker = null;
    }
  }

  // ====== 状態参照ヘルパー ======

  getSwin2SRState() {
    return this._swin2srState;
  }

  hasAIUpscaleCache() {
    return this.hasAIUpscale;
  }

  isAIUpscaleActive() {
    return this.hasAIUpscale && this.aiEnabled;
  }

  isCropped() {
    return this.cropNormRect !== null;
  }

  hasImage() {
    return this.imageLoaded;
  }

  /**
   * 現在のアクティブベース画像キャンバス（AI適用 ＞ トリミング後 ＞ 元画像）を返す。
   * @returns {HTMLCanvasElement}
   */
  getActiveBaseCanvas() {
    if (this.hasAIUpscale && this.aiEnabled && this.aiBaseCanvas.width > 0) {
      return this.aiBaseCanvas;
    }
    return this.getPreAIBaseCanvas();
  }

  /**
   * AI適用前（トリミング適用済みまたは元画像）のベースキャンバスを返す（「元画像と比較」の基準）。
   * @returns {HTMLCanvasElement}
   */
  getPreAIBaseCanvas() {
    if (this.cropNormRect && this.croppedCanvas.width > 0) {
      return this.croppedCanvas;
    }
    return this.rawCanvas;
  }

  getActiveBaseDimensions() {
    const base = this.getActiveBaseCanvas();
    return { width: base.width, height: base.height };
  }

  getPreAIBaseDimensions() {
    const base = this.getPreAIBaseCanvas();
    return { width: base.width, height: base.height };
  }

  // ====== Swin2SR ダウンロード・推論制御 ======

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
   * Swin2SRモデルをダウンロード・初期化する。
   * WebGPU (fp16 → fp32) を最優先で試行し、非対応や例外時は WASM (q8) へ自動フォールバックする。
   * @param {(prog: object) => void} [onProgress]
   * @returns {Promise<boolean>}
   */
  async downloadSwin2SRModel(onProgress) {
    if (this._swin2srPipeline) {
      this._swin2srState = 'ready';
      return true;
    }
    if (this._swin2srLoadPromise) {
      return this._swin2srLoadPromise;
    }

    const loaded = await this._loadTransformersRuntime();
    if (!loaded || !this._transformersModule) {
      this._swin2srState = 'unavailable';
      throw new Error('AI推論ランタイムを読み込めませんでした');
    }

    this._swin2srState = 'loading';
    const fileTotals = new Map();
    const fileLoaded = new Map();

    const wrappedProgress = (prog) => {
      if (typeof onProgress !== 'function') return;
      if (prog && prog.status === 'progress' && prog.file) {
        if (typeof prog.total === 'number' && prog.total > 0) {
          fileTotals.set(prog.file, prog.total);
          fileLoaded.set(prog.file, prog.loaded || 0);
          let sumTotal = 0;
          let sumLoaded = 0;
          for (const [f, t] of fileTotals) {
            sumTotal += t;
            sumLoaded += fileLoaded.get(f) || 0;
          }
          const combinedPct = sumTotal > 0 ? Math.min(99, Math.round((sumLoaded / sumTotal) * 100)) : Math.round(prog.progress || 0);
          onProgress({ ...prog, progress: combinedPct });
          return;
        }
      }
      onProgress(prog);
    };

    const loadTask = (async () => {
      const candidates = [];
      if (this.runtimeInfo.renderBackend === 'webgpu') {
        if (this.runtimeInfo.webgpuF16) {
          candidates.push({ device: 'webgpu', dtype: 'fp16' });
        }
        candidates.push({ device: 'webgpu', dtype: 'fp32' });
      }
      candidates.push({ device: 'wasm', dtype: 'q8' });

      let lastErr = null;
      for (const opt of candidates) {
        try {
          const pipe = await this._transformersModule.pipeline(
            'image-to-image',
            SWIN2SR_MODEL_ID,
            {
              device: opt.device,
              dtype: opt.dtype,
              progress_callback: wrappedProgress
            }
          );
          this._swin2srPipeline = pipe;
          this._swin2srState = 'ready';
          this._swin2srActiveDtype = opt.dtype;
          this.runtimeInfo.aiBackend = opt.device;
          this._updateRuntimeLabel();
          return true;
        } catch (err) {
          lastErr = err;
          fileTotals.clear();
          fileLoaded.clear();
        }
      }

      this._swin2srState = 'unavailable';
      this.runtimeInfo.aiBackend = 'unavailable';
      this._updateRuntimeLabel();
      throw new Error(lastErr?.message || 'Swin2SRモデルのダウンロードに失敗しました');
    })();

    this._swin2srLoadPromise = loadTask;
    try {
      return await loadTask;
    } finally {
      this._swin2srLoadPromise = null;
    }
  }

  /**
   * 現在の未AIベース画像に対し、Swin2SR入力上限 (MAX_SWIN2SR_INPUT_EDGE) を超えるか事前判定する。
   * @returns {{needsShrinkConfirm:boolean, baseW:number, baseH:number, inputW:number, inputH:number, estimatedOutW:number, estimatedOutH:number, willReduceResolution:boolean}}
   */
  getSwin2SRInputPlan() {
    const base = this.getPreAIBaseCanvas();
    const baseW = base.width;
    const baseH = base.height;
    let inputW = baseW;
    let inputH = baseH;
    let needsShrinkConfirm = false;

    if (baseW > MAX_SWIN2SR_INPUT_EDGE || baseH > MAX_SWIN2SR_INPUT_EDGE) {
      needsShrinkConfirm = true;
      const ratio = Math.min(MAX_SWIN2SR_INPUT_EDGE / baseW, MAX_SWIN2SR_INPUT_EDGE / baseH);
      inputW = Math.max(16, Math.round(baseW * ratio));
      inputH = Math.max(16, Math.round(baseH * ratio));
    }

    const estimatedOutW = inputW * 2;
    const estimatedOutH = inputH * 2;
    const willReduceResolution = estimatedOutW < baseW || estimatedOutH < baseH;

    return {
      needsShrinkConfirm,
      baseW,
      baseH,
      inputW,
      inputH,
      estimatedOutW,
      estimatedOutH,
      willReduceResolution
    };
  }

  /**
   * 現在の未AIベース画像に対して Swin2SR 超解像推論を実行し、AIベース画像を生成する。
   * WebGPU推論時に失敗した場合は自動で WASM バックエンドで再試行する。
   * @param {(prog: object) => void} [onProgress]
   * @returns {Promise<{width:number, height:number, cancelled?:boolean}>}
   */
  async runSwin2SRSuperResolution(onProgress) {
    if (!this.imageLoaded) {
      throw new Error('画像が読み込まれていません');
    }

    const genAtStart = this._imageGeneration;
    const myJobSeq = ++this._aiJobSeq;

    if (!this._swin2srPipeline) {
      await this.downloadSwin2SRModel(onProgress);
    }

    if (genAtStart !== this._imageGeneration || myJobSeq !== this._aiJobSeq) {
      return { width: 0, height: 0, cancelled: true };
    }

    if (typeof onProgress === 'function') {
      onProgress({ status: 'inference' });
    }

    const sourceBase = this.getPreAIBaseCanvas();
    const plan = this.getSwin2SRInputPlan();
    const inW = plan.inputW;
    const inH = plan.inputH;

    // 入力キャンバスへスケーリングしてピクセルデータを抽出（toDataURLのPNGエンコードを排除）
    const inputCanvas = document.createElement('canvas');
    inputCanvas.width = inW;
    inputCanvas.height = inH;
    const inCtx = inputCanvas.getContext('2d', { willReadFrequently: true });
    inCtx.imageSmoothingEnabled = true;
    inCtx.imageSmoothingQuality = 'high';
    inCtx.drawImage(sourceBase, 0, 0, inW, inH);
    const inImageData = inCtx.getImageData(0, 0, inW, inH);
    const rgba = inImageData.data;

    // 透過の有無を検出しつつ RGB 3ch バッファを作成
    const rgbBytes = new Uint8ClampedArray(inW * inH * 3);
    let hasTransparency = false;
    for (let s = 0, d = 0; s < rgba.length; s += 4, d += 3) {
      rgbBytes[d] = rgba[s];
      rgbBytes[d + 1] = rgba[s + 1];
      rgbBytes[d + 2] = rgba[s + 2];
      if (rgba[s + 3] < 255) hasTransparency = true;
    }

    const { RawImage } = this._transformersModule;
    const rawInput = new RawImage(rgbBytes, inW, inH, 3);

    // UIスレッドでスピナーを描画させるため1フレーム譲る
    await new Promise(r => setTimeout(r, 20));

    let outputRawImage = null;
    try {
      outputRawImage = await this._swin2srPipeline(rawInput);
    } catch (err) {
      // WebGPU推論実行時に演算子エラー等が起きた場合は WASM (q8) へ安全にフォールバックして再実行
      if (this.runtimeInfo.aiBackend === 'webgpu') {
        this._swin2srPipeline = null;
        this._swin2srPipeline = await this._transformersModule.pipeline(
          'image-to-image',
          SWIN2SR_MODEL_ID,
          { device: 'wasm', dtype: 'q8', progress_callback: onProgress }
        );
        this.runtimeInfo.aiBackend = 'wasm';
        this._swin2srActiveDtype = 'q8';
        this._updateRuntimeLabel();
        outputRawImage = await this._swin2srPipeline(rawInput);
      } else {
        throw err;
      }
    }

    // 推論待機中に画像が切り替わったりキャンセルされた場合は結果を破棄
    if (genAtStart !== this._imageGeneration || myJobSeq !== this._aiJobSeq) {
      inputCanvas.width = 0;
      inputCanvas.height = 0;
      return { width: 0, height: 0, cancelled: true };
    }

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

    // 透過画像だった場合、入力画像のアルファチャンネルを出力解像度へ補間して復元する
    if (hasTransparency) {
      const alphaCanvas = document.createElement('canvas');
      alphaCanvas.width = outW;
      alphaCanvas.height = outH;
      const aCtx = alphaCanvas.getContext('2d', { willReadFrequently: true });
      aCtx.imageSmoothingEnabled = true;
      aCtx.imageSmoothingQuality = 'high';
      aCtx.drawImage(inputCanvas, 0, 0, outW, outH);
      const alphaData = aCtx.getImageData(0, 0, outW, outH).data;
      for (let i = 3; i < dst.length; i += 4) {
        dst[i] = alphaData[i];
      }
      alphaCanvas.width = 0;
      alphaCanvas.height = 0;
    }

    this.aiBaseCanvas.width = outW;
    this.aiBaseCanvas.height = outH;
    this.aiBaseCtx.putImageData(outImageData, 0, 0);
    this.hasAIUpscale = true;
    this.aiEnabled = true;
    this.sourceVersion++;
    this._gpuRenderer.invalidateSourceTexture();

    inputCanvas.width = 0;
    inputCanvas.height = 0;

    return { width: outW, height: outH };
  }

  /**
   * 生成済み Swin2SR ベースの ON/OFF を非破壊で切り替える。
   * @param {boolean} enabled
   */
  setAIUpscaleEnabled(enabled) {
    if (!this.hasAIUpscale) return false;
    const next = Boolean(enabled);
    if (this.aiEnabled !== next) {
      this.aiEnabled = next;
      this.sourceVersion++;
      this._gpuRenderer.invalidateSourceTexture();
    }
    return this.aiEnabled;
  }

  /**
   * AIベースキャッシュを破棄する（トリミング範囲変更時など）。
   */
  clearAIUpscaleCache() {
    this.hasAIUpscale = false;
    this.aiEnabled = false;
    this.aiBaseCanvas.width = 0;
    this.aiBaseCanvas.height = 0;
    this.sourceVersion++;
    this._gpuRenderer.invalidateSourceTexture();
  }

  // ====== トリミング（非破壊管理） ======

  /**
   * アクティブベースに対する正規化座標から、元画像 (rawCanvas) 基準のトリミングを適用する。
   * トリミングは常に元画像から再切り出しされるため、何度やり直しても画質劣化しない。
   * @param {{x:number, y:number, w:number, h:number}} normRect 現プレビュー/ベースに対する正規化矩形
   * @returns {{aiReset:boolean, mosaicCleared:boolean}}
   */
  applyCropNormalized(normRect) {
    if (!this.imageLoaded) return { aiReset: false, mosaicCleared: false };

    // 現在のトリミング枠（未適用なら全体 0..1）の中での相対範囲を元画像座標へ合成
    const baseRect = this.cropNormRect || { x: 0, y: 0, w: 1, h: 1 };
    const nextNorm = {
      x: Math.max(0, Math.min(1, baseRect.x + normRect.x * baseRect.w)),
      y: Math.max(0, Math.min(1, baseRect.y + normRect.y * baseRect.h)),
      w: Math.max(0, Math.min(1, normRect.w * baseRect.w)),
      h: Math.max(0, Math.min(1, normRect.h * baseRect.h))
    };

    const rw = this.rawCanvas.width;
    const rh = this.rawCanvas.height;
    const minEdge = Math.min(4, rw, rh);
    const cx = Math.max(0, Math.min(rw - minEdge, Math.round(nextNorm.x * rw)));
    const cy = Math.max(0, Math.min(rh - minEdge, Math.round(nextNorm.y * rh)));
    const cw = Math.max(minEdge, Math.min(rw - cx, Math.round(nextNorm.w * rw)));
    const ch = Math.max(minEdge, Math.min(rh - cy, Math.round(nextNorm.h * rh)));

    this.cropNormRect = {
      x: cx / rw,
      y: cy / rh,
      w: cw / rw,
      h: ch / rh
    };

    this.croppedCanvas.width = cw;
    this.croppedCanvas.height = ch;
    this.croppedCtx.clearRect(0, 0, cw, ch);
    this.croppedCtx.drawImage(this.rawCanvas, cx, cy, cw, ch, 0, 0, cw, ch);

    const aiReset = this.hasAIUpscale;
    if (aiReset) {
      this.clearAIUpscaleCache();
    }

    const mosaicCleared = this.mosaicRegions.length > 0;
    this.mosaicRegions = [];
    this.sourceVersion++;
    this._gpuRenderer.invalidateSourceTexture();

    return { aiReset, mosaicCleared };
  }

  /**
   * トリミングを解除し、元の画像全体へ戻す。
   * 座標系が変わるため既存モザイクとAIキャッシュを整理する。
   * @returns {{aiReset:boolean, mosaicCleared:boolean}}
   */
  resetCrop() {
    if (!this.imageLoaded) return { aiReset: false, mosaicCleared: false };
    this.cropNormRect = null;
    this.croppedCanvas.width = 0;
    this.croppedCanvas.height = 0;

    const aiReset = this.hasAIUpscale;
    if (aiReset) {
      this.clearAIUpscaleCache();
    }

    const mosaicCleared = this.mosaicRegions.length > 0;
    this.mosaicRegions = [];
    this.sourceVersion++;
    this._gpuRenderer.invalidateSourceTexture();

    return { aiReset, mosaicCleared };
  }

  /**
   * トリミング・AI適用・モザイクすべてを解除し、読込直後の状態へ戻す。
   */
  resetToOriginalSource() {
    if (!this.imageLoaded) return;
    this._aiJobSeq++; // 進行中のAI推論があれば無効化
    this.cropNormRect = null;
    this.croppedCanvas.width = 0;
    this.croppedCanvas.height = 0;
    this.clearAIUpscaleCache();
    this.mosaicRegions = [];
    this.sourceVersion++;
    this._gpuRenderer.invalidateSourceTexture();
  }

  // ====== 画像読み込みと署名検証 ======

  /**
   * 先頭バイト列（マジックナンバー）から画像ファイル形式の妥当性を検証する。
   * @param {Blob} file
   * @returns {Promise<boolean>}
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
    // WebP: RIFF .... WEBP
    if (
      bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
      bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50
    ) return true;
    // BMP: 42 4D ("BM")
    if (bytes[0] === 0x42 && bytes[1] === 0x4D) return true;

    return false;
  }

  /**
   * 画像ファイルを検証・デコードし、エンジンへ読み込む（最大長辺4096px制限付き）。
   * @param {File|Blob} file
   * @returns {Promise<{width:number, height:number, downscaledFrom?:{width:number,height:number}}>}
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

      const origW = bitmap.width;
      const origH = bitmap.height;
      let w = origW;
      let h = origH;
      let downscaledFrom = null;

      if (w > MAX_OUTPUT_DIMENSION || h > MAX_OUTPUT_DIMENSION) {
        const ratio = Math.min(MAX_OUTPUT_DIMENSION / w, MAX_OUTPUT_DIMENSION / h);
        w = Math.round(w * ratio);
        h = Math.round(h * ratio);
        downscaledFrom = { width: origW, height: origH };
      }

      this._imageGeneration++;
      this._aiJobSeq++;

      this.rawWidth = w;
      this.rawHeight = h;
      this.rawCanvas.width = w;
      this.rawCanvas.height = h;
      this.rawCtx.clearRect(0, 0, w, h);
      this.rawCtx.drawImage(bitmap, 0, 0, w, h);

      this.croppedCanvas.width = 0;
      this.croppedCanvas.height = 0;
      this.aiBaseCanvas.width = 0;
      this.aiBaseCanvas.height = 0;

      this.imageLoaded = true;
      this.cropNormRect = null;
      this.hasAIUpscale = false;
      this.aiEnabled = false;
      this.mosaicRegions = [];
      this.sourceVersion++;
      this._gpuRenderer.invalidateSourceTexture();

      return { width: w, height: h, downscaledFrom };
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

  // ====== モザイク領域管理 ======

  addMosaicRegion(normRect) {
    const x = Math.max(0, Math.min(1, Number(normRect.x) || 0));
    const y = Math.max(0, Math.min(1, Number(normRect.y) || 0));
    const w = Math.max(0, Math.min(1 - x, Number(normRect.w) || 0));
    const h = Math.max(0, Math.min(1 - y, Number(normRect.h) || 0));
    if (w > 0.005 && h > 0.005) {
      this.mosaicRegions.push({ x, y, w, h });
      return this.mosaicRegions.length - 1;
    }
    return -1;
  }

  removeMosaicRegionAt(index) {
    if (index >= 0 && index < this.mosaicRegions.length) {
      this.mosaicRegions.splice(index, 1);
      return true;
    }
    return false;
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

  // ====== 画像処理パイプライン調停 (GPU優先 + Worker CPUフォールバック) ======

  /**
   * 調整パラメータとモザイクを非同期に適用し、メインキャンバスへ描画する。
   * 連続呼び出し時は実行中の1ジョブ完了後に最新パラメータのみを処理する（デバウンス結合）。
   * @param {object} rawParams
   * @returns {Promise<{width:number, height:number, backend:string, elapsedMs:number}|null>}
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
        result = await this._executePipeline(currentParams);
        currentParams = this._pendingParams;
      }
      return result;
    } finally {
      this._isProcessing = false;
    }
  }

  _normalizeParams(rawParams) {
    return {
      scale: Math.min(3.0, Math.max(0.1, Math.round((Number(rawParams.scale) || 1.0) * 10) / 10)),
      superResolution: Math.min(100, Math.max(0, Math.round(Number(rawParams.superResolution) || 0))),
      detail: Math.min(100, Math.max(0, Math.round(Number(rawParams.detail) || 0))),
      sharpness: Math.min(10, Math.max(0, Math.round(Number(rawParams.sharpness) || 0))),
      brightness: Math.min(100, Math.max(-100, Math.round(Number(rawParams.brightness) || 0))),
      contrast: Math.min(100, Math.max(-100, Math.round(Number(rawParams.contrast) || 0))),
      mosaicSize: Math.min(80, Math.max(2, Math.round(Number(rawParams.mosaicSize) || 15)))
    };
  }

  async _executePipeline(rawParams) {
    const startTime = performance.now();
    const genAtStart = this._imageGeneration;
    const params = this._normalizeParams(rawParams);

    const baseCanvas = this.getActiveBaseCanvas();
    const baseW = baseCanvas.width;
    const baseH = baseCanvas.height;

    let targetW = Math.max(1, Math.round(baseW * params.scale));
    let targetH = Math.max(1, Math.round(baseH * params.scale));

    if (targetW > MAX_OUTPUT_DIMENSION || targetH > MAX_OUTPUT_DIMENSION) {
      const limitRatio = Math.min(MAX_OUTPUT_DIMENSION / targetW, MAX_OUTPUT_DIMENSION / targetH);
      targetW = Math.max(1, Math.round(targetW * limitRatio));
      targetH = Math.max(1, Math.round(targetH * limitRatio));
    }

    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }

    // 1. GPU (WebGPU → WebGL2) での描画を試行
    const usedGPU = this._gpuRenderer.render(
      baseCanvas,
      this.sourceVersion,
      targetW,
      targetH,
      params,
      this.mosaicRegions,
      this.ctx
    );

    if (usedGPU) {
      this.runtimeInfo.renderBackend = this._gpuRenderer.backend;
      this._updateRuntimeLabel();
      return {
        width: targetW,
        height: targetH,
        backend: this._gpuRenderer.backend,
        elapsedMs: Math.round(performance.now() - startTime)
      };
    }

    // 2. GPU非対応・テクスチャ上限超過時は Web Worker (CPU) へフォールバック
    this.runtimeInfo.renderBackend = 'cpu-worker';
    this._updateRuntimeLabel();

    const baseCtx = baseCanvas === this.aiBaseCanvas
      ? this.aiBaseCtx
      : (baseCanvas === this.croppedCanvas ? this.croppedCtx : this.rawCtx);
    const srcImageData = baseCtx.getImageData(0, 0, baseW, baseH);
    const copyBuffer = srcImageData.data.buffer.slice(0);

    const workerResult = await this._runWorkerPipeline({
      sourceBuffer: copyBuffer,
      srcW: baseW,
      srcH: baseH,
      targetW,
      targetH,
      params,
      mosaicRegions: this.mosaicRegions.map(r => ({ ...r }))
    });

    // 非同期完了時に画像が切り替わっていた場合は描画を破棄
    if (genAtStart !== this._imageGeneration) {
      return null;
    }

    const outData = new ImageData(new Uint8ClampedArray(workerResult.buffer), targetW, targetH);
    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }
    this.ctx.putImageData(outData, 0, 0);

    return {
      width: targetW,
      height: targetH,
      backend: 'cpu-worker',
      elapsedMs: Math.round(performance.now() - startTime)
    };
  }

  _runWorkerPipeline(payload) {
    this._initWorker();
    if (!this._worker) {
      return Promise.reject(new Error('Web Workerを初期化できませんでした'));
    }

    const jobId = ++this._workerJobSeq;
    return new Promise((resolve, reject) => {
      this._workerCallbacks.set(jobId, { resolve, reject });
      this._worker.postMessage(
        { jobId, ...payload },
        [payload.sourceBuffer]
      );
    });
  }

  /**
   * 現在のメインキャンバスを指定フォーマット・品質で Blob URL に書き出す。
   * @param {'image/png'|'image/jpeg'|'image/webp'} [mimeType='image/png']
   * @param {number} [quality=0.92] 0.1..1.0 (JPEG/WebP時)
   * @returns {Promise<{url:string, size:number}>}
   */
  exportImageBlobUrl(mimeType = 'image/png', quality = 0.92) {
    return new Promise((resolve, reject) => {
      if (!this.imageLoaded) {
        reject(new Error('画像がありません'));
        return;
      }

      // JPEGは透過を持てないため、白背景に合成してから書き出す（透明部分が黒くなるのを防止）
      let exportCanvas = this.canvas;
      let tempCanvas = null;
      if (mimeType === 'image/jpeg') {
        tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        const tCtx = tempCanvas.getContext('2d');
        tCtx.fillStyle = '#ffffff';
        tCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        tCtx.drawImage(this.canvas, 0, 0);
        exportCanvas = tempCanvas;
      }

      exportCanvas.toBlob(blob => {
        if (tempCanvas) {
          tempCanvas.width = 0;
          tempCanvas.height = 0;
        }
        if (!blob) {
          reject(new Error('画像の書き出しに失敗しました'));
          return;
        }
        resolve({ url: URL.createObjectURL(blob), size: blob.size });
      }, mimeType, quality);
    });
  }

  dispose() {
    this._gpuRenderer.dispose();
    if (this._worker) {
      this._worker.terminate();
      this._worker = null;
    }
  }
}
