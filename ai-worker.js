/**
 * @fileoverview Swin2SR AI超解像専用の Web Worker
 * Transformers.js v3 + ONNX Runtime Web をメインスレッドから分離して実行し、
 * 重いモデル構築・推論中のUIフリーズを完全に防ぐとともに、
 * 推論バックエンド失敗時の Worker 再生成による確実なフォールバックを可能にする。
 */

const LOCAL_TRANSFORMERS_URL = './libs/transformers.min.js';
const CDN_TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';
const WASM_DIR_URL = new URL('./libs/', import.meta.url).href;

let transformersMod = null;
let runtimeInitPromise = null;
let activePipeline = null;

/**
 * Emscripten WASM が投げる数値ポインタや非Errorオブジェクトを含め、可読なエラー文字列に変換する。
 * @param {unknown} err
 * @returns {string}
 */
function formatWorkerError(err) {
  if (typeof err === 'number') {
    return `ONNX WebAssembly例外 (code: ${err})`;
  }
  if (err && typeof err === 'object') {
    if (typeof err.message === 'string' && err.message.trim()) {
      return err.message;
    }
    if (typeof err.error === 'string' && err.error.trim()) {
      return err.error;
    }
    try {
      const json = JSON.stringify(err);
      if (json && json !== '{}') return json;
    } catch (_) {}
  }
  if (typeof err === 'string' && err.trim()) {
    return err;
  }
  return '不明なAIランタイムエラー';
}

/**
 * ローカル libs/ を最優先し、失敗時は CDN から Transformers.js v3 を読み込んで初期化する。
 */
async function ensureTransformersRuntime() {
  if (transformersMod) return transformersMod;
  if (runtimeInitPromise) return runtimeInitPromise;

  runtimeInitPromise = (async () => {
    let mod = null;
    try {
      mod = await import(LOCAL_TRANSFORMERS_URL);
    } catch (_) {
      mod = await Promise.race([
        import(CDN_TRANSFORMERS_URL),
        new Promise((_, reject) => setTimeout(() => reject(new Error('CDN読み込みがタイムアウトしました')), 8000))
      ]);
    }

    if (!mod || !mod.env || typeof mod.pipeline !== 'function' || typeof mod.RawImage !== 'function') {
      throw new Error('Transformers.js v3 の初期化インターフェースが不正です');
    }

    mod.env.allowLocalModels = false;
    // HTTP(非localhost LANアクセス等)では Cache API が未定義のため安全に無効化する
    mod.env.useBrowserCache = typeof caches !== 'undefined';

    if (mod.env.backends?.onnx?.wasm) {
      mod.env.backends.onnx.wasm.wasmPaths = WASM_DIR_URL;
      const canMultiThread = typeof self !== 'undefined' && Boolean(self.crossOriginIsolated);
      mod.env.backends.onnx.wasm.numThreads = canMultiThread
        ? Math.min(4, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2)
        : 1;
    }

    transformersMod = mod;
    return mod;
  })();

  try {
    return await runtimeInitPromise;
  } catch (err) {
    runtimeInitPromise = null;
    throw err;
  }
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (!msg || typeof msg !== 'object') return;
  const { type, jobId } = msg;

  try {
    if (type === 'probe') {
      await ensureTransformersRuntime();
      self.postMessage({
        type: 'result',
        jobId,
        ok: true,
        cacheAvailable: typeof caches !== 'undefined'
      });
      return;
    }

    if (type === 'load') {
      const mod = await ensureTransformersRuntime();
      activePipeline = null;

      let downloadCompleted = false;
      const onnxTotals = new Map();
      const onnxLoaded = new Map();

      const progressCallback = (prog) => {
        if (!prog || typeof prog !== 'object') return;
        const fileName = typeof prog.file === 'string' ? prog.file : '';
        const isOnnx = fileName.endsWith('.onnx');

        if (prog.status === 'progress' && isOnnx && typeof prog.total === 'number' && prog.total > 0) {
          onnxTotals.set(fileName, prog.total);
          onnxLoaded.set(fileName, prog.loaded || 0);
          let sumTotal = 0;
          let sumLoaded = 0;
          for (const [f, t] of onnxTotals) {
            sumTotal += t;
            sumLoaded += onnxLoaded.get(f) || 0;
          }
          const combinedPct = sumTotal > 0 ? Math.min(99, Math.round((sumLoaded / sumTotal) * 100)) : Math.round(prog.progress || 0);
          self.postMessage({
            type: 'progress',
            jobId,
            progress: {
              status: 'progress',
              file: fileName,
              loaded: sumLoaded,
              total: sumTotal,
              progress: combinedPct
            }
          });
          return;
        }

        if (prog.status === 'done' && isOnnx && !downloadCompleted) {
          downloadCompleted = true;
          self.postMessage({
            type: 'progress',
            jobId,
            progress: { status: 'init_session', file: fileName }
          });
          return;
        }

        if (prog.status === 'initiate' || prog.status === 'download') {
          self.postMessage({
            type: 'progress',
            jobId,
            progress: { status: prog.status, file: fileName }
          });
        }
      };

      let stage = 'session_init';
      try {
        activePipeline = await mod.pipeline(
          'image-to-image',
          msg.modelId,
          {
            device: msg.device,
            dtype: msg.dtype,
            progress_callback: (p) => {
              if (!downloadCompleted && p && (p.status === 'download' || p.status === 'progress')) {
                stage = 'download';
              } else if (downloadCompleted) {
                stage = 'session_init';
              }
              progressCallback(p);
            }
          }
        );
      } catch (err) {
        const prefix = (!downloadCompleted && stage === 'download')
          ? 'モデルダウンロード失敗'
          : 'AIセッション初期化失敗';
        throw new Error(`${prefix}: ${formatWorkerError(err)}`);
      }

      self.postMessage({
        type: 'result',
        jobId,
        ok: true,
        device: msg.device,
        dtype: msg.dtype
      });
      return;
    }

    if (type === 'infer') {
      const mod = await ensureTransformersRuntime();
      if (!activePipeline) {
        throw new Error('Swin2SRモデルが初期化されていません');
      }

      const rgbBytes = new Uint8ClampedArray(msg.rgbBuffer);
      const rawInput = new mod.RawImage(rgbBytes, msg.width, msg.height, 3);
      const out = await activePipeline(rawInput);

      const src = out.data;
      const packed = (src instanceof Uint8ClampedArray && src.byteOffset === 0 && src.byteLength === src.buffer.byteLength)
        ? src
        : new Uint8ClampedArray(src);

      self.postMessage(
        {
          type: 'result',
          jobId,
          ok: true,
          width: out.width,
          height: out.height,
          channels: out.channels || Math.round(packed.length / (out.width * out.height)),
          buffer: packed.buffer
        },
        [packed.buffer]
      );
      return;
    }

    throw new Error(`未対応のAI Worker命令: ${type}`);
  } catch (err) {
    self.postMessage({
      type: 'result',
      jobId,
      ok: false,
      error: formatWorkerError(err)
    });
  }
};
