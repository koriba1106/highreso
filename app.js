/**
 * app.js
 * アプリケーションUIコントローラ。
 * 単一状態同期・非破壊プレビューズーム（Fit / 100%）・ハイブリッド元画像比較・
 * Swin2SR 同意/高解像度縮小確認モーダル・保存形式設定モーダルを制御する。
 */

import { ImageCoreEngine, PRESETS } from './engine.js';
import { OverlayController } from './overlay-crop.js';

const els = {
  // ヘッダー・ファイル操作
  btnOpenFile: document.getElementById('btn-open-file'),
  btnEmptyOpen: document.getElementById('btn-empty-open'),
  fileInput: document.getElementById('file-input'),
  dropZone: document.getElementById('drop-zone'),
  emptyState: document.getElementById('empty-state'),
  scrollArea: document.getElementById('canvas-scroll-area'),
  canvasWrapper: document.getElementById('canvas-wrapper'),
  mainCanvas: document.getElementById('main-canvas'),
  overlayCanvas: document.getElementById('overlay-canvas'),
  processingOverlay: document.getElementById('processing-overlay'),
  processingText: document.getElementById('processing-text'),
  engineBadge: document.getElementById('engine-badge'),
  imageInfo: document.getElementById('image-info'),
  btnCompare: document.getElementById('btn-compare'),
  btnResetAll: document.getElementById('btn-reset-all'),
  btnDownload: document.getElementById('btn-download'),

  // Swin2SR AI UI
  btnSwin2sr: document.getElementById('btn-swin2sr'),
  swin2srBadge: document.getElementById('swin2sr-status-badge'),
  btnAiPrefReset: document.getElementById('btn-ai-pref-reset'),
  aiToggleRow: document.getElementById('ai-toggle-row'),
  toggleAiActive: document.getElementById('toggle-ai-active'),
  aiProgressWrap: document.getElementById('ai-progress-wrap'),
  aiProgressFill: document.getElementById('ai-progress-fill'),
  aiProgressText: document.getElementById('ai-progress-text'),

  // 同意モーダル
  aiConsentModal: document.getElementById('ai-consent-modal'),
  chkRememberConsent: document.getElementById('chk-remember-consent'),
  btnConsentYes: document.getElementById('btn-consent-yes'),
  btnConsentNo: document.getElementById('btn-consent-no'),

  // 高解像度縮小確認モーダル
  aiShrinkModal: document.getElementById('ai-shrink-modal'),
  shrinkModalDesc: document.getElementById('shrink-modal-desc'),
  shrinkModalWarning: document.getElementById('shrink-modal-warning'),
  btnShrinkCancel: document.getElementById('btn-shrink-cancel'),
  btnShrinkConfirm: document.getElementById('btn-shrink-confirm'),

  // 保存オプションモーダル
  exportModal: document.getElementById('export-modal'),
  exportFilename: document.getElementById('export-filename'),
  exportFormat: document.getElementById('export-format'),
  exportQualityRow: document.getElementById('export-quality-row'),
  exportQuality: document.getElementById('export-quality'),
  valExportQuality: document.getElementById('val-export-quality'),
  exportMetaInfo: document.getElementById('export-meta-info'),
  btnExportCancel: document.getElementById('btn-export-cancel'),
  btnExportConfirm: document.getElementById('btn-export-confirm'),

  // ズームバー
  viewZoomSlider: document.getElementById('view-zoom-slider'),
  viewZoomLabel: document.getElementById('view-zoom-label'),
  btnZoomIn: document.getElementById('btn-zoom-in'),
  btnZoomOut: document.getElementById('btn-zoom-out'),
  btnZoomFit: document.getElementById('btn-zoom-fit'),
  btnZoom100: document.getElementById('btn-zoom-100'),

  // パラメータスライダー
  paramSuperres: document.getElementById('param-superres'),
  paramDetail: document.getElementById('param-detail'),
  paramSharpness: document.getElementById('param-sharpness'),
  paramScale: document.getElementById('param-scale'),
  paramBrightness: document.getElementById('param-brightness'),
  paramContrast: document.getElementById('param-contrast'),
  paramMosaic: document.getElementById('param-mosaic'),

  // パラメータ値ラベル
  valSuperres: document.getElementById('val-superres'),
  valDetail: document.getElementById('val-detail'),
  valSharpness: document.getElementById('val-sharpness'),
  valScale: document.getElementById('val-scale'),
  valBrightness: document.getElementById('val-brightness'),
  valContrast: document.getElementById('val-contrast'),
  valMosaic: document.getElementById('val-mosaic'),

  // トリミングUI
  toggleCropMode: document.getElementById('toggle-crop-mode'),
  ratioBtns: document.querySelectorAll('.ratio-btn'),
  cropX: document.getElementById('crop-x'),
  cropY: document.getElementById('crop-y'),
  cropW: document.getElementById('crop-w'),
  cropH: document.getElementById('crop-h'),
  btnCropApply: document.getElementById('btn-crop-apply'),
  btnCropReset: document.getElementById('btn-crop-reset'),
  cropStatusLabel: document.getElementById('crop-status-label'),

  // モザイクUI
  toggleMosaicMode: document.getElementById('toggle-mosaic-mode'),
  btnMosaicRemoveSelected: document.getElementById('btn-mosaic-remove-selected'),
  btnMosaicUndo: document.getElementById('btn-mosaic-undo'),
  btnMosaicClear: document.getElementById('btn-mosaic-clear'),
  mosaicCount: document.getElementById('mosaic-count'),

  presetBtns: document.querySelectorAll('.preset-btn'),
  resetHeaders: document.querySelectorAll('.control-header[data-reset-target]'),
  toast: document.getElementById('toast')
};

const engine = new ImageCoreEngine(els.mainCanvas);

const appState = {
  sourceFileName: 'image',
  viewZoom: 100,
  fitMode: true,
  selectedAspectRatio: null,
  loadingCount: 0,
  compareLocked: false,
  comparePointerDownAt: 0,
  compareHolding: false,
  lastBackend: 'cpu-worker',
  lastElapsedMs: 0,
  renderDebounceTimer: null,
  dragCounter: 0,
  pendingConsentResolver: null,
  pendingShrinkResolver: null
};

const overlay = new OverlayController({
  overlayCanvas: els.overlayCanvas,
  engine,
  onDraftCropChange: (normRect) => syncCropInputsFromNorm(normRect),
  onMosaicAdd: (normRect) => {
    const addedIdx = engine.addMosaicRegion(normRect);
    overlay.setSelectedMosaicIndex(addedIdx);
    syncMosaicUI();
    scheduleProcess(true);
  },
  onMosaicSelect: () => {
    syncMosaicUI();
  }
});

// ====== ユーティリティ ＆ ローディング制御 ======

function showToast(msg, duration = 3200) {
  els.toast.textContent = msg;
  els.toast.classList.remove('hidden');
  clearTimeout(els.toast._timer);
  els.toast._timer = setTimeout(() => els.toast.classList.add('hidden'), duration);
}

/**
 * 参照カウンタ付きで処理中オーバーレイを表示する（非同期競合による早期消滅を防止）。
 * @param {string} text
 */
function beginLoading(text = '処理中...') {
  appState.loadingCount++;
  els.processingText.textContent = text;
  els.processingOverlay.classList.remove('hidden');
}

function endLoading() {
  appState.loadingCount = Math.max(0, appState.loadingCount - 1);
  if (appState.loadingCount === 0) {
    els.processingOverlay.classList.add('hidden');
  }
}

function updateEngineBadge() {
  const info = engine.runtimeInfo;
  els.engineBadge.textContent = info.label;
  const isGpu = info.renderBackend === 'webgpu' || info.renderBackend === 'webgl2';
  els.engineBadge.className = `badge ${isGpu ? 'badge-ready' : 'badge-wasm'}`;
}

// ====== UI状態一元同期 ======

function syncParamLabels() {
  const sr = Number(els.paramSuperres.value);
  els.valSuperres.textContent = sr === 0 ? 'OFF (0)' : `Lv.${sr}`;
  els.valDetail.textContent = els.paramDetail.value;
  const sh = Number(els.paramSharpness.value);
  els.valSharpness.textContent = sh === 0 ? 'OFF' : `Lv.${sh} / 10`;
  els.valScale.textContent = `${Number(els.paramScale.value).toFixed(1)}x`;
  const br = Number(els.paramBrightness.value);
  els.valBrightness.textContent = br > 0 ? `+${br}` : `${br}`;
  const ct = Number(els.paramContrast.value);
  els.valContrast.textContent = ct > 0 ? `+${ct}` : `${ct}`;
  els.valMosaic.textContent = `${els.paramMosaic.value}px`;
}

function syncSwin2SRUI() {
  const state = engine.getSwin2SRState();
  const hasImg = engine.hasImage();
  const hasCache = engine.hasAIUpscaleCache();
  const isActive = engine.isAIUpscaleActive();

  els.swin2srBadge.className = 'ai-status-pill';

  if (state === 'ready') {
    const devText = engine.runtimeInfo.aiBackend === 'webgpu' ? 'WebGPU' : 'WASM';
    els.swin2srBadge.textContent = `準備完了 (${devText})`;
    els.swin2srBadge.classList.add('status-ready');
  } else if (state === 'loading') {
    els.swin2srBadge.textContent = 'モデル取得・準備中...';
    els.swin2srBadge.classList.add('status-loading');
  } else if (state === 'declined') {
    els.swin2srBadge.textContent = '利用しない設定 (DLスキップ)';
    els.swin2srBadge.classList.add('status-declined');
  } else if (state === 'unavailable') {
    els.swin2srBadge.textContent = '利用不可 (数式エンジンのみ)';
    els.swin2srBadge.classList.add('status-declined');
  } else {
    els.swin2srBadge.textContent = '未取得 (初回DL約21〜54MB)';
    els.swin2srBadge.classList.add('status-unloaded');
  }

  if (state === 'loading') {
    els.btnSwin2sr.textContent = '⏳ Swin2SR モデルを準備中...';
  } else if (!hasImg) {
    els.btnSwin2sr.textContent = '🧠 AI超解像 (Swin2SR ×2) を実行';
  } else if (hasCache) {
    els.btnSwin2sr.textContent = '🔄 AI超解像 (Swin2SR ×2) を再実行';
  } else {
    els.btnSwin2sr.textContent = '🧠 AI超解像 (Swin2SR ×2) を実行';
  }

  els.btnSwin2sr.classList.toggle('active', isActive);
  els.aiToggleRow.classList.toggle('hidden', !hasCache);
  els.toggleAiActive.checked = isActive;
  els.toggleAiActive.disabled = !hasCache;
}

function syncCropButtonsUI() {
  const cropped = engine.isCropped();
  els.btnCropReset.disabled = !cropped;
  if (cropped) {
    const dims = engine.getPreAIBaseDimensions();
    els.cropStatusLabel.textContent = `適用中 (${dims.width}×${dims.height})`;
  } else {
    els.cropStatusLabel.textContent = '未適用 (全体)';
  }
}

function syncMosaicUI() {
  const regions = engine.getMosaicRegions();
  const count = regions.length;
  els.mosaicCount.textContent = `${count}箇所`;
  els.btnMosaicUndo.disabled = count === 0;
  els.btnMosaicClear.disabled = count === 0;
  els.btnMosaicRemoveSelected.disabled =
    overlay.selectedMosaicIdx < 0 || overlay.selectedMosaicIdx >= count;
}

function syncCropInputsFromNorm(normRect) {
  if (!engine.hasImage()) return;
  const dims = engine.getActiveBaseDimensions();
  els.cropX.value = Math.round(normRect.x * dims.width);
  els.cropY.value = Math.round(normRect.y * dims.height);
  els.cropW.value = Math.max(4, Math.round(normRect.w * dims.width));
  els.cropH.value = Math.max(4, Math.round(normRect.h * dims.height));
}

function enableControls(enabled) {
  const inputs = [
    els.btnCompare, els.btnResetAll, els.btnDownload,
    els.paramSuperres, els.paramDetail, els.paramSharpness,
    els.paramScale, els.paramBrightness, els.paramContrast,
    els.paramMosaic, els.toggleMosaicMode,
    els.toggleCropMode, els.cropX, els.cropY, els.cropW, els.cropH,
    els.btnCropApply
  ];
  inputs.forEach(el => { el.disabled = !enabled; });
  els.presetBtns.forEach(btn => { btn.disabled = !enabled; });
  els.ratioBtns.forEach(btn => { btn.disabled = !enabled; });
  syncCropButtonsUI();
  syncMosaicUI();
}

function resetParametersUI() {
  els.paramSuperres.value = 0;
  els.paramDetail.value = 0;
  els.paramSharpness.value = 0;
  els.paramScale.value = 1.0;
  els.paramBrightness.value = 0;
  els.paramContrast.value = 0;
  els.paramMosaic.value = 15;
  els.presetBtns.forEach(b => b.classList.remove('active'));
  els.toggleCropMode.checked = false;
  els.toggleMosaicMode.checked = false;
  overlay.setMode('none');
  setCompareMode(false);
  syncParamLabels();
  syncSwin2SRUI();
  syncCropButtonsUI();
  syncMosaicUI();
}

// ====== 初期化（起動時強制モーダルは表示しない） ======

async function initApp() {
  try {
    await engine.initializeRuntime();
    updateEngineBadge();
  } catch (_) {
    updateEngineBadge();
  }

  const savedConsent = engine.getSavedConsentPreference();
  if (savedConsent === 'no') {
    engine.declineSwin2SRDownload(false);
  }
  syncSwin2SRUI();
}
initApp();

// ====== ファイル読み込み ======

async function handleFile(file) {
  if (!file) return;
  beginLoading('画像を検証・展開中...');
  try {
    const meta = await engine.loadImage(file);
    appState.sourceFileName = (file.name || 'image').replace(/\.[^.]+$/, '') || 'image';
    els.emptyState.classList.add('hidden');
    els.scrollArea.classList.remove('hidden');
    enableControls(true);
    resetParametersUI();

    await triggerProcess();
    applyFitZoom();
    overlay.resetDraftCropToFull();

    if (meta.downscaledFrom) {
      showToast(
        `最大サイズ制限(4096px)のため ${meta.downscaledFrom.width}×${meta.downscaledFrom.height} から ${meta.width}×${meta.height} px へ調整しました`,
        4500
      );
    } else {
      showToast(`画像を読み込みました (${meta.width} × ${meta.height} px)`);
    }
  } catch (err) {
    showToast(`エラー: ${err.message}`, 4500);
  } finally {
    endLoading();
    els.fileInput.value = '';
  }
}

// ====== パイプライン処理実行 ＆ ズーム ======

function getParamsFromUI() {
  return {
    superResolution: Number(els.paramSuperres.value),
    detail: Number(els.paramDetail.value),
    sharpness: Number(els.paramSharpness.value),
    scale: Number(els.paramScale.value),
    brightness: Number(els.paramBrightness.value),
    contrast: Number(els.paramContrast.value),
    mosaicSize: Number(els.paramMosaic.value)
  };
}

function scheduleProcess(immediate = false) {
  syncParamLabels();
  clearTimeout(appState.renderDebounceTimer);
  if (immediate) {
    triggerProcess();
  } else {
    const delay = engine.runtimeInfo.renderBackend === 'cpu-worker' ? 80 : 16;
    appState.renderDebounceTimer = setTimeout(() => triggerProcess(), delay);
  }
}

async function triggerProcess() {
  if (!engine.hasImage()) return;
  const params = getParamsFromUI();
  const isWorkerSlow = engine.runtimeInfo.renderBackend === 'cpu-worker' && params.scale >= 1.8;

  if (isWorkerSlow) {
    beginLoading('高精細レンダリング中...');
  }
  try {
    const resultInfo = await engine.process(params);
    if (resultInfo) {
      appState.lastBackend = resultInfo.backend;
      appState.lastElapsedMs = resultInfo.elapsedMs;
      updateEngineBadge();
      if (appState.fitMode) {
        applyFitZoom();
      } else {
        applyViewZoom(appState.viewZoom, false);
      }

      const aiTag = engine.isAIUpscaleActive() ? '[AI×2]' : '[通常]';
      const cropTag = engine.isCropped() ? '[切抜]' : '';
      const backendTag = resultInfo.backend === 'webgpu'
        ? 'WebGPU'
        : (resultInfo.backend === 'webgl2' ? 'WebGL2' : 'CPU');
      els.imageInfo.textContent = `${aiTag}${cropTag} ${resultInfo.width}×${resultInfo.height} px (${params.scale.toFixed(1)}x | ${backendTag} ${resultInfo.elapsedMs}ms)`;
    }
  } catch (err) {
    showToast(`処理エラー: ${err.message}`);
  } finally {
    if (isWorkerSlow) endLoading();
  }
}

/**
 * 現在のプレビュー領域に画像全体が収まるズーム倍率を計算して適用する。
 */
function applyFitZoom() {
  appState.fitMode = true;
  if (!engine.hasImage()) return;

  const w = els.mainCanvas.width;
  const h = els.mainCanvas.height;
  if (w <= 0 || h <= 0) return;

  const availW = Math.max(120, els.scrollArea.clientWidth - 44);
  const availH = Math.max(120, els.scrollArea.clientHeight - 88);
  const ratio = Math.min(availW / w, availH / h, 1.0); // 100%以下で画面フィット（小さい画像は等倍表示）
  const fitPct = Math.max(10, Math.min(300, Math.floor((ratio * 100) / 5) * 5));
  applyViewZoom(fitPct, true);
}

/**
 * 表示ズーム倍率を適用し、オーバーレイ解像度を同期する。
 * @param {number} [newZoom]
 * @param {boolean} [keepFitMode=false]
 */
function applyViewZoom(newZoom, keepFitMode = false) {
  if (!keepFitMode) {
    appState.fitMode = false;
  }
  if (typeof newZoom === 'number') {
    appState.viewZoom = Math.min(300, Math.max(10, Math.round(newZoom / 5) * 5));
    els.viewZoomSlider.value = appState.viewZoom;
  } else {
    appState.viewZoom = Number(els.viewZoomSlider.value);
  }
  els.viewZoomLabel.textContent = `${appState.viewZoom}%`;

  if (!engine.hasImage()) return;
  const w = els.mainCanvas.width;
  const h = els.mainCanvas.height;
  const displayW = Math.max(1, Math.round(w * (appState.viewZoom / 100)));
  const displayH = Math.max(1, Math.round(h * (appState.viewZoom / 100)));

  els.canvasWrapper.style.width = `${displayW}px`;
  els.canvasWrapper.style.height = `${displayH}px`;
  overlay.resizeToDisplay(displayW, displayH);
}

// ====== トリミング操作 (数値確定・比率・適用/解除) ======

function commitCropNumericInput(changedField) {
  if (!engine.hasImage()) return;
  const dims = engine.getActiveBaseDimensions();
  const cw = dims.width;
  const ch = dims.height;
  const minEdge = Math.min(4, cw, ch);

  let x = Math.max(0, Math.min(cw - minEdge, Math.round(Number(els.cropX.value) || 0)));
  let y = Math.max(0, Math.min(ch - minEdge, Math.round(Number(els.cropY.value) || 0)));
  let w = Math.max(minEdge, Math.min(cw - x, Math.round(Number(els.cropW.value) || minEdge)));
  let h = Math.max(minEdge, Math.min(ch - y, Math.round(Number(els.cropH.value) || minEdge)));

  if (appState.selectedAspectRatio) {
    if (changedField === 'h') {
      w = Math.round(h * appState.selectedAspectRatio);
      if (x + w > cw) {
        w = cw - x;
        h = Math.max(minEdge, Math.round(w / appState.selectedAspectRatio));
      }
    } else if (changedField === 'w') {
      h = Math.round(w / appState.selectedAspectRatio);
      if (y + h > ch) {
        h = ch - y;
        w = Math.max(minEdge, Math.round(h * appState.selectedAspectRatio));
      }
    }
  }

  const norm = { x: x / cw, y: y / ch, w: w / cw, h: h / ch };
  overlay.setDraftCropNorm(norm);
  syncCropInputsFromNorm(overlay.getDraftCropNorm());

  if (!els.toggleCropMode.checked) {
    els.toggleCropMode.checked = true;
    els.toggleMosaicMode.checked = false;
    overlay.setMode('crop');
  }
}

[
  [els.cropX, 'x'],
  [els.cropY, 'y'],
  [els.cropW, 'w'],
  [els.cropH, 'h']
].forEach(([inputEl, field]) => {
  inputEl.addEventListener('change', () => commitCropNumericInput(field));
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commitCropNumericInput(field);
      inputEl.blur();
    }
  });
});

els.ratioBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    if (!engine.hasImage()) return;
    els.ratioBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const ratioStr = btn.dataset.ratio;
    if (ratioStr === 'free') {
      appState.selectedAspectRatio = null;
    } else {
      const [rw, rh] = ratioStr.split(':').map(Number);
      appState.selectedAspectRatio = rw / rh;
    }

    els.toggleCropMode.checked = true;
    els.toggleMosaicMode.checked = false;
    overlay.setMode('crop');
    overlay.setAspectRatio(appState.selectedAspectRatio);
  });
});

els.btnCropApply.addEventListener('click', async () => {
  if (!engine.hasImage()) return;
  const draftNorm = overlay.getDraftCropNorm();
  const dims = engine.getActiveBaseDimensions();
  if (draftNorm.w * dims.width < 4 || draftNorm.h * dims.height < 4) {
    showToast('トリミング範囲が小さすぎます');
    return;
  }

  const { aiReset, mosaicCleared } = engine.applyCropNormalized(draftNorm);
  els.toggleCropMode.checked = false;
  overlay.setMode('none');
  syncCropButtonsUI();
  syncSwin2SRUI();
  syncMosaicUI();
  await triggerProcess();
  overlay.resetDraftCropToFull();

  if (aiReset) {
    showToast('トリミングを適用しました（範囲変更のためAIベースをリセットしました）', 4000);
  } else if (mosaicCleared) {
    showToast('トリミングを適用しました（座標変更のためモザイクをクリアしました）', 3500);
  } else {
    showToast('トリミングを適用しました');
  }
});

els.btnCropReset.addEventListener('click', async () => {
  if (!engine.hasImage() || !engine.isCropped()) return;
  const { aiReset } = engine.resetCrop();
  syncCropButtonsUI();
  syncSwin2SRUI();
  syncMosaicUI();
  await triggerProcess();
  overlay.resetDraftCropToFull();
  showToast(
    aiReset
      ? 'トリミングを解除し全体へ戻しました（AIベースをリセットしました）'
      : 'トリミングを解除し、元の画像全体に戻しました'
  );
});

// ====== Swin2SR 同意・縮小確認・推論実行フロー ======

function askConsentModal() {
  return new Promise(resolve => {
    appState.pendingConsentResolver = resolve;
    els.chkRememberConsent.checked = true;
    els.aiConsentModal.classList.remove('hidden');
  });
}

function resolveConsentModal(accepted) {
  els.aiConsentModal.classList.add('hidden');
  const resolver = appState.pendingConsentResolver;
  appState.pendingConsentResolver = null;
  if (resolver) resolver(accepted);
}

/**
 * 長辺640px超過時に警告モーダルを表示してユーザー確認を取る。
 * @param {ReturnType<ImageCoreEngine['getSwin2SRInputPlan']>} plan
 * @returns {Promise<boolean>}
 */
function askShrinkConfirmModal(plan) {
  return new Promise(resolve => {
    appState.pendingShrinkResolver = resolve;

    els.shrinkModalDesc.innerHTML =
      `対象画像（<strong>${plan.baseW} × ${plan.baseH} px</strong>）は AI推論の上限（長辺 640 px）を超えています。<br>` +
      `実行すると長辺を <strong>${plan.inputW} × ${plan.inputH} px</strong> に縮小してから AI 2倍超解像（出力予定: <strong>${plan.estimatedOutW} × ${plan.estimatedOutH} px</strong>）を行います。`;

    if (plan.willReduceResolution) {
      els.shrinkModalWarning.textContent =
        `⚠️ 注意: AI超解像後の出力サイズ (${plan.estimatedOutW}×${plan.estimatedOutH}) は、元画像 (${plan.baseW}×${plan.baseH}) よりも画素数が少なくなります。`;
      els.shrinkModalWarning.classList.remove('hidden');
    } else {
      els.shrinkModalWarning.classList.add('hidden');
    }

    els.aiShrinkModal.classList.remove('hidden');
  });
}

function resolveShrinkModal(confirmed) {
  els.aiShrinkModal.classList.add('hidden');
  const resolver = appState.pendingShrinkResolver;
  appState.pendingShrinkResolver = null;
  if (resolver) resolver(confirmed);
}

els.btnConsentYes.addEventListener('click', () => {
  if (els.chkRememberConsent.checked) {
    engine.saveConsentPreference('yes');
  } else {
    engine.clearConsentPreference();
  }
  resolveConsentModal(true);
});

els.btnConsentNo.addEventListener('click', () => {
  engine.declineSwin2SRDownload(els.chkRememberConsent.checked);
  syncSwin2SRUI();
  resolveConsentModal(false);
});

els.btnShrinkConfirm.addEventListener('click', () => resolveShrinkModal(true));
els.btnShrinkCancel.addEventListener('click', () => resolveShrinkModal(false));

els.btnAiPrefReset.addEventListener('click', async () => {
  const accepted = await askConsentModal();
  if (accepted) {
    await executeSwin2SRDownloadAndRun(false);
  } else {
    showToast('AIモデルを利用しない設定にしました（数式アルゴリズムのみ使用）');
  }
});

async function executeSwin2SRDownloadAndRun(askShrink = true) {
  if (engine.getSwin2SRState() === 'loading') return;

  if (askShrink && engine.hasImage()) {
    const plan = engine.getSwin2SRInputPlan();
    if (plan.needsShrinkConfirm) {
      const ok = await askShrinkConfirmModal(plan);
      if (!ok) {
        showToast('AI超解像をキャンセルしました（トリミング後に再実行できます）');
        return;
      }
    }
  }

  els.aiProgressWrap.classList.remove('hidden');
  els.aiProgressFill.style.width = '0%';
  els.aiProgressText.textContent = 'Swin2SRモデルを接続・準備中...';
  els.btnSwin2sr.disabled = true;
  syncSwin2SRUI();

  const progressCallback = (prog) => {
    if (prog.status === 'progress' && typeof prog.progress === 'number') {
      const pct = Math.min(100, Math.max(0, Math.round(prog.progress)));
      els.aiProgressFill.style.width = `${pct}%`;
      els.aiProgressText.textContent = `モデル準備中... ${pct}%`;
    } else if (prog.status === 'inference') {
      els.aiProgressFill.style.width = '100%';
      els.aiProgressText.textContent = 'Swin2SR AI超解像推論を実行中...';
      els.processingText.textContent = `Swin2SR AI超解像推論中 (${engine.runtimeInfo.aiBackend === 'webgpu' ? 'WebGPU' : 'WASM'})...`;
    }
  };

  let loadingActive = false;
  try {
    if (engine.getSwin2SRState() !== 'ready') {
      await engine.downloadSwin2SRModel(progressCallback);
      updateEngineBadge();
    }

    if (engine.hasImage()) {
      beginLoading('Swin2SR AI超解像を実行中...');
      loadingActive = true;
      const res = await engine.runSwin2SRSuperResolution(progressCallback);
      if (!res.cancelled) {
        updateEngineBadge();
        syncSwin2SRUI();
        await triggerProcess();
        overlay.resetDraftCropToFull();
        const devLabel = engine.runtimeInfo.aiBackend === 'webgpu' ? 'WebGPU' : 'WASM';
        showToast(`Swin2SR AI超解像 (${devLabel}) が完了しました (${res.width} × ${res.height} px)`, 4000);
      }
    } else {
      showToast('Swin2SR AIモデルの準備が完了しました！画像を開いて実行できます');
    }
  } catch (err) {
    showToast(`Swin2SRを実行できないため数式処理のみ利用します (${err.message})`, 4500);
  } finally {
    if (loadingActive) endLoading();
    els.aiProgressWrap.classList.add('hidden');
    els.btnSwin2sr.disabled = false;
    syncSwin2SRUI();
    updateEngineBadge();
  }
}

els.btnSwin2sr.addEventListener('click', async () => {
  if (!engine.hasImage()) {
    showToast('先に「📂 画像を開く」から画像を選択してください');
    return;
  }

  const state = engine.getSwin2SRState();
  if (state === 'unavailable') {
    showToast('この環境ではAIランタイムを利用できません。下のパラメータ調整をご利用ください', 3800);
    return;
  }

  if (state !== 'ready') {
    const saved = engine.getSavedConsentPreference();
    if (saved !== 'yes') {
      const accepted = await askConsentModal();
      if (!accepted) {
        showToast('AIモデルのダウンロードをスキップしました。下の数式スライダーで高精細化できます');
        return;
      }
    }
  }

  await executeSwin2SRDownloadAndRun(true);
});

els.toggleAiActive.addEventListener('change', async () => {
  if (!engine.hasImage() || !engine.hasAIUpscaleCache()) return;
  const active = engine.setAIUpscaleEnabled(els.toggleAiActive.checked);
  syncSwin2SRUI();
  await triggerProcess();
  showToast(active ? 'Swin2SR AIベースを有効にしました' : '通常の元ベースに戻しました');
});

// ====== 元画像と比較（長押し＋ワンクリック切替ハイブリッド） ======

function setCompareMode(active) {
  appState.compareHolding = active;
  if (!active) {
    appState.compareLocked = false;
  }
  els.btnCompare.classList.toggle('active', active || appState.compareLocked);
  overlay.setComparing(active || appState.compareLocked);
}

els.btnCompare.addEventListener('pointerdown', (e) => {
  if (!engine.hasImage()) return;
  e.preventDefault();
  appState.comparePointerDownAt = performance.now();
  overlay.setComparing(true);
  els.btnCompare.classList.add('active');
});

const handleCompareRelease = (e, isLeave) => {
  if (!engine.hasImage()) return;
  if (isLeave && appState.compareLocked) return;

  const elapsed = performance.now() - appState.comparePointerDownAt;
  if (!isLeave && elapsed < 300) {
    // 短いクリックは比較ロックトグルとして動作
    appState.compareLocked = !appState.compareLocked;
    overlay.setComparing(appState.compareLocked);
    els.btnCompare.classList.toggle('active', appState.compareLocked);
  } else {
    // 長押しリリースは即座に処理後表示へ復帰
    appState.compareLocked = false;
    overlay.setComparing(false);
    els.btnCompare.classList.remove('active');
  }
};

els.btnCompare.addEventListener('pointerup', (e) => handleCompareRelease(e, false));
els.btnCompare.addEventListener('pointercancel', (e) => handleCompareRelease(e, true));
els.btnCompare.addEventListener('pointerleave', (e) => {
  if (e.buttons > 0) handleCompareRelease(e, true);
});

// ====== 一般イベントリスナー ======

els.btnOpenFile.addEventListener('click', () => els.fileInput.click());
els.btnEmptyOpen.addEventListener('click', () => els.fileInput.click());
els.fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

els.dropZone.addEventListener('dragenter', (e) => {
  e.preventDefault();
  appState.dragCounter++;
  els.dropZone.classList.add('drag-over');
});
els.dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
});
els.dropZone.addEventListener('dragleave', () => {
  appState.dragCounter = Math.max(0, appState.dragCounter - 1);
  if (appState.dragCounter === 0) {
    els.dropZone.classList.remove('drag-over');
  }
});
els.dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  appState.dragCounter = 0;
  els.dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

[
  els.paramSuperres, els.paramDetail, els.paramSharpness,
  els.paramScale, els.paramBrightness, els.paramContrast, els.paramMosaic
].forEach(slider => {
  slider.addEventListener('input', () => {
    els.presetBtns.forEach(b => b.classList.remove('active'));
    scheduleProcess(false);
  });
});

els.resetHeaders.forEach(header => {
  header.addEventListener('dblclick', () => {
    const targetId = header.dataset.resetTarget;
    const defaultVal = header.dataset.default;
    const inputEl = document.getElementById(targetId);
    if (inputEl && !inputEl.disabled) {
      inputEl.value = defaultVal;
      els.presetBtns.forEach(b => b.classList.remove('active'));
      scheduleProcess(true);
    }
  });
});

els.presetBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const key = btn.dataset.preset;
    const preset = PRESETS[key];
    if (!preset || !engine.hasImage()) return;

    els.presetBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    els.paramSuperres.value = preset.superResolution;
    els.paramDetail.value = preset.detail;
    els.paramSharpness.value = preset.sharpness;
    els.paramScale.value = preset.scale;
    els.paramBrightness.value = preset.brightness;
    els.paramContrast.value = preset.contrast;

    showToast(`プリセット「${preset.name}」を適用しました`);
    scheduleProcess(true);
  });
});

// ズーム操作
els.viewZoomSlider.addEventListener('input', () => applyViewZoom());
els.btnZoomIn.addEventListener('click', () => applyViewZoom(appState.viewZoom + 10));
els.btnZoomOut.addEventListener('click', () => applyViewZoom(appState.viewZoom - 10));
els.btnZoomFit.addEventListener('click', () => applyFitZoom());
els.btnZoom100.addEventListener('click', () => applyViewZoom(100));

window.addEventListener('resize', () => {
  if (appState.fitMode && engine.hasImage()) {
    applyFitZoom();
  }
});

// トリミング ＆ モザイク 排他トグル
els.toggleCropMode.addEventListener('change', () => {
  if (els.toggleCropMode.checked) {
    els.toggleMosaicMode.checked = false;
    setCompareMode(false);
    overlay.setMode('crop');
    syncCropInputsFromNorm(overlay.getDraftCropNorm());
    showToast('枠の移動・角のドラッグ、またはXYWH数値で範囲を指定してください');
  } else {
    overlay.setMode('none');
  }
});

els.toggleMosaicMode.addEventListener('change', () => {
  if (els.toggleMosaicMode.checked) {
    els.toggleCropMode.checked = false;
    setCompareMode(false);
    overlay.setMode('mosaic');
    showToast('画像上をドラッグしてモザイクを追加、枠クリックで個別削除できます');
  } else {
    overlay.setMode('none');
  }
  syncMosaicUI();
});

els.btnMosaicRemoveSelected.addEventListener('click', () => {
  if (overlay.selectedMosaicIdx >= 0 && engine.removeMosaicRegionAt(overlay.selectedMosaicIdx)) {
    overlay.setSelectedMosaicIndex(-1);
    syncMosaicUI();
    scheduleProcess(true);
  }
});

els.btnMosaicUndo.addEventListener('click', () => {
  engine.undoMosaicRegion();
  overlay.setSelectedMosaicIndex(-1);
  syncMosaicUI();
  scheduleProcess(true);
});

els.btnMosaicClear.addEventListener('click', () => {
  engine.clearMosaicRegions();
  overlay.setSelectedMosaicIndex(-1);
  syncMosaicUI();
  scheduleProcess(true);
});

els.btnResetAll.addEventListener('click', async () => {
  if (!engine.hasImage()) return;
  engine.resetToOriginalSource();
  resetParametersUI();
  await triggerProcess();
  applyFitZoom();
  overlay.resetDraftCropToFull();
  showToast('トリミング・モザイク・AI適用を含むすべての設定を初期状態に戻しました');
});

// ====== 画像保存モーダル ======

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function updateExportModalUI() {
  const fmt = els.exportFormat.value;
  const showQuality = fmt !== 'image/png';
  els.exportQualityRow.classList.toggle('hidden', !showQuality);
  els.valExportQuality.textContent = `${els.exportQuality.value}%`;
  els.exportMetaInfo.textContent = `出力解像度: ${els.mainCanvas.width} × ${els.mainCanvas.height} px`;
}

els.btnDownload.addEventListener('click', () => {
  if (!engine.hasImage()) return;
  els.exportFilename.value = `${appState.sourceFileName}_edited`;
  updateExportModalUI();
  els.exportModal.classList.remove('hidden');
});

els.exportFormat.addEventListener('change', updateExportModalUI);
els.exportQuality.addEventListener('input', updateExportModalUI);
els.btnExportCancel.addEventListener('click', () => els.exportModal.classList.add('hidden'));

els.btnExportConfirm.addEventListener('click', async () => {
  if (!engine.hasImage()) return;
  const mime = els.exportFormat.value;
  const extMap = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp' };
  const ext = extMap[mime] || 'png';
  const rawName = (els.exportFilename.value || 'edited_image').trim().replace(/[\\/:*?"<>|]/g, '_');
  const fileName = rawName.toLowerCase().endsWith(`.${ext}`) ? rawName : `${rawName}.${ext}`;
  const quality = Math.min(1, Math.max(0.1, Number(els.exportQuality.value) / 100));

  els.exportModal.classList.add('hidden');
  beginLoading('画像を書き出し中...');
  try {
    const { url, size } = await engine.exportImageBlobUrl(mime, quality);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    showToast(`画像を保存しました (${fileName} / ${formatBytes(size)})`, 3800);
  } catch (err) {
    showToast(`保存に失敗しました: ${err.message}`, 4000);
  } finally {
    endLoading();
  }
});
