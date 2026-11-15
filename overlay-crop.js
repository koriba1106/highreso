/**
 * overlay-crop.js
 * プレビュー上のオーバーレイキャンバス（トリミング枠の移動・8方向リサイズ、
 * モザイク範囲の追加・選択、元画像の比較プレビュー）を管理するモジュール。
 *
 * メモリ最適化:
 *   overlayCanvas の解像度は元画像の実ピクセル数(最大4096px)ではなく、
 *   画面上の表示CSSサイズ × devicePixelRatio (上限2) に一致させることで
 *   Retina環境での鮮明な線描画と省メモリを両立する。
 */

const HANDLE_HIT_RADIUS_PX = 14;
const MIN_CROP_NORM_DEFAULT = 0.01;

export class OverlayController {
  /**
   * @param {object} options
   * @param {HTMLCanvasElement} options.overlayCanvas
   * @param {import('./engine.js').ImageCoreEngine} options.engine
   * @param {(normRect: {x:number,y:number,w:number,h:number}) => void} options.onDraftCropChange
   * @param {(normRect: {x:number,y:number,w:number,h:number}) => void} options.onMosaicAdd
   * @param {(selectedIdx: number) => void} options.onMosaicSelect
   */
  constructor({ overlayCanvas, engine, onDraftCropChange, onMosaicAdd, onMosaicSelect }) {
    this.canvas = overlayCanvas;
    this.ctx = overlayCanvas.getContext('2d');
    this.engine = engine;

    this.onDraftCropChange = onDraftCropChange;
    this.onMosaicAdd = onMosaicAdd;
    this.onMosaicSelect = onMosaicSelect;

    /** @type {'none' | 'crop' | 'mosaic'} */
    this.mode = 'none';
    this.isComparing = false;

    /** 正規化トリミングドラフト矩形 (0.0 .. 1.0) */
    this.draftCropNorm = { x: 0, y: 0, w: 1, h: 1 };
    /** アスペクト比固定値 (実ピクセル比 width / height)、自由時は null */
    this.aspectRatio = null;

    /** モザイク関連状態 */
    this.selectedMosaicIdx = -1;
    this._currentMosaicNorm = null;

    /** ポインタードラッグ内部状態 */
    this._dragAction = null; // 'new-crop' | 'move-crop' | 'nw'|'n'|'ne'|'e'|'se'|'s'|'sw'|'w' | 'new-mosaic'
    this._dragStartNorm = { x: 0, y: 0 };
    this._dragStartCss = { x: 0, y: 0 };
    this._dragStartCrop = null;
    this._displayW = 0;
    this._displayH = 0;
    this._dpr = 1;

    this._bindEvents();
  }

  /**
   * 表示CSSサイズに合わせてオーバーレイキャンバスの物理解像度を更新し再描画する。
   * @param {number} displayW CSS表示幅 (px)
   * @param {number} displayH CSS表示高さ (px)
   */
  resizeToDisplay(displayW, displayH) {
    this._displayW = Math.max(1, Math.round(displayW));
    this._displayH = Math.max(1, Math.round(displayH));
    this._dpr = Math.min(window.devicePixelRatio || 1, 2);

    const targetW = Math.max(1, Math.round(this._displayW * this._dpr));
    const targetH = Math.max(1, Math.round(this._displayH * this._dpr));
    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }
    this.render();
  }

  /**
   * 現在の操作モードを設定する。
   * @param {'none'|'crop'|'mosaic'} nextMode
   */
  setMode(nextMode) {
    this.mode = nextMode;
    if (nextMode !== 'mosaic') {
      this.selectedMosaicIdx = -1;
      this._currentMosaicNorm = null;
    }
    this._updateCursor({ x: -1, y: -1 });
    this.render();
  }

  setComparing(active) {
    this.isComparing = Boolean(active);
    this.render();
  }

  setSelectedMosaicIndex(idx) {
    this.selectedMosaicIdx = idx;
    this.render();
  }

  setAspectRatio(ratio) {
    this.aspectRatio = ratio;
    if (this.engine.hasImage()) {
      this.fitDraftCropToAspectRatio(ratio);
    }
  }

  resetDraftCropToFull() {
    this.draftCropNorm = { x: 0, y: 0, w: 1, h: 1 };
    if (this.aspectRatio && this.engine.hasImage()) {
      this.fitDraftCropToAspectRatio(this.aspectRatio);
    } else {
      this.onDraftCropChange({ ...this.draftCropNorm });
      this.render();
    }
  }

  /**
   * ソース画像ピクセル比率に基づき、指定アスペクト比で中央最大のトリミング矩形を設定する。
   * @param {number|null} ratio width / height
   */
  fitDraftCropToAspectRatio(ratio) {
    if (!this.engine.hasImage()) return;
    if (!ratio) {
      this.draftCropNorm = { x: 0, y: 0, w: 1, h: 1 };
    } else {
      const dims = this.engine.getActiveBaseDimensions();
      const cw = dims.width;
      const ch = dims.height;
      let w = cw;
      let h = Math.round(w / ratio);
      if (h > ch) {
        h = ch;
        w = Math.round(h * ratio);
      }
      const x = Math.round((cw - w) / 2);
      const y = Math.round((ch - h) / 2);
      this.draftCropNorm = {
        x: x / cw,
        y: y / ch,
        w: Math.max(4, w) / cw,
        h: Math.max(4, h) / ch
      };
    }
    this.onDraftCropChange({ ...this.draftCropNorm });
    this.render();
  }

  /**
   * 外部数値入力から正規化トリミング矩形を更新する。
   * @param {{x:number,y:number,w:number,h:number}} normRect
   */
  setDraftCropNorm(normRect) {
    const x = Math.max(0, Math.min(0.99, normRect.x));
    const y = Math.max(0, Math.min(0.99, normRect.y));
    const w = Math.max(MIN_CROP_NORM_DEFAULT, Math.min(1 - x, normRect.w));
    const h = Math.max(MIN_CROP_NORM_DEFAULT, Math.min(1 - y, normRect.h));
    this.draftCropNorm = { x, y, w, h };
    this.render();
  }

  getDraftCropNorm() {
    return { ...this.draftCropNorm };
  }

  // ====== 描画処理 ======

  render() {
    const ctx = this.ctx;
    const w = this._displayW;
    const h = this._displayH;
    if (w <= 0 || h <= 0) return;

    ctx.save();
    ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    if (!this.engine.hasImage()) {
      ctx.restore();
      return;
    }

    // 1. 「元画像と比較」表示中は AI未適用/フィルタ未適用ベースをオーバーレイ全面に描画
    if (this.isComparing) {
      const preBase = this.engine.getPreAIBaseCanvas();
      // 透過PNGの比較時に下の加工後画像が透けないようチェッカー地またはクリア色で覆ってから描く
      ctx.fillStyle = '#e5e7eb';
      ctx.fillRect(0, 0, w, h);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(preBase, 0, 0, w, h);

      // 比較中ステータスバッジを左上に表示
      const label = '👁️ 元画像（加工前）を表示中';
      ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      const padX = 10;
      const textW = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
      this._roundRect(ctx, 10, 10, textW + padX * 2, 26, 13);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, 10 + padX, 23);

      ctx.restore();
      return;
    }

    // 2. トリミング範囲指定モードの描画
    if (this.mode === 'crop') {
      this._drawCropOverlay(ctx, w, h);
    }

    // 3. モザイク選択モードの描画
    if (this.mode === 'mosaic' || this._currentMosaicNorm) {
      this._drawMosaicOverlay(ctx, w, h);
    }

    ctx.restore();
  }

  _drawCropOverlay(ctx, w, h) {
    const rx = this.draftCropNorm.x * w;
    const ry = this.draftCropNorm.y * h;
    const rw = this.draftCropNorm.w * w;
    const rh = this.draftCropNorm.h * h;

    // 周囲を暗転マスク
    ctx.fillStyle = 'rgba(15, 23, 42, 0.56)';
    ctx.fillRect(0, 0, w, ry);
    ctx.fillRect(0, ry + rh, w, Math.max(0, h - (ry + rh)));
    ctx.fillRect(0, ry, rx, rh);
    ctx.fillRect(rx + rw, ry, Math.max(0, w - (rx + rw)), rh);

    // 二重境界枠（白＋青破線）
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.strokeStyle = '#2563eb';
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);

    // 三分割グリッド線
    if (rw >= 36 && rh >= 36) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
      ctx.beginPath();
      ctx.moveTo(rx + rw / 3, ry);
      ctx.lineTo(rx + rw / 3, ry + rh);
      ctx.moveTo(rx + (rw * 2) / 3, ry);
      ctx.lineTo(rx + (rw * 2) / 3, ry + rh);
      ctx.moveTo(rx, ry + rh / 3);
      ctx.lineTo(rx + rw, ry + rh / 3);
      ctx.moveTo(rx, ry + (rh * 2) / 3);
      ctx.lineTo(rx + rw, ry + (rh * 2) / 3);
      ctx.stroke();
    }

    // 8方向リサイズハンドル（四隅＋4辺中央）
    const handles = this._getHandlePositions(rx, ry, rw, rh);
    for (const hPos of Object.values(handles)) {
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(hPos.x, hPos.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // トリミング寸法ピル
    const baseDims = this.engine.getActiveBaseDimensions();
    const pxW = Math.max(4, Math.round(this.draftCropNorm.w * baseDims.width));
    const pxH = Math.max(4, Math.round(this.draftCropNorm.h * baseDims.height));
    const dimText = `${pxW} × ${pxH} px`;
    ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    const textW = ctx.measureText(dimText).width;
    const pillW = textW + 14;
    const pillH = 20;
    const pillX = Math.max(4, Math.min(w - pillW - 4, rx + 4));
    const pillY = ry >= pillH + 6 ? ry - pillH - 4 : ry + 6;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
    this._roundRect(ctx, pillX, pillY, pillW, pillH, 6);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(dimText, pillX + 7, pillY + pillH / 2);
  }

  _drawMosaicOverlay(ctx, w, h) {
    const regions = this.engine.getMosaicRegions();
    regions.forEach((r, idx) => {
      const rx = r.x * w;
      const ry = r.y * h;
      const rw = r.w * w;
      const rh = r.h * h;
      const isSelected = idx === this.selectedMosaicIdx;

      ctx.lineWidth = isSelected ? 2.5 : 1.75;
      ctx.strokeStyle = isSelected ? '#dc2626' : '#2563eb';
      ctx.fillStyle = isSelected ? 'rgba(220, 38, 38, 0.16)' : 'rgba(37, 99, 235, 0.12)';
      ctx.setLineDash(isSelected ? [] : [5, 4]);
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.setLineDash([]);

      // 番号バッジ
      const badgeText = `#${idx + 1}`;
      ctx.font = '700 10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      const tw = ctx.measureText(badgeText).width;
      const bw = tw + 8;
      ctx.fillStyle = isSelected ? '#dc2626' : '#2563eb';
      this._roundRect(ctx, rx + 2, ry + 2, bw, 16, 4);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.textBaseline = 'middle';
      ctx.fillText(badgeText, rx + 6, ry + 10);
    });

    if (this._currentMosaicNorm) {
      const m = this._currentMosaicNorm;
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#dc2626';
      ctx.fillStyle = 'rgba(220, 38, 38, 0.18)';
      ctx.setLineDash([]);
      ctx.fillRect(m.x * w, m.y * h, m.w * w, m.h * h);
      ctx.strokeRect(m.x * w, m.y * h, m.w * w, m.h * h);
    }
  }

  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  _getHandlePositions(rx, ry, rw, rh) {
    return {
      nw: { x: rx, y: ry },
      n:  { x: rx + rw / 2, y: ry },
      ne: { x: rx + rw, y: ry },
      e:  { x: rx + rw, y: ry + rh / 2 },
      se: { x: rx + rw, y: ry + rh },
      s:  { x: rx + rw / 2, y: ry + rh },
      sw: { x: rx, y: ry + rh },
      w:  { x: rx, y: ry + rh / 2 }
    };
  }

  // ====== ポインターイベント ＆ ヒットテスト ======

  _getCssAndNormPoint(e) {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return { css: { x: 0, y: 0 }, norm: { x: 0, y: 0 } };
    }
    const cx = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const cy = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
    return {
      css: { x: cx, y: cy },
      norm: { x: cx / rect.width, y: cy / rect.height }
    };
  }

  _hitTestCropAction(cssPt) {
    const w = this._displayW;
    const h = this._displayH;
    const rx = this.draftCropNorm.x * w;
    const ry = this.draftCropNorm.y * h;
    const rw = this.draftCropNorm.w * w;
    const rh = this.draftCropNorm.h * h;

    const handles = this._getHandlePositions(rx, ry, rw, rh);
    let bestHandle = null;
    let bestDist = HANDLE_HIT_RADIUS_PX;
    for (const [key, pos] of Object.entries(handles)) {
      const dist = Math.hypot(cssPt.x - pos.x, cssPt.y - pos.y);
      if (dist <= bestDist) {
        bestDist = dist;
        bestHandle = key;
      }
    }
    if (bestHandle) return bestHandle;

    if (cssPt.x >= rx && cssPt.x <= rx + rw && cssPt.y >= ry && cssPt.y <= ry + rh) {
      return 'move-crop';
    }
    return 'new-crop';
  }

  _updateCursor(cssPt) {
    if (this.isComparing || this.mode === 'none') {
      this.canvas.style.cursor = 'default';
      return;
    }
    if (this.mode === 'mosaic') {
      this.canvas.style.cursor = 'crosshair';
      return;
    }
    if (cssPt.x < 0) {
      this.canvas.style.cursor = 'crosshair';
      return;
    }
    const action = this._hitTestCropAction(cssPt);
    const cursorMap = {
      nw: 'nwse-resize',
      se: 'nwse-resize',
      ne: 'nesw-resize',
      sw: 'nesw-resize',
      n: 'ns-resize',
      s: 'ns-resize',
      e: 'ew-resize',
      w: 'ew-resize',
      'move-crop': 'move',
      'new-crop': 'crosshair'
    };
    this.canvas.style.cursor = cursorMap[action] || 'crosshair';
  }

  _bindEvents() {
    this.canvas.addEventListener('pointerdown', (e) => {
      if (!this.engine.hasImage() || this.isComparing || this.mode === 'none') return;
      e.preventDefault();
      try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
      const { css, norm } = this._getCssAndNormPoint(e);
      this._dragStartCss = css;
      this._dragStartNorm = norm;

      if (this.mode === 'crop') {
        this._dragAction = this._hitTestCropAction(css);
        this._dragStartCrop = { ...this.draftCropNorm };
        if (this._dragAction === 'new-crop') {
          this.draftCropNorm = { x: norm.x, y: norm.y, w: 0.005, h: 0.005 };
          this.render();
        }
      } else if (this.mode === 'mosaic') {
        this._dragAction = 'new-mosaic';
        this._currentMosaicNorm = { x: norm.x, y: norm.y, w: 0, h: 0 };
        this.render();
      }
    });

    this.canvas.addEventListener('pointermove', (e) => {
      const { css, norm } = this._getCssAndNormPoint(e);
      if (!this._dragAction) {
        this._updateCursor(css);
        return;
      }

      e.preventDefault();
      if (this.mode === 'crop') {
        this._handleCropDragMove(norm);
      } else if (this._dragAction === 'new-mosaic') {
        this._currentMosaicNorm = {
          x: Math.min(this._dragStartNorm.x, norm.x),
          y: Math.min(this._dragStartNorm.y, norm.y),
          w: Math.abs(norm.x - this._dragStartNorm.x),
          h: Math.abs(norm.y - this._dragStartNorm.y)
        };
        this.render();
      }
    });

    const endDrag = (e) => {
      if (!this._dragAction) return;
      try {
        if (this.canvas.hasPointerCapture(e.pointerId)) {
          this.canvas.releasePointerCapture(e.pointerId);
        }
      } catch (_) {}

      const { css, norm } = this._getCssAndNormPoint(e);
      const movedPx = Math.hypot(css.x - this._dragStartCss.x, css.y - this._dragStartCss.y);

      if (this.mode === 'crop') {
        // 極小ドラッグ誤操作だった場合は元の枠へ戻す
        if (this._dragAction === 'new-crop' && movedPx < 5) {
          this.draftCropNorm = { ...this._dragStartCrop };
        }
        this.onDraftCropChange({ ...this.draftCropNorm });
      } else if (this._dragAction === 'new-mosaic') {
        if (movedPx < 5) {
          // クリック時は既存モザイク枠の選択切替
          const regions = this.engine.getMosaicRegions();
          let hitIdx = -1;
          for (let i = regions.length - 1; i >= 0; i--) {
            const r = regions[i];
            if (norm.x >= r.x && norm.x <= r.x + r.w && norm.y >= r.y && norm.y <= r.y + r.h) {
              hitIdx = i;
              break;
            }
          }
          this.selectedMosaicIdx = (hitIdx === this.selectedMosaicIdx) ? -1 : hitIdx;
          this.onMosaicSelect(this.selectedMosaicIdx);
        } else if (this._currentMosaicNorm && this._currentMosaicNorm.w > 0.005 && this._currentMosaicNorm.h > 0.005) {
          this.onMosaicAdd({ ...this._currentMosaicNorm });
        }
        this._currentMosaicNorm = null;
      }

      this._dragAction = null;
      this.render();
    };

    this.canvas.addEventListener('pointerup', endDrag);
    this.canvas.addEventListener('pointercancel', endDrag);
  }

  _handleCropDragMove(norm) {
    const dims = this.engine.getActiveBaseDimensions();
    const minNormW = Math.min(0.25, Math.max(4, dims.width * 0.01) / dims.width);
    const minNormH = Math.min(0.25, Math.max(4, dims.height * 0.01) / dims.height);
    const ratio = this.aspectRatio; // 実ピクセル w / h
    // 正規化座標上での w_norm / h_norm 比: ratio * (dims.height / dims.width)
    const normRatio = ratio ? ratio * (dims.height / dims.width) : null;

    if (this._dragAction === 'move-crop') {
      const dx = norm.x - this._dragStartNorm.x;
      const dy = norm.y - this._dragStartNorm.y;
      const nx = Math.max(0, Math.min(1 - this._dragStartCrop.w, this._dragStartCrop.x + dx));
      const ny = Math.max(0, Math.min(1 - this._dragStartCrop.h, this._dragStartCrop.y + dy));
      this.draftCropNorm = { ...this._dragStartCrop, x: nx, y: ny };
      this.onDraftCropChange({ ...this.draftCropNorm });
      this.render();
      return;
    }

    if (this._dragAction === 'new-crop') {
      let w = Math.max(minNormW, Math.abs(norm.x - this._dragStartNorm.x));
      let h = Math.max(minNormH, Math.abs(norm.y - this._dragStartNorm.y));
      if (normRatio) {
        h = w / normRatio;
        const maxH = norm.y >= this._dragStartNorm.y ? (1 - this._dragStartNorm.y) : this._dragStartNorm.y;
        if (h > maxH) {
          h = maxH;
          w = h * normRatio;
        }
      }
      const x = norm.x >= this._dragStartNorm.x ? this._dragStartNorm.x : Math.max(0, this._dragStartNorm.x - w);
      const y = norm.y >= this._dragStartNorm.y ? this._dragStartNorm.y : Math.max(0, this._dragStartNorm.y - h);
      this.draftCropNorm = {
        x: Math.max(0, Math.min(1 - minNormW, x)),
        y: Math.max(0, Math.min(1 - minNormH, y)),
        w: Math.max(minNormW, Math.min(1 - x, w)),
        h: Math.max(minNormH, Math.min(1 - y, h))
      };
      this.onDraftCropChange({ ...this.draftCropNorm });
      this.render();
      return;
    }

    // 8方向リサイズハンドル処理
    const start = this._dragStartCrop;
    let left = start.x;
    let top = start.y;
    let right = start.x + start.w;
    let bottom = start.y + start.h;
    const act = this._dragAction;

    if (act.includes('w')) left = Math.max(0, Math.min(right - minNormW, norm.x));
    if (act.includes('e')) right = Math.min(1, Math.max(left + minNormW, norm.x));
    if (act.includes('n')) top = Math.max(0, Math.min(bottom - minNormH, norm.y));
    if (act.includes('s')) bottom = Math.min(1, Math.max(top + minNormH, norm.y));

    if (normRatio) {
      let nw = right - left;
      let nh = bottom - top;
      if (act === 'n' || act === 's') {
        nw = nh * normRatio;
        const cx = (left + right) / 2;
        left = Math.max(0, cx - nw / 2);
        right = left + nw;
        if (right > 1) {
          right = 1;
          left = Math.max(0, 1 - nw);
          nw = right - left;
          nh = nw / normRatio;
          if (act === 'n') top = bottom - nh;
          else bottom = top + nh;
        }
      } else {
        nh = nw / normRatio;
        if (act.includes('n')) {
          top = bottom - nh;
          if (top < 0) {
            top = 0;
            nh = bottom;
            nw = nh * normRatio;
            if (act.includes('w')) left = right - nw;
            else right = left + nw;
          }
        } else {
          bottom = top + nh;
          if (bottom > 1) {
            bottom = 1;
            nh = 1 - top;
            nw = nh * normRatio;
            if (act.includes('w')) left = right - nw;
            else right = left + nw;
          }
        }
      }
    }

    this.draftCropNorm = {
      x: left,
      y: top,
      w: Math.max(minNormW, right - left),
      h: Math.max(minNormH, bottom - top)
    };
    this.onDraftCropChange({ ...this.draftCropNorm });
    this.render();
  }
}
