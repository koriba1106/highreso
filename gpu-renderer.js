/**
 * gpu-renderer.js
 * WebGPU（最優先）および WebGL2（次点）による高速GPU画像処理パイプライン。
 * CPU側への毎フレームの getImageData() 読み戻しを行わず、すべての画像処理パスをGPUテクスチャ上で完結させる。
 *
 * 実行優先順位:
 *   1. WebGPU (WGSL フラグメントパイプライン)
 *   2. WebGL2 (GLSL ES 3.00 マルチパスFBOパイプライン)
 *   3. 非対応・上限超過時は呼び出し元（engine.js）が cpu-worker.js へ自動フォールバック
 */

const MAX_MOSAIC_REGIONS = 64;

// ====== WebGPU シェーダ定義 (WGSL) ======

const WGSL_COMMON_VERTEX = /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  let p = positions[vid];
  var out: VSOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  // WebGPUテクスチャのUV原点(0,0)は左上のため、NDCのY軸を反転してマッピングする
  out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return out;
}
`;

const WGSL_RESIZE_BC_FRAG = /* wgsl */ `
struct ResizeParams {
  srcSize: vec2<f32>,
  dstSize: vec2<f32>,
  cFactor: f32,
  bOffsetNorm: f32,
  invScaleMax: f32,
  _pad: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: ResizeParams;

fn catmullWeight(x: f32) -> f32 {
  let ax = abs(x);
  if (ax <= 1.0) {
    return (1.5 * ax - 2.5) * ax * ax + 1.0;
  } else if (ax < 2.0) {
    return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0;
  }
  return 0.0;
}

fn sampleBicubic(uv: vec2<f32>) -> vec4<f32> {
  let dims = vec2<i32>(params.srcSize);
  let fcoord = uv * params.srcSize - vec2<f32>(0.5);
  let base = floor(fcoord);
  let frac = fcoord - base;
  let ibase = vec2<i32>(base);

  var colorAcc = vec3<f32>(0.0);
  var alphaAcc = 0.0;
  var colorWeightSum = 0.0;
  var totalWeight = 0.0;

  for (var j: i32 = -1; j <= 2; j = j + 1) {
    let wy = catmullWeight(f32(j) - frac.y);
    let sy = clamp(ibase.y + j, 0, dims.y - 1);
    for (var i: i32 = -1; i <= 2; i = i + 1) {
      let wx = catmullWeight(f32(i) - frac.x);
      let w = wx * wy;
      let sx = clamp(ibase.x + i, 0, dims.x - 1);
      let texel = textureLoad(srcTex, vec2<i32>(sx, sy), 0);
      alphaAcc = alphaAcc + texel.a * w;
      totalWeight = totalWeight + w;
      // 透明ピクセルの色が輪郭に混ざる黒フリンジを防ぐため、不透明画素の重みで正規化する
      if (texel.a > 0.001) {
        let cw = w * texel.a;
        colorAcc = colorAcc + texel.rgb * cw;
        colorWeightSum = colorWeightSum + cw;
      }
    }
  }

  let a = clamp(alphaAcc / max(totalWeight, 0.0001), 0.0, 1.0);
  let rgb = select(vec3<f32>(0.0), clamp(colorAcc / colorWeightSum, vec3<f32>(0.0), vec3<f32>(1.0)), colorWeightSum > 0.0001);
  return vec4<f32>(rgb, a);
}

fn sampleDownscale(uv: vec2<f32>) -> vec4<f32> {
  // 縮小倍率に応じて最大6x6のスーパーサンプリングでモアレを防ぐ
  let taps = clamp(i32(ceil(params.invScaleMax)), 2, 6);
  let stepUv = (1.0 / params.dstSize) / f32(taps);
  let startUv = uv - 0.5 * (1.0 / params.dstSize) + 0.5 * stepUv;

  var colorAcc = vec3<f32>(0.0);
  var alphaAcc = 0.0;
  var alphaColorWeight = 0.0;
  var count = 0.0;

  for (var y: i32 = 0; y < 6; y = y + 1) {
    if (y >= taps) { break; }
    for (var x: i32 = 0; x < 6; x = x + 1) {
      if (x >= taps) { break; }
      let suv = startUv + vec2<f32>(f32(x), f32(y)) * stepUv;
      let texel = textureSampleLevel(srcTex, srcSampler, suv, 0.0);
      alphaAcc = alphaAcc + texel.a;
      if (texel.a > 0.001) {
        colorAcc = colorAcc + texel.rgb * texel.a;
        alphaColorWeight = alphaColorWeight + texel.a;
      }
      count = count + 1.0;
    }
  }

  let a = clamp(alphaAcc / max(count, 1.0), 0.0, 1.0);
  let rgb = select(vec3<f32>(0.0), clamp(colorAcc / alphaColorWeight, vec3<f32>(0.0), vec3<f32>(1.0)), alphaColorWeight > 0.0001);
  return vec4<f32>(rgb, a);
}

@fragment
fn fsMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  var c: vec4<f32>;
  if (params.invScaleMax > 1.25) {
    c = sampleDownscale(uv);
  } else {
    c = sampleBicubic(uv);
  }

  if (c.a > 0.001) {
    let adjusted = params.cFactor * (c.rgb + vec3<f32>(params.bOffsetNorm) - vec3<f32>(0.5)) + vec3<f32>(0.5);
    c = vec4<f32>(clamp(adjusted, vec3<f32>(0.0), vec3<f32>(1.0)), c.a);
  }
  return c;
}
`;

const WGSL_CLARITY_FRAG = /* wgsl */ `
struct ClarityParams {
  srGain: f32,
  clarityGain: f32,
  hasSR: f32,
  hasDetail: f32,
};

@group(0) @binding(0) var inTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: ClarityParams;

fn loadSafeRgb(coord: vec2<i32>, dims: vec2<i32>, fallbackRgb: vec3<f32>) -> vec3<f32> {
  let c = textureLoad(inTex, clamp(coord, vec2<i32>(0), dims - vec2<i32>(1)), 0);
  return select(fallbackRgb, c.rgb, c.a > 0.001);
}

@fragment
fn fsMain(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = vec2<i32>(textureDimensions(inTex));
  let ip = vec2<i32>(pos.xy);
  let center = textureLoad(inTex, clamp(ip, vec2<i32>(0), dims - vec2<i32>(1)), 0);
  if (center.a <= 0.001) {
    return center;
  }

  let tl = loadSafeRgb(ip + vec2<i32>(-1, -1), dims, center.rgb);
  let t  = loadSafeRgb(ip + vec2<i32>( 0, -1), dims, center.rgb);
  let tr = loadSafeRgb(ip + vec2<i32>( 1, -1), dims, center.rgb);
  let l  = loadSafeRgb(ip + vec2<i32>(-1,  0), dims, center.rgb);
  let r  = loadSafeRgb(ip + vec2<i32>( 1,  0), dims, center.rgb);
  let bl = loadSafeRgb(ip + vec2<i32>(-1,  1), dims, center.rgb);
  let b  = loadSafeRgb(ip + vec2<i32>( 0,  1), dims, center.rgb);
  let br = loadSafeRgb(ip + vec2<i32>( 1,  1), dims, center.rgb);

  let localMean = (tl + tr + bl + br + (t + b + l + r) * 2.0 + center.rgb * 4.0) / 16.0;
  let diff = center.rgb - localMean;
  let absDiff = abs(diff);

  let midWeight = max(vec3<f32>(0.15), vec3<f32>(1.0) - abs(center.rgb - vec3<f32>(0.5)) / (140.0 / 255.0));

  var delta = vec3<f32>(0.0);
  if (params.hasSR > 0.5) {
    let maskSR = absDiff >= vec3<f32>(2.0 / 255.0);
    delta = delta + select(vec3<f32>(0.0), diff * params.srGain, maskSR);
  }
  if (params.hasDetail > 0.5) {
    let maskDet = absDiff >= vec3<f32>(4.0 / 255.0);
    delta = delta + select(vec3<f32>(0.0), diff * params.clarityGain * midWeight, maskDet);
  }

  return vec4<f32>(clamp(center.rgb + delta, vec3<f32>(0.0), vec3<f32>(1.0)), center.a);
}
`;

const WGSL_SHARP_FRAG = /* wgsl */ `
struct SharpParams {
  amountDiv6: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var inTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: SharpParams;

fn loadSafeRgb(coord: vec2<i32>, dims: vec2<i32>, fallbackRgb: vec3<f32>) -> vec3<f32> {
  let c = textureLoad(inTex, clamp(coord, vec2<i32>(0), dims - vec2<i32>(1)), 0);
  return select(fallbackRgb, c.rgb, c.a > 0.001);
}

@fragment
fn fsMain(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = vec2<i32>(textureDimensions(inTex));
  let ip = vec2<i32>(pos.xy);
  let center = textureLoad(inTex, clamp(ip, vec2<i32>(0), dims - vec2<i32>(1)), 0);
  if (center.a <= 0.001) {
    return center;
  }

  let tl = loadSafeRgb(ip + vec2<i32>(-1, -1), dims, center.rgb);
  let t  = loadSafeRgb(ip + vec2<i32>( 0, -1), dims, center.rgb);
  let tr = loadSafeRgb(ip + vec2<i32>( 1, -1), dims, center.rgb);
  let l  = loadSafeRgb(ip + vec2<i32>(-1,  0), dims, center.rgb);
  let r  = loadSafeRgb(ip + vec2<i32>( 1,  0), dims, center.rgb);
  let bl = loadSafeRgb(ip + vec2<i32>(-1,  1), dims, center.rgb);
  let b  = loadSafeRgb(ip + vec2<i32>( 0,  1), dims, center.rgb);
  let br = loadSafeRgb(ip + vec2<i32>( 1,  1), dims, center.rgb);

  let neighbors = t + b + l + r + (tl + tr + bl + br) * 0.5;
  let hp6 = center.rgb * 6.0 - neighbors;
  let mask = abs(hp6) > vec3<f32>(12.0 / 255.0);
  let outRgb = clamp(center.rgb + select(vec3<f32>(0.0), hp6 * params.amountDiv6, mask), vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(outRgb, center.a);
}
`;

const WGSL_MOSAIC_PRESENT_FRAG = /* wgsl */ `
struct MosaicParams {
  imgSize: vec2<f32>,
  blockSize: f32,
  regionCount: u32,
  regions: array<vec4<f32>, ${MAX_MOSAIC_REGIONS}>,
};

@group(0) @binding(0) var inTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: MosaicParams;

@fragment
fn fsMain(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = vec2<i32>(textureDimensions(inTex));
  let ip = clamp(vec2<i32>(pos.xy), vec2<i32>(0), dims - vec2<i32>(1));
  var color = textureLoad(inTex, ip, 0);

  if (params.regionCount > 0u && params.blockSize >= 2.0) {
    let fpix = vec2<f32>(ip);
    let uv = (fpix + vec2<f32>(0.5)) / params.imgSize;
    var matched = false;
    var matchedRect = vec4<f32>(0.0);

    for (var i: u32 = 0u; i < params.regionCount; i = i + 1u) {
      let reg = params.regions[i];
      if (uv.x >= reg.x && uv.x < (reg.x + reg.z) && uv.y >= reg.y && uv.y < (reg.y + reg.w)) {
        matched = true;
        matchedRect = reg;
      }
    }

    if (matched) {
      let startPx = floor(matchedRect.xy * params.imgSize);
      let endPx = min(params.imgSize, ceil((matchedRect.xy + matchedRect.zw) * params.imgSize));
      let blockIdx = floor((fpix - startPx) / params.blockSize);
      let bStart = startPx + blockIdx * params.blockSize;
      let bEnd = min(endPx, bStart + vec2<f32>(params.blockSize));
      let bSpan = max(vec2<f32>(1.0), bEnd - bStart);

      var sumRgb = vec3<f32>(0.0);
      var sumA = 0.0;
      var opaqueSum = 0.0;
      var count = 0.0;

      // ブロック内の4x4代表点をサンプリングしてブロック平均色を算出
      for (var sy: i32 = 0; sy < 4; sy = sy + 1) {
        let py = bStart.y + ((f32(sy) + 0.5) * 0.25) * bSpan.y;
        for (var sx: i32 = 0; sx < 4; sx = sx + 1) {
          let px = bStart.x + ((f32(sx) + 0.5) * 0.25) * bSpan.x;
          let sc = clamp(vec2<i32>(floor(vec2<f32>(px, py))), vec2<i32>(0), dims - vec2<i32>(1));
          let s = textureLoad(inTex, sc, 0);
          sumA = sumA + s.a;
          if (s.a > 0.001) {
            sumRgb = sumRgb + s.rgb * s.a;
            opaqueSum = opaqueSum + s.a;
          }
          count = count + 1.0;
        }
      }

      let avgA = clamp(sumA / count, 0.0, 1.0);
      let avgRgb = select(vec3<f32>(0.0), clamp(sumRgb / opaqueSum, vec3<f32>(0.0), vec3<f32>(1.0)), opaqueSum > 0.001);
      color = vec4<f32>(avgRgb, avgA);
    }
  }

  // WebGPUキャンバスの alphaMode: 'premultiplied' に合わせて最終出力のみアルファ乗算を行う
  return vec4<f32>(color.rgb * color.a, color.a);
}
`;

// ====== WebGL2 シェーダ定義 (GLSL ES 3.00) ======

const GLSL_COMMON_VERT = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 pos[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
  );
  vec2 p = pos[gl_VertexID];
  gl_Position = vec4(p, 0.0, 1.0);
  vUv = p * 0.5 + 0.5;
}
`;

const GLSL_RESIZE_BC_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uSrcTex;
uniform vec2 uSrcSize;
uniform vec2 uDstSize;
uniform float uCFactor;
uniform float uBOffsetNorm;
uniform float uInvScaleMax;
in vec2 vUv;
out vec4 fragColor;

float catmullWeight(float x) {
  float ax = abs(x);
  if (ax <= 1.0) {
    return (1.5 * ax - 2.5) * ax * ax + 1.0;
  } else if (ax < 2.0) {
    return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0;
  }
  return 0.0;
}

vec4 sampleBicubic(vec2 uv) {
  ivec2 dims = ivec2(uSrcSize);
  vec2 fcoord = uv * uSrcSize - vec2(0.5);
  vec2 base = floor(fcoord);
  vec2 frac = fcoord - base;
  ivec2 ibase = ivec2(base);

  vec3 colorAcc = vec3(0.0);
  float alphaAcc = 0.0;
  float colorWeightSum = 0.0;
  float totalWeight = 0.0;

  for (int j = -1; j <= 2; j++) {
    float wy = catmullWeight(float(j) - frac.y);
    int sy = clamp(ibase.y + j, 0, dims.y - 1);
    for (int i = -1; i <= 2; i++) {
      float wx = catmullWeight(float(i) - frac.x);
      float w = wx * wy;
      int sx = clamp(ibase.x + i, 0, dims.x - 1);
      vec4 texel = texelFetch(uSrcTex, ivec2(sx, sy), 0);
      alphaAcc += texel.a * w;
      totalWeight += w;
      if (texel.a > 0.001) {
        float cw = w * texel.a;
        colorAcc += texel.rgb * cw;
        colorWeightSum += cw;
      }
    }
  }

  float a = clamp(alphaAcc / max(totalWeight, 0.0001), 0.0, 1.0);
  vec3 rgb = colorWeightSum > 0.0001 ? clamp(colorAcc / colorWeightSum, 0.0, 1.0) : vec3(0.0);
  return vec4(rgb, a);
}

vec4 sampleDownscale(vec2 uv) {
  // ミップマップ生成済みの入力テクスチャからバイリニア補間＋適応サンプリングで滑らかに縮小
  int taps = clamp(int(ceil(uInvScaleMax * 0.5)), 2, 4);
  vec2 stepUv = (1.0 / uDstSize) / float(taps);
  vec2 startUv = uv - 0.5 * (1.0 / uDstSize) + 0.5 * stepUv;

  vec3 colorAcc = vec3(0.0);
  float alphaAcc = 0.0;
  float alphaWeight = 0.0;
  float count = 0.0;

  for (int y = 0; y < 4; y++) {
    if (y >= taps) break;
    for (int x = 0; x < 4; x++) {
      if (x >= taps) break;
      vec4 texel = texture(uSrcTex, startUv + vec2(float(x), float(y)) * stepUv);
      alphaAcc += texel.a;
      if (texel.a > 0.001) {
        colorAcc += texel.rgb * texel.a;
        alphaWeight += texel.a;
      }
      count += 1.0;
    }
  }

  float a = clamp(alphaAcc / max(count, 1.0), 0.0, 1.0);
  vec3 rgb = alphaWeight > 0.0001 ? clamp(colorAcc / alphaWeight, 0.0, 1.0) : vec3(0.0);
  return vec4(rgb, a);
}

void main() {
  vec4 c = uInvScaleMax > 1.25 ? sampleDownscale(vUv) : sampleBicubic(vUv);
  if (c.a > 0.001) {
    vec3 adjusted = uCFactor * (c.rgb + vec3(uBOffsetNorm) - vec3(0.5)) + vec3(0.5);
    c = vec4(clamp(adjusted, 0.0, 1.0), c.a);
  }
  fragColor = c;
}
`;

const GLSL_CLARITY_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uInTex;
uniform float uSrGain;
uniform float uClarityGain;
uniform float uHasSR;
uniform float uHasDetail;
out vec4 fragColor;

vec3 loadSafeRgb(ivec2 coord, ivec2 dims, vec3 fallbackRgb) {
  vec4 c = texelFetch(uInTex, clamp(coord, ivec2(0), dims - ivec2(1)), 0);
  return c.a > 0.001 ? c.rgb : fallbackRgb;
}

void main() {
  ivec2 dims = textureSize(uInTex, 0);
  ivec2 ip = clamp(ivec2(gl_FragCoord.xy), ivec2(0), dims - ivec2(1));
  vec4 center = texelFetch(uInTex, ip, 0);
  if (center.a <= 0.001) {
    fragColor = center;
    return;
  }

  vec3 tl = loadSafeRgb(ip + ivec2(-1, -1), dims, center.rgb);
  vec3 t  = loadSafeRgb(ip + ivec2( 0, -1), dims, center.rgb);
  vec3 tr = loadSafeRgb(ip + ivec2( 1, -1), dims, center.rgb);
  vec3 l  = loadSafeRgb(ip + ivec2(-1,  0), dims, center.rgb);
  vec3 r  = loadSafeRgb(ip + ivec2( 1,  0), dims, center.rgb);
  vec3 bl = loadSafeRgb(ip + ivec2(-1,  1), dims, center.rgb);
  vec3 b  = loadSafeRgb(ip + ivec2( 0,  1), dims, center.rgb);
  vec3 br = loadSafeRgb(ip + ivec2( 1,  1), dims, center.rgb);

  vec3 localMean = (tl + tr + bl + br + (t + b + l + r) * 2.0 + center.rgb * 4.0) / 16.0;
  vec3 diff = center.rgb - localMean;
  vec3 absDiff = abs(diff);
  vec3 midWeight = max(vec3(0.15), vec3(1.0) - abs(center.rgb - vec3(0.5)) / (140.0 / 255.0));

  vec3 delta = vec3(0.0);
  if (uHasSR > 0.5) {
    bvec3 maskSR = greaterThanEqual(absDiff, vec3(2.0 / 255.0));
    delta += mix(vec3(0.0), diff * uSrGain, maskSR);
  }
  if (uHasDetail > 0.5) {
    bvec3 maskDet = greaterThanEqual(absDiff, vec3(4.0 / 255.0));
    delta += mix(vec3(0.0), diff * uClarityGain * midWeight, maskDet);
  }

  fragColor = vec4(clamp(center.rgb + delta, 0.0, 1.0), center.a);
}
`;

const GLSL_SHARP_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uInTex;
uniform float uAmountDiv6;
out vec4 fragColor;

vec3 loadSafeRgb(ivec2 coord, ivec2 dims, vec3 fallbackRgb) {
  vec4 c = texelFetch(uInTex, clamp(coord, ivec2(0), dims - ivec2(1)), 0);
  return c.a > 0.001 ? c.rgb : fallbackRgb;
}

void main() {
  ivec2 dims = textureSize(uInTex, 0);
  ivec2 ip = clamp(ivec2(gl_FragCoord.xy), ivec2(0), dims - ivec2(1));
  vec4 center = texelFetch(uInTex, ip, 0);
  if (center.a <= 0.001) {
    fragColor = center;
    return;
  }

  vec3 tl = loadSafeRgb(ip + ivec2(-1, -1), dims, center.rgb);
  vec3 t  = loadSafeRgb(ip + ivec2( 0, -1), dims, center.rgb);
  vec3 tr = loadSafeRgb(ip + ivec2( 1, -1), dims, center.rgb);
  vec3 l  = loadSafeRgb(ip + ivec2(-1,  0), dims, center.rgb);
  vec3 r  = loadSafeRgb(ip + ivec2( 1,  0), dims, center.rgb);
  vec3 bl = loadSafeRgb(ip + ivec2(-1,  1), dims, center.rgb);
  vec3 b  = loadSafeRgb(ip + ivec2( 0,  1), dims, center.rgb);
  vec3 br = loadSafeRgb(ip + ivec2( 1,  1), dims, center.rgb);

  vec3 neighbors = t + b + l + r + (tl + tr + bl + br) * 0.5;
  vec3 hp6 = center.rgb * 6.0 - neighbors;
  bvec3 mask = greaterThan(abs(hp6), vec3(12.0 / 255.0));
  vec3 outRgb = clamp(center.rgb + mix(vec3(0.0), hp6 * uAmountDiv6, mask), 0.0, 1.0);
  fragColor = vec4(outRgb, center.a);
}
`;

const GLSL_MOSAIC_PRESENT_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uInTex;
uniform vec2 uImgSize;
uniform float uBlockSize;
uniform int uRegionCount;
uniform vec4 uRegions[${MAX_MOSAIC_REGIONS}];
in vec2 vUv;
out vec4 fragColor;

void main() {
  ivec2 dims = textureSize(uInTex, 0);
  ivec2 ip = clamp(ivec2(gl_FragCoord.xy), ivec2(0), dims - ivec2(1));
  vec4 color = texelFetch(uInTex, ip, 0);

  if (uRegionCount > 0 && uBlockSize >= 2.0) {
    // 領域指定座標系は上端原点 (0,0) だが、WebGLの gl_FragCoord.y は下端原点のため反転して合わせる
    vec2 topDownUv = vec2(vUv.x, 1.0 - vUv.y);
    vec2 topDownPx = floor(topDownUv * uImgSize);
    bool matched = false;
    vec4 matchedRect = vec4(0.0);

    for (int i = 0; i < ${MAX_MOSAIC_REGIONS}; i++) {
      if (i >= uRegionCount) break;
      vec4 reg = uRegions[i];
      if (topDownUv.x >= reg.x && topDownUv.x < (reg.x + reg.z) &&
          topDownUv.y >= reg.y && topDownUv.y < (reg.y + reg.w)) {
        matched = true;
        matchedRect = reg;
      }
    }

    if (matched) {
      vec2 startPx = floor(matchedRect.xy * uImgSize);
      vec2 endPx = min(uImgSize, ceil((matchedRect.xy + matchedRect.zw) * uImgSize));
      vec2 blockIdx = floor((topDownPx - startPx) / uBlockSize);
      vec2 bStart = startPx + blockIdx * uBlockSize;
      vec2 bEnd = min(endPx, bStart + vec2(uBlockSize));
      vec2 bSpan = max(vec2(1.0), bEnd - bStart);

      vec3 sumRgb = vec3(0.0);
      float sumA = 0.0;
      float opaqueSum = 0.0;
      float count = 0.0;

      for (int sy = 0; sy < 4; sy++) {
        float py = bStart.y + ((float(sy) + 0.5) * 0.25) * bSpan.y;
        for (int sx = 0; sx < 4; sx++) {
          float px = bStart.x + ((float(sx) + 0.5) * 0.25) * bSpan.x;
          // テクスチャフェッチ座標は下端原点に戻す
          int tx = clamp(int(floor(px)), 0, dims.x - 1);
          int ty = clamp(dims.y - 1 - int(floor(py)), 0, dims.y - 1);
          vec4 s = texelFetch(uInTex, ivec2(tx, ty), 0);
          sumA += s.a;
          if (s.a > 0.001) {
            sumRgb += s.rgb * s.a;
            opaqueSum += s.a;
          }
          count += 1.0;
        }
      }

      float avgA = clamp(sumA / count, 0.0, 1.0);
      vec3 avgRgb = opaqueSum > 0.001 ? clamp(sumRgb / opaqueSum, 0.0, 1.0) : vec3(0.0);
      color = vec4(avgRgb, avgA);
    }
  }

  fragColor = vec4(color.rgb * color.a, color.a);
}
`;

export class GPURenderer {
  constructor() {
    /** @type {'webgpu' | 'webgl2' | 'none'} */
    this.backend = 'none';
    this.maxTextureDimension = 0;
    this.supportsShaderF16 = false;

    this._canvas = null;
    this._sourceUploadedVersion = -1;

    // WebGPU リソース
    this._gpu = {
      device: null,
      context: null,
      format: null,
      sampler: null,
      pipelines: null,
      uniforms: null,
      srcTex: null,
      srcW: 0,
      srcH: 0,
      pingTex: null,
      pongTex: null,
      pingPongW: 0,
      pingPongH: 0
    };

    // WebGL2 リソース
    this._gl = {
      ctx: null,
      vao: null,
      programs: null,
      srcTex: null,
      srcW: 0,
      srcH: 0,
      fboA: null,
      texA: null,
      fboB: null,
      texB: null,
      fboW: 0,
      fboH: 0
    };
  }

  /**
   * 実行環境のプローブを行い、WebGPU → WebGL2 の順でGPUレンダラーを初期化する。
   * @returns {Promise<{backend: 'webgpu'|'webgl2'|'none', maxTextureDimension: number, supportsShaderF16: boolean}>}
   */
  async init() {
    if (await this._initWebGPU()) {
      return this.getCapabilities();
    }
    if (this._initWebGL2()) {
      return this.getCapabilities();
    }
    this.backend = 'none';
    return this.getCapabilities();
  }

  getCapabilities() {
    return {
      backend: this.backend,
      maxTextureDimension: this.maxTextureDimension,
      supportsShaderF16: this.supportsShaderF16
    };
  }

  invalidateSourceTexture() {
    this._sourceUploadedVersion = -1;
  }

  // ====== WebGPU 初期化・描画 ======

  async _initWebGPU() {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;

    try {
      const adapter = await Promise.race([
        navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }),
        new Promise(resolve => setTimeout(() => resolve(null), 2000))
      ]);
      if (!adapter) return false;

      this.supportsShaderF16 = Boolean(adapter.features && adapter.features.has('shader-f16'));

      const device = await adapter.requestDevice();
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('webgpu');
      if (!context) {
        device.destroy();
        return false;
      }

      const format = navigator.gpu.getPreferredCanvasFormat();
      const makeModule = code => device.createShaderModule({ code });

      const vsMod = makeModule(WGSL_COMMON_VERTEX);
      const resizeMod = makeModule(WGSL_RESIZE_BC_FRAG);
      const clarityMod = makeModule(WGSL_CLARITY_FRAG);
      const sharpMod = makeModule(WGSL_SHARP_FRAG);
      const mosaicMod = makeModule(WGSL_MOSAIC_PRESENT_FRAG);

      const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      });

      const makePipe = (fragMod, targetFormat) => device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: vsMod, entryPoint: 'vsMain' },
        fragment: { module: fragMod, entryPoint: 'fsMain', targets: [{ format: targetFormat }] },
        primitive: { topology: 'triangle-list' }
      });

      this._canvas = canvas;
      this._gpu.device = device;
      this._gpu.context = context;
      this._gpu.format = format;
      this._gpu.sampler = sampler;
      this._gpu.pipelines = {
        resize: makePipe(resizeMod, 'rgba8unorm'),
        clarity: makePipe(clarityMod, 'rgba8unorm'),
        sharp: makePipe(sharpMod, 'rgba8unorm'),
        mosaicPresent: makePipe(mosaicMod, format)
      };

      const makeUniform = byteSize => device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      this._gpu.uniforms = {
        resize: makeUniform(32),
        clarity: makeUniform(16),
        sharp: makeUniform(16),
        mosaic: makeUniform(16 + MAX_MOSAIC_REGIONS * 16)
      };

      // GPUデバイスロスト時は自動でWebGL2またはCPUへフォールバックできるように状態を降格させる
      device.lost.then(() => {
        if (this.backend === 'webgpu') {
          this._destroyWebGPU();
          if (!this._initWebGL2()) {
            this.backend = 'none';
          }
        }
      });

      this.maxTextureDimension = device.limits.maxTextureDimension2D || 4096;
      this.backend = 'webgpu';
      return true;
    } catch (_) {
      this._destroyWebGPU();
      return false;
    }
  }

  _ensureWebGPUTextures(srcW, srcH, dstW, dstH) {
    const { device } = this._gpu;
    if (!this._gpu.srcTex || this._gpu.srcW !== srcW || this._gpu.srcH !== srcH) {
      if (this._gpu.srcTex) this._gpu.srcTex.destroy();
      this._gpu.srcTex = device.createTexture({
        size: [srcW, srcH],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });
      this._gpu.srcW = srcW;
      this._gpu.srcH = srcH;
      this._sourceUploadedVersion = -1;
    }

    if (!this._gpu.pingTex || this._gpu.pingPongW !== dstW || this._gpu.pingPongH !== dstH) {
      if (this._gpu.pingTex) this._gpu.pingTex.destroy();
      if (this._gpu.pongTex) this._gpu.pongTex.destroy();
      const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT;
      this._gpu.pingTex = device.createTexture({ size: [dstW, dstH], format: 'rgba8unorm', usage });
      this._gpu.pongTex = device.createTexture({ size: [dstW, dstH], format: 'rgba8unorm', usage });
      this._gpu.pingPongW = dstW;
      this._gpu.pingPongH = dstH;
    }

    if (this._canvas.width !== dstW || this._canvas.height !== dstH) {
      this._canvas.width = dstW;
      this._canvas.height = dstH;
      this._gpu.context.configure({
        device,
        format: this._gpu.format,
        alphaMode: 'premultiplied'
      });
    }
  }

  _renderWebGPU(sourceCanvas, sourceVersion, targetW, targetH, params, mosaicRegions, target2DCtx) {
    const srcW = sourceCanvas.width;
    const srcH = sourceCanvas.height;
    if (
      Math.max(srcW, srcH, targetW, targetH) > this.maxTextureDimension ||
      mosaicRegions.length > MAX_MOSAIC_REGIONS
    ) {
      return false;
    }

    const { device, context, sampler, pipelines, uniforms } = this._gpu;
    this._ensureWebGPUTextures(srcW, srcH, targetW, targetH);

    // ベース画像が変化した時のみGPUへテクスチャをアップロード（スライダー調整時は転送ゼロ）
    if (this._sourceUploadedVersion !== sourceVersion) {
      device.queue.copyExternalImageToTexture(
        { source: sourceCanvas, flipY: false },
        { texture: this._gpu.srcTex, premultipliedAlpha: false },
        [srcW, srcH]
      );
      this._sourceUploadedVersion = sourceVersion;
    }

    // パス1 Uniforms
    const cFactor = (259 * (params.contrast + 255)) / (255 * (259 - params.contrast));
    const bOffsetNorm = (params.brightness / 100) * (128 / 255);
    const invScaleMax = Math.max(srcW / targetW, srcH / targetH);
    device.queue.writeBuffer(
      uniforms.resize,
      0,
      new Float32Array([srcW, srcH, targetW, targetH, cFactor, bOffsetNorm, invScaleMax, 0])
    );

    const encoder = device.createCommandEncoder();

    const runPass = (pipeline, entries, targetView) => {
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries
      });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: targetView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store'
        }]
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    };

    // パス1: srcTex -> pingTex
    let currentTex = this._gpu.pingTex;
    let spareTex = this._gpu.pongTex;
    runPass(
      pipelines.resize,
      [
        { binding: 0, resource: this._gpu.srcTex.createView() },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: uniforms.resize } }
      ],
      currentTex.createView()
    );

    // パス2: 高精細化
    if (params.superResolution > 0 || params.detail > 0) {
      const srGain = (params.superResolution / 100) * 0.65;
      const clarityGain = (params.detail / 100) * 1.15;
      device.queue.writeBuffer(
        uniforms.clarity,
        0,
        new Float32Array([
          srGain,
          clarityGain,
          params.superResolution > 0 ? 1 : 0,
          params.detail > 0 ? 1 : 0
        ])
      );
      runPass(
        pipelines.clarity,
        [
          { binding: 0, resource: currentTex.createView() },
          { binding: 1, resource: { buffer: uniforms.clarity } }
        ],
        spareTex.createView()
      );
      [currentTex, spareTex] = [spareTex, currentTex];
    }

    // パス3: シャープネス
    if (params.sharpness >= 1) {
      const amountDiv6 = (params.sharpness * 0.18 + params.sharpness * params.sharpness * 0.002) / 6;
      device.queue.writeBuffer(uniforms.sharp, 0, new Float32Array([amountDiv6, 0, 0, 0]));
      runPass(
        pipelines.sharp,
        [
          { binding: 0, resource: currentTex.createView() },
          { binding: 1, resource: { buffer: uniforms.sharp } }
        ],
        spareTex.createView()
      );
      [currentTex, spareTex] = [spareTex, currentTex];
    }

    // パス4: モザイク＋最終キャンバス出力
    const mosaicBuf = new ArrayBuffer(16 + MAX_MOSAIC_REGIONS * 16);
    const f32View = new Float32Array(mosaicBuf);
    const u32View = new Uint32Array(mosaicBuf);
    f32View[0] = targetW;
    f32View[1] = targetH;
    f32View[2] = params.mosaicSize;
    u32View[3] = mosaicRegions.length;
    for (let i = 0; i < mosaicRegions.length; i++) {
      const r = mosaicRegions[i];
      const base = 4 + i * 4;
      f32View[base] = r.x;
      f32View[base + 1] = r.y;
      f32View[base + 2] = r.w;
      f32View[base + 3] = r.h;
    }
    device.queue.writeBuffer(uniforms.mosaic, 0, mosaicBuf);

    runPass(
      pipelines.mosaicPresent,
      [
        { binding: 0, resource: currentTex.createView() },
        { binding: 1, resource: { buffer: uniforms.mosaic } }
      ],
      context.getCurrentTexture().createView()
    );

    device.queue.submit([encoder.finish()]);
    // GPUキャンバスへのsubmit直後（同一タスク内）に2D出力キャンバスへGPU間コピーする
    target2DCtx.clearRect(0, 0, targetW, targetH);
    target2DCtx.drawImage(this._canvas, 0, 0, targetW, targetH);
    return true;
  }

  _destroyWebGPU() {
    if (this._gpu.srcTex) this._gpu.srcTex.destroy();
    if (this._gpu.pingTex) this._gpu.pingTex.destroy();
    if (this._gpu.pongTex) this._gpu.pongTex.destroy();
    if (this._gpu.uniforms) {
      Object.values(this._gpu.uniforms).forEach(b => b && b.destroy());
    }
    if (this._gpu.device) {
      try { this._gpu.device.destroy(); } catch (_) {}
    }
    this._gpu = {
      device: null, context: null, format: null, sampler: null,
      pipelines: null, uniforms: null, srcTex: null, srcW: 0, srcH: 0,
      pingTex: null, pongTex: null, pingPongW: 0, pingPongH: 0
    };
  }

  // ====== WebGL2 初期化・描画 ======

  _initWebGL2() {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2', {
        alpha: true,
        premultipliedAlpha: true,
        preserveDrawingBuffer: true,
        antialias: false,
        powerPreference: 'high-performance'
      });
      if (!gl) return false;

      const compileShader = (type, src) => {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, src);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
          const log = gl.getShaderInfoLog(shader);
          gl.deleteShader(shader);
          throw new Error(log || 'Shader compile error');
        }
        return shader;
      };

      const makeProgram = (vertSrc, fragSrc) => {
        const vs = compileShader(gl.VERTEX_SHADER, vertSrc);
        const fs = compileShader(gl.FRAGMENT_SHADER, fragSrc);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
          const log = gl.getProgramInfoLog(prog);
          gl.deleteProgram(prog);
          throw new Error(log || 'Program link error');
        }
        return prog;
      };

      const programs = {
        resize: makeProgram(GLSL_COMMON_VERT, GLSL_RESIZE_BC_FRAG),
        clarity: makeProgram(GLSL_COMMON_VERT, GLSL_CLARITY_FRAG),
        sharp: makeProgram(GLSL_COMMON_VERT, GLSL_SHARP_FRAG),
        mosaicPresent: makeProgram(GLSL_COMMON_VERT, GLSL_MOSAIC_PRESENT_FRAG)
      };

      const vao = gl.createVertexArray();
      const srcTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, srcTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);

      this._canvas = canvas;
      this._gl.ctx = gl;
      this._gl.vao = vao;
      this._gl.programs = programs;
      this._gl.srcTex = srcTex;
      this.maxTextureDimension = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 4096;
      this.backend = 'webgl2';
      return true;
    } catch (_) {
      this._destroyWebGL2();
      return false;
    }
  }

  _ensureWebGL2FBOs(dstW, dstH) {
    const gl = this._gl.ctx;
    if (this._gl.fboW === dstW && this._gl.fboH === dstH) return;

    const createTarget = () => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, dstW, dstH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      return { tex, fbo };
    };

    if (this._gl.texA) gl.deleteTexture(this._gl.texA);
    if (this._gl.fboA) gl.deleteFramebuffer(this._gl.fboA);
    if (this._gl.texB) gl.deleteTexture(this._gl.texB);
    if (this._gl.fboB) gl.deleteFramebuffer(this._gl.fboB);

    const a = createTarget();
    const b = createTarget();
    this._gl.texA = a.tex;
    this._gl.fboA = a.fbo;
    this._gl.texB = b.tex;
    this._gl.fboB = b.fbo;
    this._gl.fboW = dstW;
    this._gl.fboH = dstH;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _renderWebGL2(sourceCanvas, sourceVersion, targetW, targetH, params, mosaicRegions, target2DCtx) {
    const gl = this._gl.ctx;
    if (!gl || gl.isContextLost()) return false;

    const srcW = sourceCanvas.width;
    const srcH = sourceCanvas.height;
    if (
      Math.max(srcW, srcH, targetW, targetH) > this.maxTextureDimension ||
      mosaicRegions.length > MAX_MOSAIC_REGIONS
    ) {
      return false;
    }

    if (this._canvas.width !== targetW || this._canvas.height !== targetH) {
      this._canvas.width = targetW;
      this._canvas.height = targetH;
    }
    this._ensureWebGL2FBOs(targetW, targetH);

    // ベース画像変化時のみテクスチャ再転送（Y軸反転してWebGL標準座標に揃える）
    if (this._sourceUploadedVersion !== sourceVersion || this._gl.srcW !== srcW || this._gl.srcH !== srcH) {
      gl.bindTexture(gl.TEXTURE_2D, this._gl.srcTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, sourceCanvas);
      gl.generateMipmap(gl.TEXTURE_2D);
      this._gl.srcW = srcW;
      this._gl.srcH = srcH;
      this._sourceUploadedVersion = sourceVersion;
    }

    gl.bindVertexArray(this._gl.vao);
    gl.viewport(0, 0, targetW, targetH);
    gl.disable(gl.BLEND);

    // パス1: srcTex -> fboA
    let currentTex = this._gl.texA;
    let spareFbo = this._gl.fboB;
    let spareTex = this._gl.texB;
    let currentFbo = this._gl.fboA;

    const pResize = this._gl.programs.resize;
    gl.useProgram(pResize);
    gl.bindFramebuffer(gl.FRAMEBUFFER, currentFbo);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._gl.srcTex);
    gl.uniform1i(gl.getUniformLocation(pResize, 'uSrcTex'), 0);
    gl.uniform2f(gl.getUniformLocation(pResize, 'uSrcSize'), srcW, srcH);
    gl.uniform2f(gl.getUniformLocation(pResize, 'uDstSize'), targetW, targetH);
    gl.uniform1f(
      gl.getUniformLocation(pResize, 'uCFactor'),
      (259 * (params.contrast + 255)) / (255 * (259 - params.contrast))
    );
    gl.uniform1f(gl.getUniformLocation(pResize, 'uBOffsetNorm'), (params.brightness / 100) * (128 / 255));
    gl.uniform1f(gl.getUniformLocation(pResize, 'uInvScaleMax'), Math.max(srcW / targetW, srcH / targetH));
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // パス2: 高精細化
    if (params.superResolution > 0 || params.detail > 0) {
      const pClarity = this._gl.programs.clarity;
      gl.useProgram(pClarity);
      gl.bindFramebuffer(gl.FRAMEBUFFER, spareFbo);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, currentTex);
      gl.uniform1i(gl.getUniformLocation(pClarity, 'uInTex'), 0);
      gl.uniform1f(gl.getUniformLocation(pClarity, 'uSrGain'), (params.superResolution / 100) * 0.65);
      gl.uniform1f(gl.getUniformLocation(pClarity, 'uClarityGain'), (params.detail / 100) * 1.15);
      gl.uniform1f(gl.getUniformLocation(pClarity, 'uHasSR'), params.superResolution > 0 ? 1 : 0);
      gl.uniform1f(gl.getUniformLocation(pClarity, 'uHasDetail'), params.detail > 0 ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      [currentTex, spareTex] = [spareTex, currentTex];
      [currentFbo, spareFbo] = [spareFbo, currentFbo];
    }

    // パス3: シャープネス
    if (params.sharpness >= 1) {
      const pSharp = this._gl.programs.sharp;
      gl.useProgram(pSharp);
      gl.bindFramebuffer(gl.FRAMEBUFFER, spareFbo);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, currentTex);
      gl.uniform1i(gl.getUniformLocation(pSharp, 'uInTex'), 0);
      gl.uniform1f(
        gl.getUniformLocation(pSharp, 'uAmountDiv6'),
        (params.sharpness * 0.18 + params.sharpness * params.sharpness * 0.002) / 6
      );
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      [currentTex, spareTex] = [spareTex, currentTex];
      [currentFbo, spareFbo] = [spareFbo, currentFbo];
    }

    // パス4: モザイク＋最終描画（デフォルトフレームバッファ）
    const pMosaic = this._gl.programs.mosaicPresent;
    gl.useProgram(pMosaic);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentTex);
    gl.uniform1i(gl.getUniformLocation(pMosaic, 'uInTex'), 0);
    gl.uniform2f(gl.getUniformLocation(pMosaic, 'uImgSize'), targetW, targetH);
    gl.uniform1f(gl.getUniformLocation(pMosaic, 'uBlockSize'), params.mosaicSize);
    gl.uniform1i(gl.getUniformLocation(pMosaic, 'uRegionCount'), mosaicRegions.length);

    if (mosaicRegions.length > 0) {
      const flat = new Float32Array(MAX_MOSAIC_REGIONS * 4);
      for (let i = 0; i < mosaicRegions.length; i++) {
        const r = mosaicRegions[i];
        flat[i * 4] = r.x;
        flat[i * 4 + 1] = r.y;
        flat[i * 4 + 2] = r.w;
        flat[i * 4 + 3] = r.h;
      }
      gl.uniform4fv(gl.getUniformLocation(pMosaic, 'uRegions[0]'), flat);
    }

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);

    target2DCtx.clearRect(0, 0, targetW, targetH);
    target2DCtx.drawImage(this._canvas, 0, 0, targetW, targetH);
    return true;
  }

  _destroyWebGL2() {
    const gl = this._gl.ctx;
    if (gl) {
      if (this._gl.srcTex) gl.deleteTexture(this._gl.srcTex);
      if (this._gl.texA) gl.deleteTexture(this._gl.texA);
      if (this._gl.fboA) gl.deleteFramebuffer(this._gl.fboA);
      if (this._gl.texB) gl.deleteTexture(this._gl.texB);
      if (this._gl.fboB) gl.deleteFramebuffer(this._gl.fboB);
      if (this._gl.vao) gl.deleteVertexArray(this._gl.vao);
      if (this._gl.programs) {
        Object.values(this._gl.programs).forEach(p => p && gl.deleteProgram(p));
      }
    }
    this._gl = {
      ctx: null, vao: null, programs: null, srcTex: null, srcW: 0, srcH: 0,
      fboA: null, texA: null, fboB: null, texB: null, fboW: 0, fboH: 0
    };
  }

  /**
   * 選択されているGPUバックエンドでパイプライン全体を実行し、target2DCtx に出力する。
   * 失敗時・非対応時は false を返し、呼び出し元が CPU Worker へフォールバックできるようにする。
   * @param {HTMLCanvasElement} sourceCanvas 入力ベースキャンバス
   * @param {number} sourceVersion ソース更新バージョン番号
   * @param {number} targetW 出力幅
   * @param {number} targetH 出力高さ
   * @param {object} params 調整パラメータ
   * @param {Array<{x:number,y:number,w:number,h:number}>} mosaicRegions モザイク領域配列
   * @param {CanvasRenderingContext2D} target2DCtx 出力先2Dコンテキスト
   * @returns {boolean} GPU描画が成功したか
   */
  render(sourceCanvas, sourceVersion, targetW, targetH, params, mosaicRegions, target2DCtx) {
    if (this.backend === 'webgpu') {
      try {
        return this._renderWebGPU(sourceCanvas, sourceVersion, targetW, targetH, params, mosaicRegions, target2DCtx);
      } catch (_) {
        // WebGPU描画例外時はWebGL2へ降格試行
        this._destroyWebGPU();
        if (this._initWebGL2()) {
          try {
            return this._renderWebGL2(sourceCanvas, sourceVersion, targetW, targetH, params, mosaicRegions, target2DCtx);
          } catch (_) {
            this._destroyWebGL2();
            this.backend = 'none';
            return false;
          }
        }
        this.backend = 'none';
        return false;
      }
    }

    if (this.backend === 'webgl2') {
      try {
        return this._renderWebGL2(sourceCanvas, sourceVersion, targetW, targetH, params, mosaicRegions, target2DCtx);
      } catch (_) {
        this._destroyWebGL2();
        this.backend = 'none';
        return false;
      }
    }

    return false;
  }

  dispose() {
    this._destroyWebGPU();
    this._destroyWebGL2();
    this.backend = 'none';
    this._canvas = null;
  }
}
