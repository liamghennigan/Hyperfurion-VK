// ═══ LFM3 AUDIO — speech to text, in this tab ════════════════════════════
// Liquid AI's LFM3-Audio-2B, the ONNX export, on WebGPU. The browser build
// Liquid documents is onnxruntime-web's WebGPU entry: decoder_q4f16 with the
// float32 audio encoder. The speech-output graphs (depth decoder, codec) are
// not fetched — dictation only wants text. There is no in-tab CPU copy: the
// same card says the default onnxruntime-web build cannot run these graphs,
// and the CPU weights are a different, larger download.
//
// Pinned revision df2b3ed2ea10824030b3ff29ee46705754e72b99 (2026-10-28), so
// the byte totals stay true. Uncompressed sizes the progress bar divides by
// (the runtime's JS module is a script import, so it is not in the sum):
//   ort-wasm-simd-threaded.jsep.wasm      29,520,971
//   ort-wasm-simd-threaded.jsep.mjs           47,250
//   tokenizer.json                        17,908,634
//   onnx/audio_encoder.onnx                  162,507
//   onnx/audio_encoder.onnx_data         510,351,360
//   onnx/decoder_q4f16.onnx                  274,325
//   onnx/decoder_q4f16.onnx_data       1,413,388,288
//                                        -------------
//                                        1,971,653,335  (about 1.97 GB)
//
// One utterance is one clip. The encoder turns log-mel frames into one
// vector per eight frames. Those vectors replace the <|audio|> tokens
// (id 125031) in a fixed ASR prompt. The decoder then writes text, greedily,
// and is not allowed to switch into speech (<|audio_start|>, <|mixed_start|>,
// or a placeholder). Prefill is cut into chunks of at most 1024 tokens, which
// is the limit the ONNX card states for WebGPU. Audio never leaves the tab.
import { logMel } from "./lfm3-mel.js";
import { bytesToUnicode, decodeTokens } from "./lfm3-text.js";

const REV = "df2b3ed2ea10824030b3ff29ee46705754e72b99";
const BASE = "https://huggingface.co/LiquidAI/LFM3-Audio-2B-ONNX/resolve/" + REV;
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
const WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

export const LFM3_BYTES = 1971653335;
export const LFM3_LABEL = "LFM3 Audio";

// "<|startoftext|><|im_start|>system\nPerform ASR in English.<|im_end|>\n<|im_start|>user\n"
const PREFIX = [124894, 124899, 23630, 207, 116068, 11365, 58, 296, 3478, 22, 124900, 207, 124899, 5922, 207];
// "<|im_end|>\n<|im_start|>assistant\n"
const SUFFIX = [124900, 207, 124899, 63514, 207];
const AUDIO = 125031;
const STOP = new Set([124900, 125014]); // <|im_end|>, <|text_end|>
// placeholders, plus the two tokens that would switch the model into speech
const SUPPRESS = new Set([125031, 125032, 125012, 125015]);
const CHUNK = 1024;

const CONV = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
const ATTN = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22];

let ort = null, encoder = null, decoder = null, words = null, special = null, u2b = null, ready = null;

function f16(len) {
  const C = globalThis.Float16Array || Uint16Array;
  return new C(len);
}
function drop(t) { try { t && t.dispose && t.dispose(); } catch {} }
function i64(nums, dims) {
  const data = new BigInt64Array(nums.length);
  for (let i = 0; i < nums.length; i++) data[i] = BigInt(nums[i]);
  return new ort.Tensor("int64", data, dims);
}
function ones(n) {
  const data = new BigInt64Array(n);
  data.fill(1n);
  return new ort.Tensor("int64", data, [1, n]);
}
function positions(start, len) {
  const data = new BigInt64Array(len);
  for (let i = 0; i < len; i++) data[i] = BigInt(start + i);
  return new ort.Tensor("int64", data, [1, len]);
}
function emptyCodes() {
  return new ort.Tensor("int64", new BigInt64Array(0), [0, 12]);
}
function emptyFeats() {
  return new ort.Tensor("float32", new Float32Array(0), [0, 2048]);
}
function emptyPast() {
  const past = {};
  for (const i of CONV) past["past_conv." + i] = new ort.Tensor("float16", f16(2048 * 7), [1, 2048, 7]);
  for (const i of ATTN) {
    past["past_key_values." + i + ".key"] = new ort.Tensor("float16", f16(0), [1, 4, 0, 128]);
    past["past_key_values." + i + ".value"] = new ort.Tensor("float16", f16(0), [1, 4, 0, 128]);
  }
  return past;
}
function sliceFeats(data, hidden, start, count) {
  const out = new Float32Array(count * hidden);
  if (count) out.set(data.subarray(start * hidden, (start + count) * hidden));
  return new ort.Tensor("float32", out, [count, hidden]);
}
async function cpuData(tensor) {
  if (tensor.location === "cpu" || tensor.location === "cpu-pinned") return tensor.data;
  return tensor.getData(true);
}
function argmax(data) {
  let best = 0, bestV = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (SUPPRESS.has(i)) continue;
    const v = data[i];
    if (v > bestV) { bestV = v; best = i; }
  }
  return best;
}
function takePast(out) {
  const past = {};
  for (const i of CONV) {
    const t = out["present_conv." + i];
    if (!t) throw new Error("decoder did not return present_conv." + i);
    past["past_conv." + i] = t;
  }
  for (const i of ATTN) {
    const k = out["present." + i + ".key"], v = out["present." + i + ".value"];
    if (!k || !v) throw new Error("decoder did not return present." + i);
    past["past_key_values." + i + ".key"] = k;
    past["past_key_values." + i + ".value"] = v;
  }
  return past;
}
async function step(ids, past, pastLen, feats) {
  const seq = ids.length;
  const feeds = {
    input_ids: i64(ids, [1, seq]),
    attention_mask: ones(pastLen + seq),
    position_ids: positions(pastLen, seq),
    audio_features: feats,
    audio_out_codes: emptyCodes(),
    logits_to_keep: new ort.Tensor("int64", BigInt64Array.of(1n), [1]),
  };
  for (const name in past) feeds[name] = past[name];
  const out = await decoder.run(feeds);
  try {
    const logits = await cpuData(out.logits);
    const id = argmax(logits);
    const pastOut = takePast(out);
    const kept = {};
    for (const name in pastOut) {
      const t = pastOut[name];
      const data = await cpuData(t);
      const copy = data.slice(0);
      kept[name] = new ort.Tensor(t.type, copy, Array.from(t.dims));
    }
    return { id, past: kept };
  } finally {
    for (const name in out) drop(out[name]);
    for (const name in feeds) drop(feeds[name]);
  }
}

async function loadVocab() {
  const res = await fetch(BASE + "/tokenizer.json");
  if (!res.ok) throw new Error("tokenizer download failed (" + res.status + ")");
  const raw = await res.json();
  const table = new Array(raw.model.vocab ? 125033 : 0);
  for (const tok in raw.model.vocab) table[raw.model.vocab[tok]] = tok;
  special = new Set();
  for (const a of raw.added_tokens || []) {
    table[a.id] = a.content;
    if (a.special) special.add(a.id);
  }
  words = table;
  u2b = bytesToUnicode();
}

export function loadLfm3() {
  if (ready) return ready;
  ready = (async () => {
    ort = await import(/* webpackIgnore: true */ ORT);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = WASM;
    const session = { executionProviders: ["webgpu"] };
    await Promise.all([
      loadVocab(),
      ort.InferenceSession.create(BASE + "/onnx/audio_encoder.onnx", session).then((s) => { encoder = s; }),
      ort.InferenceSession.create(BASE + "/onnx/decoder_q4f16.onnx", session).then((s) => { decoder = s; }),
    ]);
    // a short silence proves the GPU can run both graphs before we call it ready
    await transcribe(new Float32Array(8000), { maxNew: 4 });
  })().catch((e) => { ready = null; encoder = null; decoder = null; throw e; });
  return ready;
}

export async function transcribe(samples, opts = {}) {
  if (!decoder) throw new Error("LFM3 Audio is not loaded");
  const { data, frames } = logMel(samples);
  const features = new ort.Tensor("float32", data, [1, 128, frames]);
  let encoded;
  try { encoded = await encoder.run({ input_features: features }); }
  finally { drop(features); }
  const audio = encoded.audio_features;
  if (!audio || audio.dims.length !== 2 || audio.dims[1] !== 2048) {
    drop(audio);
    throw new Error("unexpected audio features");
  }
  const tokens = audio.dims[0];
  const raw = (audio.location === "cpu" || audio.location === "cpu-pinned") ? audio.data : await audio.getData(true);
  const feat = raw instanceof Float32Array ? new Float32Array(raw) : Float32Array.from(raw);
  drop(audio);
  const ids = PREFIX.concat(Array(tokens).fill(AUDIO), SUFFIX);
  let past = null, pastLen = 0, next = 0, used = 0;
  for (let at = 0; at < ids.length; at += CHUNK) {
    const chunk = ids.slice(at, Math.min(ids.length, at + CHUNK));
    let nAudio = 0;
    for (const id of chunk) if (id === AUDIO) nAudio++;
    const piece = sliceFeats(feat, 2048, used, nAudio);
    used += nAudio;
    const held = past;
    const out = await step(chunk, held || emptyPast(), pastLen, piece);
    if (held) for (const name in held) drop(held[name]);
    past = out.past;
    pastLen += chunk.length;
    next = out.id;
  }
  const cap = opts.maxNew == null ? Math.min(256, Math.max(32, tokens * 2)) : opts.maxNew;
  const said = [];
  for (let n = 0; n < cap && !STOP.has(next); n++) {
    said.push(next);
    if (n % 4 === 3) await new Promise((r) => setTimeout(r, 0));
    const out = await step([next], past, pastLen, emptyFeats());
    for (const name in past) drop(past[name]);
    past = out.past;
    pastLen += 1;
    next = out.id;
  }
  if (past) for (const name in past) drop(past[name]);
  return decodeTokens(said, words, special, u2b).trim();
}
