// The encoder and decoder graphs are small. Their weights are the sidecars
// audio_encoder.onnx_data and decoder_q4f16.onnx_data. onnxruntime-web does
// not fetch a sidecar unless the session is created with externalData, and
// path has to be the bare filename stored in the graph. Missing that is what
// closed the mic: session create threw, and the page stopped the recording.
// Creating the real session needs the 0.51 GB encoder weights and a WebGPU
// browser (or a wasm build with those bytes local). This locks the options
// loadLfm3 passes, which the mel and text tests never touch.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { graphSession } from "../lfm3.js";
import { LocalSTT } from "../stt-local.js";

const REV = "df2b3ed2ea10824030b3ff29ee46705754e72b99";
const BASE = "https://huggingface.co/LiquidAI/LFM3-Audio-2B-ONNX/resolve/" + REV + "/onnx/";

test("a missing sidecar is explained as weight files that did not load", () => {
  const seen = "Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: Failed to load external data file \"audio_encoder.onnx_data\", error: Module.MountedFiles is not available.";
  assert.equal(LocalSTT.explain(new Error(seen)), "the model's weight files didn't load");
  assert.equal(LocalSTT.explain("Can't create a session."), "the model's weight files didn't load");
  assert.equal(LocalSTT.explain("no WebGPU adapter"), "this browser has no WebGPU GPU, which LFM3 Audio needs");
});

test("loadLfm3 mounts both sidecars, decoder after encoder", () => {
  const enc = graphSession("audio_encoder", "webgpu");
  const dec = graphSession("decoder_q4f16", "webgpu");
  assert.equal(enc.url, BASE + "audio_encoder.onnx");
  assert.deepEqual(enc.options.externalData, [{ path: "audio_encoder.onnx_data", data: BASE + "audio_encoder.onnx_data" }]);
  assert.equal(dec.url, BASE + "decoder_q4f16.onnx");
  assert.deepEqual(dec.options.externalData, [{ path: "decoder_q4f16.onnx_data", data: BASE + "decoder_q4f16.onnx_data" }]);
  assert.deepEqual(enc.options.executionProviders, ["webgpu"]);
  assert.deepEqual(dec.options.executionProviders, ["webgpu"]);

  const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "../lfm3.js"), "utf8");
  const body = src.slice(src.indexOf("export function loadLfm3"));
  const encAt = body.indexOf('graphSession("audio_encoder"');
  const decAt = body.indexOf('graphSession("decoder_q4f16"');
  assert.ok(encAt > 0 && decAt > encAt, "decoder session is created after the encoder");
  assert.equal(body.includes("Promise.all"), false);
});
