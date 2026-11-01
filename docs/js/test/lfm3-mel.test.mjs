// The log-mel frontend against a NumPy port of the same Slaney settings.
// A 220 Hz + 660 Hz tone, 1600 samples at 16 kHz, plus a silent window.
import { test } from "node:test";
import assert from "node:assert/strict";
import { logMel } from "../lfm3-mel.js";

const TONE = [
  [0, 0, 0.01865609],
  [1, 0, 0.11622052],
  [64, 0, -1.31632889],
  [127, 0, -2],
  [0, 3, -0.01536277],
  [10, 8, 1.12063722],
  [127, 8, -0.69137877],
];

test("log-mel matches the Slaney extractor on a two-tone clip", () => {
  const n = 1600;
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    samples[i] = Math.sin(2 * Math.PI * 220 * i / 16000) * 0.2
      + 0.05 * Math.sin(2 * Math.PI * 660 * i / 16000);
  }
  const { data, frames } = logMel(samples);
  assert.equal(frames, 9);
  assert.equal(data.length, 128 * 9);
  for (const [m, f, want] of TONE) {
    const got = data[m * frames + f];
    assert.ok(Math.abs(got - want) < 2e-4, m + "," + f + " " + got + " vs " + want);
  }
});

test("a silent window is the log floor, one frame", () => {
  const { data, frames } = logMel(new Float32Array(400));
  assert.equal(frames, 1);
  assert.equal(data[0], -2);
  assert.equal(data[127], -2);
});
