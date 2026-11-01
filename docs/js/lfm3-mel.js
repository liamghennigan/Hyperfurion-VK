// Log-mel frontend for LFM3 Audio. This is the checkpoint's extractor, in the
// page: a periodic Hann window of 400 samples every 160, no centering, 128
// Slaney mel filters from 0 to 8 kHz, base-10 log clamped at -12, then
// (value + 4) / 4. Clips are right-padded so the last window is whole.
// The layout is [128, frames], the shape the ONNX encoder takes as
// [1, 128, frames].

const N_FFT = 400;
const HOP = 160;
const MELS = 128;
const BINS = N_FFT / 2 + 1;

let filters = null;
let window = null;
let twiddle = null;

function linspace(a, b, n) {
  const out = new Float64Array(n);
  if (n === 1) { out[0] = a; return out; }
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = a + step * i;
  return out;
}
function hzToMel(freq) {
  const minLogHz = 1000, minLogMel = 15, logstep = 27 / Math.log(6.4);
  return freq >= minLogHz ? minLogMel + Math.log(freq / minLogHz) * logstep : (3 * freq) / 200;
}
function melToHz(mel) {
  const minLogHz = 1000, minLogMel = 15, logstep = Math.log(6.4) / 27;
  return mel >= minLogMel ? minLogHz * Math.exp(logstep * (mel - minLogMel)) : (200 * mel) / 3;
}
function melFilters() {
  if (filters) return filters;
  const n = MELS;
  const melFreqs = linspace(hzToMel(0), hzToMel(8000), n + 2);
  const filterFreqs = new Float64Array(n + 2);
  for (let i = 0; i < filterFreqs.length; i++) filterFreqs[i] = melToHz(melFreqs[i]);
  const fftFreqs = linspace(0, 8000, BINS);
  const bank = new Float64Array(BINS * n);
  for (let b = 0; b < BINS; b++) {
    for (let m = 0; m < n; m++) {
      const down = -(filterFreqs[m] - fftFreqs[b]) / (filterFreqs[m + 1] - filterFreqs[m]);
      const up = (filterFreqs[m + 2] - fftFreqs[b]) / (filterFreqs[m + 2] - filterFreqs[m + 1]);
      bank[b * n + m] = Math.max(0, Math.min(down, up));
    }
  }
  for (let m = 0; m < n; m++) {
    const enorm = 2 / (filterFreqs[m + 2] - filterFreqs[m]);
    for (let b = 0; b < BINS; b++) bank[b * n + m] *= enorm;
  }
  filters = bank;
  return bank;
}
function hann() {
  if (window) return window;
  window = new Float64Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  return window;
}
function twiddles() {
  if (twiddle) return twiddle;
  const cos = new Float64Array(BINS * N_FFT);
  const sin = new Float64Array(BINS * N_FFT);
  for (let k = 0; k < BINS; k++) {
    const base = (-2 * Math.PI * k) / N_FFT;
    for (let t = 0; t < N_FFT; t++) {
      cos[k * N_FFT + t] = Math.cos(base * t);
      sin[k * N_FFT + t] = Math.sin(base * t);
    }
  }
  twiddle = { cos, sin };
  return twiddle;
}

// samples shorter than one window become one window of zeros on the right;
// longer clips grow to a whole number of hops so the tail is not dropped
export function padLength(numSamples) {
  if (numSamples < N_FFT) return N_FFT;
  const rem = (numSamples - N_FFT) % HOP;
  return rem === 0 ? numSamples : numSamples + (HOP - rem);
}

export function logMel(samples) {
  const src = samples instanceof Float32Array ? samples : Float32Array.from(samples);
  const padded = padLength(src.length);
  const frames = 1 + (padded - N_FFT) / HOP;
  const wave = new Float64Array(padded);
  for (let i = 0; i < src.length; i++) wave[i] = src[i];
  const bank = melFilters();
  const win = hann();
  const { cos, sin } = twiddles();
  const spec = new Float64Array(MELS * frames);
  const frame = new Float64Array(N_FFT);
  const power = new Float64Array(BINS);
  for (let f = 0; f < frames; f++) {
    const off = f * HOP;
    for (let t = 0; t < N_FFT; t++) frame[t] = wave[off + t] * win[t];
    for (let k = 0; k < BINS; k++) {
      let re = 0, im = 0;
      const base = k * N_FFT;
      for (let t = 0; t < N_FFT; t++) {
        const w = frame[t];
        re += w * cos[base + t];
        im += w * sin[base + t];
      }
      power[k] = re * re + im * im;
    }
    for (let m = 0; m < MELS; m++) {
      let acc = 0;
      for (let k = 0; k < BINS; k++) acc += bank[k * MELS + m] * power[k];
      let v = acc > 0 ? Math.log10(acc) : -Infinity;
      if (v < -12) v = -12;
      spec[m * frames + f] = (v + 4) / 4;
    }
  }
  return { data: Float32Array.from(spec), frames };
}
