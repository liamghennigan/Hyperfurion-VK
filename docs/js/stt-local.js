// ═══ LOCAL STT — an open-source speech model, in this tab ════════════════
// LFM3 Audio (LFM3-Audio-2B, Liquid AI) through ONNX Runtime Web on WebGPU.
// See lfm3.js for the graphs, the pinned revision and the byte total. The
// device is picked before anything heavy is fetched: a real WebGPU adapter,
// or a failure that downloads nothing. There is no CPU copy of this model
// in the page. If the GPU fails after the download, the session stops and
// the page offers the browser's own speech service.
//
// Nothing is fetched until you tap the mic and the browser lets the page
// use it. Then the runtime comes from jsDelivr and the tokenizer and weights
// from Hugging Face (whose downloads redirect to its CDN). Files are kept in
// this browser's Cache Storage ("lfm3-audio-cache"), so a later visit reads
// them from disk. Audio never leaves the tab.
//
// Capture starts as soon as the mic is granted, before the model is ready:
// what you say while it loads is held in memory and recognized once it is.
// Each utterance, closed by ~1.2 s of quiet, is trimmed to its voiced audio
// (a quarter second before the first voiced frame, 0.3 s after the last) so
// a quiet lead-in is not what the model transcribes. While an utterance is
// open it is re-recognized, never more often than a pass takes.
import { bus } from "./bus.js";
import { LFM3_BYTES, LFM3_LABEL, loadLfm3, transcribe } from "./lfm3.js";

export const LocalSTT = (() => {
  const BYTES = { webgpu: LFM3_BYTES };
  // pct: share of the download that has arrived, while files are arriving
  // (null once they are in, or when everything came from the cache)
  const S = { state: "idle", device: null, error: "", label: LFM3_LABEL, pct: null, fetched: 0,
              stats: { passes: 0, lastMs: 0, audioSec: 0 } };
  let pipe = null, loadP = null, copies = 0, fetched = 0, inflight = 0;
  const listeners = new Set();

  S.supported = () => !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia &&
    (window.AudioContext || window.webkitAudioContext) && typeof WebAssembly === "object");
  S.gpu = () => !!navigator.gpu;

  // ── what went wrong, in words a visitor can use ─────────────────────────
  S.explain = (e) => {
    const m = String((e && (e.message || e.name)) || e || "");
    if (/out of memory|allocation failed|memory/i.test(m)) return "this device ran out of memory loading it";
    if (/fetch|network|import|load failed|could not locate|status|cors|security polic|40\d|50\d/i.test(m))
      return "it couldn't be downloaded (you may be offline, or something is blocking cdn.jsdelivr.net or huggingface.co)";
    if (/no WebGPU adapter/i.test(m)) return "this browser has no WebGPU GPU, which LFM3 Audio needs";
    if (/webgpu|gpu|adapter/i.test(m)) return "this browser's GPU support failed while running it";
    if (/wasm|webassembly|backend/i.test(m)) return "this browser couldn't start the WebAssembly runtime it needs";
    return "this browser couldn't run it";
  };

  let lastByte = 0, settleT = 0;
  function report() {
    const total = BYTES.webgpu;
    S.fetched = fetched;
    // a moment between two files is still downloading; after that it isn't
    const downloading = inflight > 0 || performance.now() - lastByte < 500;
    if (!inflight && downloading && !settleT) settleT = setTimeout(() => { settleT = 0; report(); }, 520);
    S.pct = downloading && fetched > 0 && S.state === "loading" ? Math.min(99, Math.floor((100 * fetched) / total)) : null;
    for (const f of listeners) { try { f(S); } catch {} }
    bus.emit("stt:progress", { state: S.state, device: S.device, pct: S.pct });
  }
  // every file the runtime fetches passes through here, so the progress
  // shown is bytes that really arrived (a Cache Storage hit never comes
  // through: nothing is downloaded for it). The host that finally answered
  // is noted too:
  // Hugging Face redirects its downloads to its CDN, and the browser's
  // own request list names only the address asked for.
  const hosts = new Set();
  const MODEL_HOST = (h) => h === "cdn.jsdelivr.net" || /(^|\.)(huggingface\.co|hf\.co)$/.test(h);
  function watchFetch(id) {
    const orig = globalThis.fetch.bind(globalThis);
    let box = null;
    const cache = () => {
      if (typeof caches === "undefined") return Promise.resolve(null);
      return box || (box = caches.open("lfm3-audio-cache").catch(() => null));
    };
    globalThis.fetch = async (input, init) => {
      let url = "";
      try { url = typeof input === "string" ? new URL(input, location.href).href : input.url; } catch {}
      let model = false;
      try { model = !!(url && MODEL_HOST(new URL(url).hostname)); } catch {}
      if (!model) return orig(input, init);
      try {
        const store = await cache();
        const hit = store && await store.match(url);
        if (hit) return hit;
      } catch {}
      const res = await orig(input, init);
      try {
        const h = new URL(res.url || url).hostname;
        if (h && !hosts.has(h)) { hosts.add(h); bus.emit("stt:host", { host: h, redirected: !!res.redirected }); }
      } catch {}
      if (id !== copies || !res.body || res.status !== 200) return res;
      let last = -1, open = true;
      const done = () => { if (open && id === copies) { open = false; inflight--; report(); } };
      inflight++;
      const [branch, saved] = res.body.tee();
      const counter = new TransformStream({
        transform(chunk, ctl) {
          if (id === copies) {
            fetched += chunk.byteLength; lastByte = performance.now();
            const pct = Math.floor((100 * fetched) / BYTES.webgpu);
            if (pct !== last) { last = pct; report(); }
          }
          ctl.enqueue(chunk);
        },
        flush: done,
        cancel: done,
      });
      const out = new Response(branch.pipeThrough(counter), { status: res.status, statusText: res.statusText, headers: res.headers });
      try { Object.defineProperty(out, "url", { value: res.url }); } catch {}
      const drain = async (stream) => {
        try {
          const reader = stream.getReader();
          for (;;) { const { done } = await reader.read(); if (done) break; }
        } catch {}
      };
      cache().then((store) => store
        ? store.put(url, new Response(saved, { status: res.status, headers: res.headers })).catch(() => drain(saved))
        : drain(saved)).catch(() => drain(saved));
      return out;
    };
    return () => { globalThis.fetch = orig; };
  }

  // WebGPU only with a real adapter: Chrome exposes navigator.gpu even
  // where WebGPU is off (by default on Linux) and then has no adapter
  async function pickDevice() {
    if (!navigator.gpu) return null;
    try {
      const adapter = await Promise.race([navigator.gpu.requestAdapter(), new Promise((r) => setTimeout(() => r(null), 3000))]);
      return adapter ? "webgpu" : null;
    } catch { return null; }
  }
  async function open() {
    const id = ++copies;
    fetched = 0; inflight = 0; S.device = "webgpu"; report();
    const restore = watchFetch(id);
    try {
      await loadLfm3();
      return { textOf: transcribe };
    } finally { restore(); }
  }
  function load(onProgress) {
    if (pipe) return Promise.resolve(pipe);
    if (onProgress) listeners.add(onProgress);
    if (loadP) return loadP;
    S.state = "loading"; S.error = ""; fetched = 0; inflight = 0;
    report();
    bus.emit("relay:request", {});            // the footer counts what you asked for
    loadP = (async () => {
      const device = await pickDevice();
      if (!device) throw new Error("no WebGPU adapter");
      pipe = await open();
      S.state = "ready";
      report();
      return pipe;
    })().catch((e) => {
      S.state = "failed"; S.error = e && e.message ? e.message : String(e);
      pipe = null; loadP = null;
      report();
      throw e;
    }).finally(() => { listeners.clear(); bus.emit("relay:request", {}); });
    return loadP;
  }
  async function recognize(audio) {
    const text = await pipe.textOf(audio);
    return { text };
  }

  // ── capture: the mic at 16 kHz, off the main thread where possible ─────
  // An AudioWorklet keeps capturing while a recognition pass holds the main
  // thread; its chunks queue up and arrive in order. Browsers that can't
  // feed a 16 kHz context from a 48 kHz mic (Firefox) capture at the
  // device rate and are averaged down here.
  const RATE = 16000;
  const WORKLET = `class VKTap extends AudioWorkletProcessor {
    constructor() { super(); this.b = new Float32Array(2048); this.n = 0; this.on = true;
      this.port.onmessage = () => { this.port.postMessage(this.b.slice(0, this.n)); this.n = 0; this.on = false; this.port.postMessage("done"); }; }
    process(inputs) {
      if (!this.on) return false;
      const x = inputs[0] && inputs[0][0];
      if (x) for (let i = 0; i < x.length; i++) {
        this.b[this.n++] = x[i];
        if (this.n === this.b.length) { this.port.postMessage(this.b, [this.b.buffer]); this.b = new Float32Array(2048); this.n = 0; }
      }
      return true;
    }
  }
  registerProcessor("vk-tap", VKTap);`;
  function resampler(ratio, emit) {
    let rest = new Float32Array(0), t = 0;
    return (x) => {
      const all = new Float32Array(rest.length + x.length);
      all.set(rest); all.set(x, rest.length);
      const out = [];
      while (t + ratio <= all.length) {
        const a = Math.floor(t), b = Math.max(a + 1, Math.floor(t + ratio));
        let s = 0;
        for (let i = a; i < b; i++) s += all[i];
        out.push(s / (b - a));
        t += ratio;
      }
      const keep = Math.floor(t);
      rest = all.slice(keep); t -= keep;
      if (out.length) emit(Float32Array.from(out));
    };
  }
  async function capture(stream, onChunk) {
    const AC = window.AudioContext || window.webkitAudioContext;
    let ctx = new AC({ sampleRate: RATE }), src;
    try { src = ctx.createMediaStreamSource(stream); }
    catch {
      try { ctx.close(); } catch {}
      ctx = new AC(); src = ctx.createMediaStreamSource(stream);
    }
    if (ctx.state === "suspended") await ctx.resume().catch(() => {});
    const feed = ctx.sampleRate === RATE ? onChunk : resampler(ctx.sampleRate / RATE, onChunk);
    const sink = ctx.createGain(); sink.gain.value = 0;   // the node needs a destination, not an echo
    let node = null, flushed = null;
    if (ctx.audioWorklet && window.AudioWorkletNode) {
      try {
        const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
        try { await ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
        node = new AudioWorkletNode(ctx, "vk-tap", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
        node.port.onmessage = (e) => {
          if (e.data === "done") { if (flushed) flushed(); return; }
          if (e.data && e.data.length) feed(e.data);
        };
      } catch { node = null; }
    }
    S.capture = (node ? "AudioWorklet" : "ScriptProcessor") + " at " + ctx.sampleRate + " Hz";
    if (!node) {
      node = ctx.createScriptProcessor(2048, 1, 1);
      node.onaudioprocess = (e) => feed(new Float32Array(e.inputBuffer.getChannelData(0)));
    }
    src.connect(node); node.connect(sink); sink.connect(ctx.destination);
    return {
      // every chunk captured so far has been delivered when this resolves
      flush() {
        if (!node.port) return Promise.resolve();
        return new Promise((res) => { flushed = res; node.port.postMessage("flush"); setTimeout(res, 400); });
      },
      close() {
        try { src.disconnect(); node.disconnect(); sink.disconnect(); } catch {}
        if (node.port) node.port.onmessage = null; else node.onaudioprocess = null;
        ctx.close().catch(() => {});
      },
    };
  }

  // ── a session: mic stream in, transcripts out ──────────────────────────
  // onInterim(text): the open utterance so far; onFinal(text): an utterance
  // closed by ~1.2 s of quiet (or by stop()); onError(err): the model could
  // not load or run — the session has stopped and dropped its audio.
  // Resolves once capture is running, whether or not the model is ready.
  async function start(stream, { onInterim, onFinal, onError }) {
    const PRE = RATE * 0.25, POST = RATE * 0.3, GAP = RATE * 1.2, LONG = RATE * 28;
    const MIN = RATE * 0.35, VOICED = RATE * 0.2, EVERY = 500, FLOOR = 0.008;
    let chunks = [], len = 0, first = -1, last = -1, voiced = 0;   // the open utterance
    const closed = [];                          // trimmed utterances waiting for their final pass
    let noise = 0.004, busy = false, ended = false, failed = false, lastRun = 0, previewed = -1, lastText = "";

    function slice(from, to) {
      const out = new Float32Array(Math.max(0, to - from));
      let at = 0;
      for (const c of chunks) {
        const a = Math.max(from, at), b = Math.min(to, at + c.length);
        if (b > a) out.set(c.subarray(a - at, b - at), a - from);
        at += c.length;
        if (at >= to) break;
      }
      return out;
    }
    const voicedEnd = () => Math.min(len, last + POST);
    const voicedAudio = () => slice(Math.max(0, first - PRE), voicedEnd());
    function close() {
      if (voiced >= VOICED) closed.push(voicedAudio());   // a lone click is not an utterance
      const tail = slice(Math.max(0, len - PRE), len);     // quiet: the next utterance's lead-in
      chunks = tail.length ? [tail] : []; len = tail.length;
      first = last = -1; voiced = 0; previewed = -1;
    }
    function onChunk(f) {
      if (ended || failed) return;
      let sum = 0;
      for (let i = 0; i < f.length; i++) sum += f[i] * f[i];
      const rms = Math.sqrt(sum / f.length);
      chunks.push(f);
      if (rms > Math.max(FLOOR, noise * 3)) {     // a pause is a pause in *this* room
        if (first < 0) first = len;
        last = len + f.length; voiced += f.length;
      } else noise += (rms - noise) * 0.05;       // only quiet frames teach the floor
      len += f.length;
      if (first < 0) {                            // nothing said yet: keep only the lead-in
        while (chunks.length > 1 && len - chunks[0].length >= PRE) { len -= chunks[0].length; chunks.shift(); }
      } else if (len - last > GAP || len > LONG) close();
      pump();
    }
    async function pass(audio, final) {
      let text = "";
      if (audio.length >= MIN) {
        const t0 = performance.now();
        const r = await recognize(audio);
        S.stats.passes++; S.stats.lastMs = performance.now() - t0; S.stats.audioSec = audio.length / RATE;
        // honest about slow hardware: a pass slower than the audio it covers
        if (!S.slow && S.stats.audioSec >= 1 && S.stats.lastMs > Math.max(1500, S.stats.audioSec * 1000)) {
          S.slow = true; bus.emit("stt:slow", { ms: S.stats.lastMs });
        }
        text = (r && r.text ? r.text : "").trim();
      }
      lastRun = performance.now();
      if (final) {
        const t = text || lastText;               // an empty final keeps what the last pass heard
        lastText = "";
        if (t) onFinal(t);
      } else if (text && text !== lastText) {     // an empty pass never erases words on screen
        lastText = text; onInterim(text);
      }
    }
    async function pump() {
      if (busy || failed || !pipe) return;
      busy = true;
      try {
        for (;;) {
          if (closed.length) { await pass(closed.shift(), true); continue; }
          const key = first >= 0 ? voicedEnd() : -1;
          // preview cadence adapts to the device: never more often than a pass takes
          if (!ended && voiced >= VOICED && key !== previewed &&
              performance.now() - lastRun > Math.max(EVERY, S.stats.lastMs * 1.2)) {
            previewed = key;
            await pass(voicedAudio(), false);
            continue;
          }
          break;
        }
      } catch (e) { fail(e); }
      busy = false;
    }
    function fail(e) {
      if (failed) return;
      failed = true; chunks = []; len = 0; closed.length = 0;
      S.error = e && e.message ? e.message : String(e);
      if (onError) onError(e);
    }

    const cap = await capture(stream, onChunk);
    load().then(() => pump(), fail);
    return {
      // seconds of speech captured but not yet recognized
      pending: () => (closed.reduce((s, a) => s + a.length, 0) + (first >= 0 ? voicedEnd() - first : 0)) / RATE,
      async stop() {
        if (ended) return;
        await cap.flush();
        cap.close();
        ended = true;
        if (failed) return;
        if (first >= 0) close();                  // the open utterance closes here
        if (!pipe) { try { await load(); } catch { return; } }   // what you said while it loaded still counts
        while (!failed && (busy || closed.length)) {
          if (busy || !pipe) await new Promise((r) => setTimeout(r, 30)); else await pump();
        }
      },
    };
  }

  S.load = load; S.start = start;
  return S;
})();
