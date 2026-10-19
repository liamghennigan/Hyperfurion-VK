// ═══ AUDIO OUT — mobile-safe playback for fetched speech ══════════════════
// Mobile browsers gate playback behind a user gesture. unlock() runs
// synchronously inside one (the click on "ask" or "hear it"); after that,
// WebAudio can play fetched audio whenever it arrives. HTMLAudio is the
// fallback.
export const AudioOut = (() => {
  let actx = null, analyser = null, abuf = null;
  function unlock() {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!actx && AC) { try { actx = new AC(); } catch {} }
    if (actx && actx.state === "suspended") actx.resume().catch(() => {});
  }
  async function play(blob) {
    if (actx && actx.state === "running") {
      const audio = await actx.decodeAudioData(await blob.arrayBuffer());
      const src = actx.createBufferSource();
      src.buffer = audio;
      // tap the playback through an analyser: level() reports how loud
      // the voice is at any moment
      if (!analyser) {
        analyser = actx.createAnalyser();
        analyser.fftSize = 512;
        abuf = new Uint8Array(analyser.fftSize);
        analyser.connect(actx.destination);
      }
      src.connect(analyser);
      src.start();
      await new Promise((res) => { src.onended = res; });
      return;
    }
    const url = URL.createObjectURL(blob);
    const au = new Audio(url);
    au.onended = () => URL.revokeObjectURL(url);
    await au.play();
  }
  // instantaneous playback loudness 0..1 (0 when nothing is playing)
  function level() {
    if (!analyser) return 0;
    analyser.getByteTimeDomainData(abuf);
    let peak = 0;
    for (let i = 0; i < abuf.length; i++) peak = Math.max(peak, Math.abs(abuf[i] - 128) / 128);
    return Math.min(1, peak * 1.6);
  }
  return { unlock, play, level };
})();

// ═══ DEMO — no hosted relay ════════════════════════════════════════════════
// This page does not contact a Hyperfurion server. `?relay=` used to point
// a local experiment at a relay; with no value, nothing here fetches.
export const Demo = (() => {
  const base = (new URLSearchParams(location.search).get("relay") || "").replace(/\/$/, "");
  const D = { want: false, status: null, base, wsBase: base.replace(/^http/, "ws") };
  D.check = async () => {
    if (!base) {
      D.status = { live: false, reason: "no relay configured" };
      return D.status;
    }
    try {
      const r = await fetch(base + "/v1/demo/status", { signal: AbortSignal.timeout(4000) });
      D.status = await r.json();
    } catch { D.status = { live: false, reason: "relay unreachable" }; }
    return D.status;
  };
  D.armed = () => D.want && !!(D.status && D.status.live);
  D.tts = async (text) => {
    if (!base) throw new Error("no relay configured");
    const r = await fetch(base + "/v1/demo/tts", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.error || "upstream " + r.status);
    }
    return r.blob();
  };
  D.ask = async (question) => {
    if (!base) throw new Error("no relay configured");
    const r = await fetch(base + "/v1/demo/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const e = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(e.error || "upstream " + r.status);
    return String(e.answer || "");
  };
  return D;
})();
