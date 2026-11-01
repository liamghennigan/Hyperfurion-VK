// Byte-level BPE decode for the LFM3 tokenizer (GPT-2 bytes-to-unicode).
// Encoding the prompt is not done here: the ASR prompt is a fixed token-id
// list in lfm3.js, checked against this tokenizer. Decoding generated ids
// needs the vocab, which the page downloads with the model.

export function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 0xA1; i <= 0xAC; i++) bs.push(i);
  for (let i = 0xAE; i <= 0xFF; i++) bs.push(i);
  const cs = bs.slice();
  const have = new Set(bs);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!have.has(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const u2b = new Uint16Array(324);
  u2b.fill(0xffff);
  for (let i = 0; i < cs.length; i++) u2b[cs[i]] = bs[i];
  return u2b;
}

// words: id -> token string. special: ids to drop (the tokenizer's specials).
export function decodeTokens(ids, words, special, u2b = bytesToUnicode()) {
  let chars = "";
  for (const id of ids) {
    if (special && special.has(id)) continue;
    const tok = words[id];
    if (tok) chars += tok;
  }
  const bytes = new Uint8Array(chars.length);
  for (let i = 0; i < chars.length; i++) {
    const b = u2b[chars.charCodeAt(i)];
    if (b === 0xffff) throw new Error("tokenizer byte map has no entry for " + chars.charCodeAt(i));
    bytes[i] = b;
  }
  return new TextDecoder().decode(bytes);
}
