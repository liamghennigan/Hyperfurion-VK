// Byte-level decode, checked against the LFM3 tokenizer on a product phrase.
import { test } from "node:test";
import assert from "node:assert/strict";
import { decodeTokens } from "../lfm3-text.js";

test("product words decode from the tokenizer's pieces", () => {
  // ids 58135, 38424, 300, 518, 51, 13468, 351 and their vocab strings
  const words = [];
  words[58135] = "Hyper";
  words[38424] = "fur";
  words[300] = "ion";
  words[518] = "ĠV";
  words[51] = "K";
  words[13468] = "Ġdict";
  words[351] = "ation";
  words[124900] = "<|im_end|>";
  const special = new Set([124900]);
  const text = decodeTokens([58135, 38424, 300, 518, 51, 13468, 351, 124900], words, special);
  assert.equal(text, "Hyperfurion VK dictation");
});
