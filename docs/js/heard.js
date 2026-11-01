// ═══ HEARD — the in-tab model's usual mishearings, read as meant ═════════
// The page reads a few command-slot mishearings as the words that were meant.
// Measured on the previous in-tab model (Moonshine tiny): a spoken "VK" came
// out as "The K", "Dk" or "Decay", and "spell" as "Bell" or "Fill". LFM3
// Audio was not given a new table. The patterns stay because they only fire
// where a command is the only reading: the wake word at the start of a
// sentence and right before an instruction verb; "spell" only before "that"
// and three or more spelled letters. "Fill that out", "the K key" and
// "decay turns leaves brown" stay as said. dictation.js applies this to the
// in-tab model only.
//
// This lives outside the engine on purpose. The app's engine, and this
// page's port of it (flow.js), never guess at the wake word: what follows
// it is an instruction, never typed, so a false match would swallow
// dictation. A guess tuned to one small model belongs with that model.
const WAKE_HEARD = /(^|[.!?]\s+)(?:the k|[bdv]\.? ?k\.?|decay)[,.]?\s+(?=(?:make|run|command|execute|rewrite|rephrase|reword|translate|turn|fix|change|put|write|shorten|format)\b)/gi;
const SPELL_HEARD = /\b(?:bell|fill|build|built|spill)(?=[,.]?\s+that[,.]?\s+(?:(?:[a-z][.,]?\s+){2,}[a-z](?![\w'])|[a-z](?:-[a-z]){2,}(?![\w'])))/gi;

export function readAsMeant(text, wakeWord = "vk") {
  return String(text || "")
    .replace(WAKE_HEARD, (_, lead) => lead + String(wakeWord).toUpperCase() + ", ")
    .replace(SPELL_HEARD, "spell");
}
