export interface SyntheticSequence {
  inputIds: number[];
  targetIds: number[];
  lossMask: number[];
  contentPositions?: number[];
  bigramStarts?: number[];
  queryKey?: number;
}

// Deterministic Mulberry32 PRNG for reproducible sequence previews in the browser.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateSelectiveCopyPreview(
  length: number,
  seed: number,
  nContent = 16,
): SyntheticSequence {
  const rng = mulberry32(seed);
  const actualContent = Math.min(nContent, Math.floor(length / 2));
  const prefixLen = length - actualContent;

  const indices = Array.from({ length: prefixLen }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const chosenPositions = indices.slice(0, actualContent).sort((a, b) => a - b);

  const inputIds = new Array<number>(length).fill(0);
  const targetIds = new Array<number>(length).fill(0);
  const lossMask = new Array<number>(length).fill(0);

  const tokens: number[] = [];
  for (let i = 0; i < actualContent; i++) {
    const tok = 1 + Math.floor(rng() * 16);
    tokens.push(tok);
    inputIds[chosenPositions[i]] = tok;
  }

  for (let i = 0; i < actualContent; i++) {
    inputIds[prefixLen + i] = 17;
    targetIds[prefixLen + i] = tokens[i];
    lossMask[prefixLen + i] = 1;
  }

  return {
    inputIds,
    targetIds,
    lossMask,
    contentPositions: chosenPositions,
  };
}

export function generateInductionPreview(length: number, seed: number): SyntheticSequence {
  const rng = mulberry32(seed);
  const vocab = Array.from({ length: 16 }, (_, i) => i);
  for (let i = vocab.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [vocab[i], vocab[j]] = [vocab[j], vocab[i]];
  }
  const keys = vocab.slice(0, 4);
  const nonKeys = vocab.slice(4);
  const values = Array.from({ length: 4 }, () => nonKeys[Math.floor(rng() * nonKeys.length)]);

  const inputIds = Array.from({ length }, () => nonKeys[Math.floor(rng() * nonKeys.length)]);
  const targetIds = new Array<number>(length).fill(0);
  const lossMask = new Array<number>(length).fill(0);

  const halfLen = Math.floor(length / 2);
  const candidates: number[] = [];
  for (let p = 0; p < Math.max(8, halfLen - 1); p += 2) {
    candidates.push(p);
  }
  for (let i = candidates.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
  }
  const starts = candidates.slice(0, 4).sort((a, b) => a - b);

  for (let i = 0; i < 4; i++) {
    inputIds[starts[i]] = keys[i];
    inputIds[starts[i] + 1] = values[i];
  }

  const qIdx = Math.floor(rng() * 4);
  inputIds[length - 1] = keys[qIdx];
  targetIds[length - 1] = values[qIdx];
  lossMask[length - 1] = 1;

  return {
    inputIds,
    targetIds,
    lossMask,
    bigramStarts: starts,
    queryKey: keys[qIdx],
  };
}
