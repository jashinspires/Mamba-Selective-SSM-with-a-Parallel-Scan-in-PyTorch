export interface ScanPair {
  a: number;
  b: number;
}

export interface ScanStepInput {
  t: number;
  x: number;
  delta: number;
  B: number;
  C: number;
}

export interface ScanRoundNode {
  t: number;
  a: number;
  b: number;
  sourceEarlierT: number | null;
}

export interface ScanSimulationResult {
  A: number;
  steps: {
    t: number;
    x: number;
    delta: number;
    B: number;
    C: number;
    aBar: number;
    bBar: number;
    bInput: number;
    hSeq: number;
    ySeq: number;
    hPar: number;
    yPar: number;
    absDiff: number;
  }[];
  rounds: ScanRoundNode[][];
  maxAbsDiff: number;
  associativityMaxErr: number;
}

export function discretizeScalar(delta: number, A: number, B: number): { aBar: number; bBar: number } {
  const deltaA = delta * A;
  const aBar = Math.exp(deltaA);
  // Accurate expm1 formulation matching PyTorch torch.expm1(delta * A) / A * B
  const expm1 = Math.abs(deltaA) < 1e-7
    ? deltaA + 0.5 * deltaA * deltaA + (deltaA * deltaA * deltaA) / 6
    : Math.exp(deltaA) - 1;
  const bBar = (expm1 / A) * B;
  return { aBar, bBar };
}

export function composePairs(later: ScanPair, earlier: ScanPair): ScanPair {
  return {
    a: later.a * earlier.a,
    b: later.a * earlier.b + later.b,
  };
}

export function sequentialScan1D(
  aBars: number[],
  bBars: number[],
  xs: number[],
  Cs: number[],
): { h: number[]; y: number[] } {
  let hCurr = 0;
  const h: number[] = [];
  const y: number[] = [];
  for (let t = 0; t < xs.length; t++) {
    hCurr = aBars[t] * hCurr + bBars[t] * xs[t];
    h.push(hCurr);
    y.push(Cs[t] * hCurr);
  }
  return { h, y };
}

export function parallelAssociativeScan1D(
  aSeq: number[],
  bSeq: number[],
): { aOut: number[]; bOut: number[]; rounds: ScanRoundNode[][] } {
  const n = aSeq.length;
  let curr: ScanPair[] = aSeq.map((a, i) => ({ a, b: bSeq[i] }));
  const rounds: ScanRoundNode[][] = [
    curr.map((p, t) => ({ t, a: p.a, b: p.b, sourceEarlierT: null })),
  ];

  let stride = 1;
  while (stride < n) {
    const next: ScanPair[] = new Array(n);
    const roundNodes: ScanRoundNode[] = new Array(n);
    for (let t = 0; t < n; t++) {
      if (t < stride) {
        next[t] = curr[t];
        roundNodes[t] = { t, a: curr[t].a, b: curr[t].b, sourceEarlierT: null };
      } else {
        const merged = composePairs(curr[t], curr[t - stride]);
        next[t] = merged;
        roundNodes[t] = { t, a: merged.a, b: merged.b, sourceEarlierT: t - stride };
      }
    }
    curr = next;
    rounds.push(roundNodes);
    stride *= 2;
  }

  return {
    aOut: curr.map((p) => p.a),
    bOut: curr.map((p) => p.b),
    rounds,
  };
}

export function checkThreePairAssociativity(p1: ScanPair, p2: ScanPair, p3: ScanPair): number {
  const left = composePairs(p3, composePairs(p2, p1));
  const right = composePairs(composePairs(p3, p2), p1);
  return Math.max(Math.abs(left.a - right.a), Math.abs(left.b - right.b));
}

export function simulateMambaScan(A: number, inputs: ScanStepInput[]): ScanSimulationResult {
  const aBars: number[] = [];
  const bBars: number[] = [];
  const xs: number[] = [];
  const Cs: number[] = [];
  const bInputs: number[] = [];

  for (const step of inputs) {
    const { aBar, bBar } = discretizeScalar(step.delta, A, step.B);
    aBars.push(aBar);
    bBars.push(bBar);
    xs.push(step.x);
    Cs.push(step.C);
    bInputs.push(bBar * step.x);
  }

  const seq = sequentialScan1D(aBars, bBars, xs, Cs);
  const par = parallelAssociativeScan1D(aBars, bInputs);

  let maxAbsDiff = 0;
  const steps = inputs.map((inp, idx) => {
    const hPar = par.bOut[idx];
    const yPar = inp.C * hPar;
    const diff = Math.abs(seq.y[idx] - yPar);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
    return {
      t: inp.t,
      x: inp.x,
      delta: inp.delta,
      B: inp.B,
      C: inp.C,
      aBar: aBars[idx],
      bBar: bBars[idx],
      bInput: bInputs[idx],
      hSeq: seq.h[idx],
      ySeq: seq.y[idx],
      hPar,
      yPar,
      absDiff: diff,
    };
  });

  const assocErr =
    inputs.length >= 3
      ? checkThreePairAssociativity(
          { a: aBars[0], b: bInputs[0] },
          { a: aBars[1], b: bInputs[1] },
          { a: aBars[2], b: bInputs[2] },
        )
      : 0;

  return {
    A,
    steps,
    rounds: par.rounds,
    maxAbsDiff,
    associativityMaxErr: assocErr,
  };
}

export interface ScanPreset {
  id: string;
  label: string;
  description: string;
  A: number;
  inputs: ScanStepInput[];
}

export const SCAN_PRESETS: ScanPreset[] = [
  {
    id: 'one-state',
    label: 'One-State Numeric Check',
    description: 'A = -1, Delta = ln(2), B = 2, C = 1, x = [3, 1]. Exact ZOH yields A-bar = 0.5, B-bar = 1, y = [3.0, 2.5].',
    A: -1,
    inputs: [
      { t: 0, x: 3, delta: Math.LN2, B: 2, C: 1 },
      { t: 1, x: 1, delta: Math.LN2, B: 2, C: 1 },
    ],
  },
  {
    id: 'four-step',
    label: 'Four-Step Associative Check',
    description: 'a = [0.5, 0.5, 0.25, 0.5], b = [1, 1, 1, 1]. Sequential & parallel states equal [1.0, 1.5, 1.375, 1.6875].',
    A: -1,
    inputs: [
      { t: 0, x: 1, delta: Math.LN2, B: 2, C: 1 },
      { t: 1, x: 1, delta: Math.LN2, B: 2, C: 1 },
      { t: 2, x: 1, delta: 2 * Math.LN2, B: 4 / 3, C: 1 },
      { t: 3, x: 1, delta: Math.LN2, B: 2, C: 1 },
    ],
  },
  {
    id: 'selective-gate',
    label: 'Selective Memory & Reset',
    description: 'Small Delta retains stored state across noise (t=1..4), while large Delta at t=5 resets A-bar toward 0.',
    A: -2,
    inputs: [
      { t: 0, x: 4.0, delta: 0.4, B: 1.5, C: 1 },
      { t: 1, x: 0.0, delta: 0.02, B: 1.0, C: 1 },
      { t: 2, x: 0.0, delta: 0.02, B: 1.0, C: 1 },
      { t: 3, x: 0.0, delta: 0.02, B: 1.0, C: 1 },
      { t: 4, x: 0.0, delta: 0.02, B: 1.0, C: 1 },
      { t: 5, x: 2.5, delta: 2.8, B: 2.0, C: 1 },
      { t: 6, x: 0.0, delta: 0.03, B: 1.0, C: 1 },
      { t: 7, x: 0.0, delta: 0.03, B: 1.0, C: 1 },
    ],
  },
];
