import test from 'node:test';
import assert from 'node:assert/strict';
import {
  checkThreePairAssociativity,
  discretizeScalar,
  parallelAssociativeScan1D,
  SCAN_PRESETS,
  sequentialScan1D,
  simulateMambaScan,
} from './mambaSim';
import { generateInductionPreview, generateSelectiveCopyPreview } from './dataGenerators';

test('one-state ZOH discretization and recurrence match 2.5 within 1e-5', () => {
  const { aBar, bBar } = discretizeScalar(Math.LN2, -1, 2);
  assert.ok(Math.abs(aBar - 0.5) < 1e-5);
  assert.ok(Math.abs(bBar - 1.0) < 1e-5);

  const sim = simulateMambaScan(-1, SCAN_PRESETS[0].inputs);
  assert.ok(Math.abs(sim.steps[0].ySeq - 3.0) < 1e-5);
  assert.ok(Math.abs(sim.steps[1].ySeq - 2.5) < 1e-5);
  assert.ok(sim.maxAbsDiff < 1e-6);
});

test('four-step parallel associative scan matches [1, 1.5, 1.375, 1.6875]', () => {
  const a = [0.5, 0.5, 0.25, 0.5];
  const b = [1, 1, 1, 1];
  const seq = sequentialScan1D(a, b, [1, 1, 1, 1], [1, 1, 1, 1]);
  const par = parallelAssociativeScan1D(a, b);
  const expected = [1.0, 1.5, 1.375, 1.6875];
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(seq.h[i] - expected[i]) < 1e-6);
    assert.ok(Math.abs(par.bOut[i] - expected[i]) < 1e-6);
  }
});

test('compose is associative on arbitrary pairs', () => {
  const err = checkThreePairAssociativity(
    { a: 0.41, b: -1.3 },
    { a: 0.87, b: 0.62 },
    { a: 0.19, b: 2.15 },
  );
  assert.ok(err < 1e-12);
});

test('browser sequence preview generators preserve selective copy and induction contracts', () => {
  const sc = generateSelectiveCopyPreview(64, 20261114, 8);
  assert.equal(sc.inputIds.length, 64);
  assert.equal(sc.lossMask.slice(0, 56).reduce((a, b) => a + b, 0), 0);
  assert.equal(sc.lossMask.slice(56).reduce((a, b) => a + b, 0), 8);

  const ind = generateInductionPreview(64, 20261114);
  assert.equal(ind.inputIds.length, 64);
  assert.equal(ind.lossMask.slice(0, 63).reduce((a, b) => a + b, 0), 0);
  assert.equal(ind.lossMask[63], 1);
  const qKey = ind.inputIds[63];
  const expectedVal = ind.targetIds[63];
  const matched = ind.bigramStarts!.some((s) => ind.inputIds[s] === qKey && ind.inputIds[s + 1] === expectedVal);
  assert.equal(matched, true);
});
