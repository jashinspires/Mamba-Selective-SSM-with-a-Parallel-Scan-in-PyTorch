export interface ScanEquivalenceReport {
  length_8_max_abs_err: number;
  length_128_max_abs_err: number;
  length_8_tolerance: number;
  length_128_tolerance: number;
}

export interface LengthEvalCell {
  token_accuracy: number;
  num_sequences: number;
  eval_seed: number;
}

export interface TaskEvalEntry {
  mamba_param_count: number;
  transformer_param_count: number;
  transformer_mlp_width: number;
  param_relative_diff: number;
  mamba_step_count: number;
  transformer_step_count: number;
  models: {
    mamba: {
      step_count: number;
      lengths: Record<string, LengthEvalCell>;
    };
    transformer: {
      step_count: number;
      lengths: Record<string, LengthEvalCell>;
    };
  };
}

export interface TrainEvalReport {
  train_length: number;
  eval_lengths: number[];
  tasks: {
    selective_copy: TaskEvalEntry;
    induction: TaskEvalEntry;
  };
}

export interface InferenceReport {
  lengths: number[];
  device: string;
  mamba_state_bytes: number[];
  transformer_kv_bytes: number[];
  mamba_latency_ms: number[];
  transformer_latency_ms: number[];
  mamba_state_byte_ratio: number;
  transformer_4096_over_256_ratio: number;
}

export interface EmpiricalReportsBundle {
  scanEquivalence: ScanEquivalenceReport;
  trainEval: TrainEvalReport;
  inference: InferenceReport;
  sourceFiles: Record<string, string>;
}
