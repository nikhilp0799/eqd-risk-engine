/** One (instrument, objective) combination: mean across training seeds, with
 * the min/max range for each headline metric. */
export interface DeepHedgeRow {
  underlying: string;
  instrument: "vanilla" | "autocall" | "barrier";
  loss_type: "variance" | "cvar" | "cost";
  n_seeds: number;
  epochs_mean: number;
  learned_std: number;
  baseline_std: number;
  learned_turnover: number;
  baseline_turnover: number;
  std_reduction_pct: number;
  std_reduction_pct_min: number;
  std_reduction_pct_max: number;
  cvar_improvement_pct: number;
  cvar_improvement_pct_min: number;
  cvar_improvement_pct_max: number;
  turnover_change_pct: number;
  turnover_change_pct_min: number;
  turnover_change_pct_max: number;
}

export interface DeepHedgeData {
  asof_date: string;
  rows: DeepHedgeRow[];
}

export interface VolSurfacePillar {
  expiry: string;
  T: number;
  model: string;
  n_points: number;
  rmse_vol_points: number;
  max_abs_error_vol_points: number;
  butterfly_violations: number;
  calendar_violated: boolean;
}

export interface VolSmile {
  T: number;
  model: string;
  k_grid: number[];
  fit_iv: number[];
  market_k: number[];
  market_iv: number[];
}

export interface GreeksLadder {
  expiry: string;
  strike: number[];
  delta_spot: number[];
  gamma_spot: number[];
  vega: number[];
  theta: number[];
  price: number[];
}

export interface VolSurfaceData {
  asof_date: string;
  underlying: string;
  pillars: VolSurfacePillar[];
  smiles: Record<string, VolSmile>;
  greeks_ladder: GreeksLadder;
}

export interface PnlWaterfallStep {
  step: "time" | "rates_divs" | "spot" | "vol";
  actual_pnl: number;
  explained_pnl: number;
  residual: number;
}

export interface PnlByPosition {
  position_id: string;
  residual: number;
}

export interface PnlResidualPoint {
  day0: string;
  asof_date: string;
  residual_bp: number;
  noise_band_bp: number;
  status: "alert" | "inconclusive" | "ok";
  top_position_id: string | null;
  top_position_share: number | null;
}

export interface PnlExplainData {
  day0: string;
  asof_date: string;
  nav: number;
  residual_bp: number;
  noise_band_bp: number;
  status: "alert" | "inconclusive" | "ok";
  residual_alert_threshold_bp: number;
  positions: Record<string, string>;
  waterfall: PnlWaterfallStep[];
  by_position: PnlByPosition[];
  residual_series: PnlResidualPoint[];
  improvement: PnlImprovement;
}

export interface PnlImprovement {
  mean_abs_bp_before: number;
  mean_abs_bp_after: number;
  worst_bp_before: number;
  worst_bp_after: number;
  findings: { title: string; body: string }[];
}

export interface AiFlaggedPosition {
  position_id: string;
  reason: string;
}

export interface AiProposedChange {
  parameter: string;
  current_value: string;
  suggested_value: string;
  rationale: string;
}

export interface AiTraceEntry {
  round: number;
  tool_name: string;
  arguments: string;
  result_summary: string;
}

export interface AiFactCheck {
  status: "verified" | "unsupported_numbers" | "mislabelled_numbers" | "unparsed";
  n_numbers: number;
  unsupported: string[];
  mislabelled: string[];
  retried: boolean;
}

export interface AiAgentData {
  day0: string;
  asof_date: string;
  unexplained: number;
  unexplained_bp: number;
  top_position_id: string;
  top_position_share_pct: number;
  model: string;
  confidence: string;
  summary: string;
  root_cause_hypothesis: string;
  factcheck: AiFactCheck | null;
  flagged_positions: AiFlaggedPosition[];
  proposed_changes: AiProposedChange[];
  trace: AiTraceEntry[];
  position_underlyings: Record<string, string>;
}

export type RobustnessScenario = "in_sample" | "vol_up_25" | "vol_down_25" | "jumps" | "history";

/** One (instrument, objective, scenario): mean across seeds with min/max. */
export interface RobustnessRow {
  instrument: DeepHedgeRow["instrument"];
  loss_type: DeepHedgeRow["loss_type"];
  scenario: RobustnessScenario;
  n_seeds: number;
  std_reduction_pct: number;
  std_reduction_pct_min: number;
  std_reduction_pct_max: number;
  cvar_improvement_pct: number;
  cvar_improvement_pct_min: number;
  cvar_improvement_pct_max: number;
}

export interface RobustnessData {
  asof_date: string;
  scenarios: RobustnessScenario[];
  history_independent: Record<string, number>;
  rows: RobustnessRow[];
}

/** Vol surface PCA for one name (`eqdrisk volpca`): how its implied-vol
 * surface moves from one trading day to the next. */
export interface VolPcaName {
  underlying: string;
  n_changes: number;
  n_nodes: number;
  first_change_date: string;
  last_change_date: string;
  /** Share of day-to-day variance explained by patterns 1-3, in %. */
  explained_pct: number[];
  /** Pattern 1 loading by maturity (mean over the included strikes). */
  pc1_by_maturity: { T: number; loading: number }[];
  /** Lowest |cosine| between a rolling window's pattern and the full sample's. */
  lowest_window_match: Record<string, number>;
}

export interface VolPcaData {
  asof_date: string;
  window_days: number;
  names: VolPcaName[];
}

export interface VarPosition {
  id: string;
  label: string;
  /** Root-mean-square 1-day P&L over the scenarios, full revaluation. */
  rms_full: number;
  /** Root-mean-square gap, Greeks shortcut minus full revaluation. */
  rms_gap: number;
  /** Slope of shortcut P&L on full-revaluation P&L: 1 = captures it all. */
  taylor_capture: number;
  /** The replayed day with the largest shortcut error for this position. */
  max_gap: { date: string; full: number; taylor: number };
}

export interface VarProxyName {
  underlying: string;
  label: string;
  index: string;
  n_changes: number;
  n_nodes: number;
  r2_median: number;
  atm_1m_beta: number | null;
  atm_1m_r2: number | null;
}

export interface VarData {
  asof_date: string;
  window_days: number;
  n_scenarios: number;
  first_scenario_date: string;
  var1: { full: number; taylor: number };
  es1: { full: number; taylor: number };
  var1_ex_varswap: number;
  var10: { overlapping: number; sqrt10: number };
  windows: { window: number; var_full: number; var_taylor: number }[];
  /** 10-day ES 97.5% by which factors moved (all / spot / vol / single_name_vol) and
   * by liquidity-horizon bucket (LH10 / LH20 / LH60). */
  es10: Record<string, number>;
  es10_lh_adjusted: number;
  runtime: { full_seconds: number; taylor_seconds: number };
  histogram: { lo: number; hi: number; count: number }[];
  /** Replayed days with gains beyond the histogram's range, left out of it. */
  histogram_hidden: { count: number; above: number };
  worst_days: { date: string; full: number; taylor: number }[];
  positions: VarPosition[];
  series: { date: string; full: number; taylor: number }[];
  proxy: {
    names: VarProxyName[];
    n_days: number;
    true_sd: number;
    rms_error: number;
    varswap_rms_error: number;
    worst_day: { date: string; true: number; proxy: number };
  };
}
