export interface OverviewStat {
  label: string;
  value: string;
  detail: string;
}

export interface OverviewData {
  stats: OverviewStat[];
  generated_at: string;
}

export interface DeepHedgeRow {
  underlying: string;
  instrument: "vanilla" | "autocall" | "barrier";
  loss_type: "variance" | "cvar" | "cost";
  learned_std: number;
  baseline_std: number;
  std_reduction_pct: number;
  learned_cvar: number;
  baseline_cvar: number;
  cvar_improvement: number;
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
  asof_date: string;
  residual_bp: number;
}

export interface PnlExplainData {
  day0: string;
  asof_date: string;
  nav: number;
  residual_bp: number;
  residual_alert_threshold_bp: number;
  waterfall: PnlWaterfallStep[];
  by_position: PnlByPosition[];
  residual_series: PnlResidualPoint[];
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

export interface AiAgentData {
  day0: string;
  asof_date: string;
  model: string;
  confidence: string;
  summary: string;
  root_cause_hypothesis: string;
  flagged_positions: AiFlaggedPosition[];
  proposed_changes: AiProposedChange[];
  trace: AiTraceEntry[];
}
