// Every headline figure on the site is derived here from the exported JSON, so
// the homepage and the detail pages cannot drift apart or carry a hardcoded
// number that no longer matches the data.
import aiAgentJson from "@/data/ai_agent.json";
import deepHedgingJson from "@/data/deep_hedging.json";
import robustnessJson from "@/data/deep_hedging_robustness.json";
import pnlExplainJson from "@/data/pnl_explain.json";
import varJson from "@/data/var.json";
import volPcaJson from "@/data/vol_pca.json";
import volSurfaceJson from "@/data/vol_surface.json";
import { maturityLabel } from "@/lib/format";
import type {
  AiAgentData,
  DeepHedgeData,
  DeepHedgeRow,
  PnlExplainData,
  RobustnessData,
  RobustnessRow,
  RobustnessScenario,
  VarData,
  VolPcaData,
  VolSurfaceData,
} from "@/lib/types";

export const aiAgent = aiAgentJson as AiAgentData;
export const deepHedging = deepHedgingJson as DeepHedgeData;
export const robustness = robustnessJson as RobustnessData;
export const pnlExplain = pnlExplainJson as PnlExplainData;
export const volSurface = volSurfaceJson as VolSurfaceData;
export const volPca = volPcaJson as VolPcaData;
export const varData = varJson as VarData;

// ---- P&L explain -----------------------------------------------------------

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

export const DRIVER_LABELS: Record<string, string> = {
  time: "Passage of time",
  rates_divs: "Rates & dividends",
  spot: "Stock price moves",
  vol: "Volatility changes",
};

export const DRIVER_SHORT_LABELS: Record<string, string> = {
  time: "Time",
  rates_divs: "Rates",
  spot: "Stock",
  vol: "Vol",
};

const topPosition = pnlExplain.by_position[0];
const residualTotal = sum(pnlExplain.waterfall.map((s) => s.residual));
const breaches = pnlExplain.residual_series.filter((p) => p.status === "alert");
const improvement = pnlExplain.improvement;

export const STATUS_LABELS: Record<PnlExplainData["status"], string> = {
  alert: "Alert",
  inconclusive: "Within noise",
  ok: "Clear",
};

export const pnl = {
  actual: sum(pnlExplain.waterfall.map((s) => s.actual_pnl)),
  expected: sum(pnlExplain.waterfall.map((s) => s.explained_pnl)),
  unexplained: residualTotal,
  unexplainedPctOfBook: (100 * residualTotal) / pnlExplain.nav,
  alertPctOfBook: pnlExplain.residual_alert_threshold_bp / 100,
  topPositionId: topPosition.position_id,
  topPositionLabel: pnlExplain.positions[topPosition.position_id] ?? topPosition.position_id,
  topPositionShare: (100 * topPosition.residual) / residualTotal,
  daysRecorded: pnlExplain.residual_series.length,
  daysBreached: breaches.length,
  daysSameTopPosition: pnlExplain.residual_series.filter(
    (p) => p.top_position_id === topPosition.position_id,
  ).length,
  daysWithBreakdown: pnlExplain.residual_series.filter((p) => p.top_position_id !== null).length,
  daysInconclusive: pnlExplain.residual_series.filter((p) => p.status === "inconclusive").length,
  daysWithinThreshold: pnlExplain.residual_series.filter((p) => p.status === "ok").length,
  // The featured (latest) day's verdict: alert only beyond the noise band.
  status: pnlExplain.status,
  noiseBandPct: pnlExplain.noise_band_bp / 100,
  // How the engine was improved (fixed values from the follow-up investigation).
  meanGapBeforePct: improvement.mean_abs_bp_before / 100,
  meanGapAfterPct: improvement.mean_abs_bp_after / 100,
  meanGapReduction: 100 * (1 - improvement.mean_abs_bp_after / improvement.mean_abs_bp_before),
  findings: improvement.findings,
};

// ---- Hedging ---------------------------------------------------------------

export const OBJECTIVE_LABELS: Record<DeepHedgeRow["loss_type"], string> = {
  variance: "Stability",
  cvar: "Tail protection",
  cost: "Cost-aware",
};

export const INSTRUMENT_LABELS: Record<DeepHedgeRow["instrument"], string> = {
  vanilla: "NVIDIA option",
  autocall: "NVIDIA structured note",
  barrier: "S&P 500 barrier option",
};

export const INSTRUMENT_SHORT_LABELS: Record<DeepHedgeRow["instrument"], string> = {
  vanilla: "Option",
  autocall: "Note",
  barrier: "Barrier",
};

/** Percent improvement in the average of the worst 5% of outcomes (seed mean). */
export function tailImprovementPct(r: DeepHedgeRow): number {
  return r.cvar_improvement_pct;
}

function row(instrument: DeepHedgeRow["instrument"], loss: DeepHedgeRow["loss_type"]) {
  const r = deepHedging.rows.find((x) => x.instrument === instrument && x.loss_type === loss);
  if (!r) throw new Error(`missing deep-hedge row ${instrument}/${loss}`);
  return r;
}

const barrierRows = deepHedging.rows.filter((r) => r.instrument === "barrier");
const bestBarrier = barrierRows.reduce((a, b) => (b.std_reduction_pct > a.std_reduction_pct ? b : a));

export const hedging = {
  bestBarrierSwingReduction: bestBarrier.std_reduction_pct,
  bestBarrierObjective: OBJECTIVE_LABELS[bestBarrier.loss_type],
  barrierAllBeatBaseline: barrierRows.every(
    (r) => r.std_reduction_pct > 0 && tailImprovementPct(r) > 0,
  ),
  noteStabilitySwingReduction: row("autocall", "variance").std_reduction_pct,
  noteStabilityTail: tailImprovementPct(row("autocall", "variance")),
  noteTailProtectionTail: tailImprovementPct(row("autocall", "cvar")),
  noteTailProtectionSwingReduction: row("autocall", "cvar").std_reduction_pct,
  noteTailProtectionTradesLessPct:
    100 * (1 - row("autocall", "cvar").learned_turnover / row("autocall", "variance").learned_turnover),
  optionStabilitySwingReduction: row("vanilla", "variance").std_reduction_pct,
  optionStabilityMin: row("vanilla", "variance").std_reduction_pct_min,
  nSeeds: Math.min(...deepHedging.rows.map((r) => r.n_seeds)),
  // How much less Cost-aware trades than Stability, per product.
  costAwareTradesLessPct: (["vanilla", "autocall", "barrier"] as const).map(
    (inst) => 100 * (1 - row(inst, "cost").learned_turnover / row(inst, "variance").learned_turnover),
  ),
};

// ---- Hedging robustness (out-of-model) ---------------------------------------

export const SCENARIO_LABELS: Record<RobustnessScenario, string> = {
  in_sample: "In the lab",
  vol_up_25: "Volatility +25%",
  vol_down_25: "Volatility -25%",
  jumps: "Crash jumps",
  history: "Real history, 20y",
};

export const SCENARIO_SHORT_LABELS: Record<RobustnessScenario, string> = {
  in_sample: "Lab",
  vol_up_25: "Vol +25%",
  vol_down_25: "Vol -25%",
  jumps: "Jumps",
  history: "History",
};

export function robust(
  instrument: DeepHedgeRow["instrument"],
  loss: DeepHedgeRow["loss_type"],
  scenario: RobustnessScenario,
): RobustnessRow {
  const r = robustness.rows.find(
    (x) => x.instrument === instrument && x.loss_type === loss && x.scenario === scenario,
  );
  if (!r) throw new Error(`missing robustness row ${instrument}/${loss}/${scenario}`);
  return r;
}

const noteTailRows = robustness.scenarios.map((sc) => robust("autocall", "cvar", sc));

export const robustnessHeadlines = {
  barrierHistory: robust("barrier", "variance", "history").std_reduction_pct,
  optionHistory: robust("vanilla", "variance", "history").std_reduction_pct,
  noteStabilityLab: robust("autocall", "variance", "in_sample").std_reduction_pct,
  noteStabilityHistory: robust("autocall", "variance", "history").std_reduction_pct,
  noteTailMin: Math.min(...noteTailRows.map((r) => r.cvar_improvement_pct)),
  noteTailMax: Math.max(...noteTailRows.map((r) => r.cvar_improvement_pct)),
  historyYears: 20,
  noteIndependentWindows: robustness.history_independent["autocall"],
};

// ---- AI analyst ------------------------------------------------------------

interface RepriceResult {
  position_id: string;
  spot_shock_pct: number;
  vol_shock_pct: number;
  base_price: number;
  shocked_price: number;
  price_delta: number;
}

const firstCall = aiAgent.trace[0];
const reprice = firstCall ? (JSON.parse(firstCall.result_summary) as RepriceResult) : null;

export const ai = {
  // The featured investigation's own day (the most recent alerting day pair),
  // not the P&L page's case-study day.
  unexplained: aiAgent.unexplained,
  unexplainedPctOfBook: aiAgent.unexplained_bp / 100,
  topPositionLabel: pnlExplain.positions[aiAgent.top_position_id] ?? aiAgent.top_position_id,
  topPositionShare: aiAgent.top_position_share_pct,
  toolCalls: aiAgent.trace.length,
  reprice: reprice && {
    positionId: reprice.position_id,
    positionLabel: pnlExplain.positions[reprice.position_id] ?? reprice.position_id,
    underlying: aiAgent.position_underlyings[reprice.position_id] ?? "the underlying",
    spotShockPct: 100 * reprice.spot_shock_pct,
    volShockPct: 100 * reprice.vol_shock_pct,
    basePrice: reprice.base_price,
    priceDelta: reprice.price_delta,
    priceDeltaPct: (100 * reprice.price_delta) / reprice.base_price,
  },
};

// ---- Pricing / vol surface -------------------------------------------------

const rmses = volSurface.pillars.map((p) => p.rmse_vol_points).sort((a, b) => a - b);
const pillarTs = volSurface.pillars.map((p) => p.T);

export const pricing = {
  maturities: volSurface.pillars.length,
  maturityRange: `${maturityLabel(Math.min(...pillarTs))} to ${maturityLabel(Math.max(...pillarTs))}`,
  quotes: sum(volSurface.pillars.map((p) => p.n_points)),
  butterflyViolations: sum(volSurface.pillars.map((p) => p.butterfly_violations)),
  medianFitErrorVolPts:
    rmses.length % 2
      ? rmses[(rmses.length - 1) / 2]
      : (rmses[rmses.length / 2 - 1] + rmses[rmses.length / 2]) / 2,
  usedJointModel: volSurface.pillars.every((p) => p.model === "SSVI"),
};

// ---- Vol surface PCA -------------------------------------------------------

export const VOL_PCA_NAMES: Record<string, string> = {
  SPX: "S&P 500",
  AAPL: "Apple",
  NVDA: "NVIDIA",
};

const frontToOneYear = (n: VolPcaData["names"][number]) => {
  const front = n.pc1_by_maturity[0];
  const oneYear = n.pc1_by_maturity.find((m) => m.T === 1);
  return oneYear ? front.loading / oneYear.loading : NaN;
};
const ratios = volPca.names.map(frontToOneYear).filter(Number.isFinite);

export const volMoves = {
  names: volPca.names.map((n) => ({
    ...n,
    label: VOL_PCA_NAMES[n.underlying] ?? n.underlying,
    top3Pct: sum(n.explained_pct),
  })),
  minChanges: Math.min(...volPca.names.map((n) => n.n_changes)),
  minTop3Pct: Math.min(...volPca.names.map((n) => sum(n.explained_pct))),
  frontRatioLow: Math.min(...ratios),
  frontRatioHigh: Math.max(...ratios),
  windowDays: volPca.window_days,
};

// ---- Market risk (VaR) -------------------------------------------------------

const varswap = varData.positions.find((p) => p.label.includes("variance swap"));
const autocall = varData.positions.find((p) => p.label.includes("autocallable"));

export const marketRisk = {
  var1: varData.var1.full,
  var1Shortcut: varData.var1.taylor,
  /** How far the Greeks shortcut's VaR falls short of full revaluation, in %. */
  shortcutShortfallPct: (1 - varData.var1.taylor / varData.var1.full) * 100,
  es1: varData.es1.full,
  var1ExVarswap: varData.var1_ex_varswap,
  var10: varData.var10.overlapping,
  var10Sqrt: varData.var10.sqrt10,
  sqrt10OverPct: (varData.var10.sqrt10 / varData.var10.overlapping - 1) * 100,
  runtimeSeconds: varData.runtime.full_seconds,
  /** Shortcut P&L per dollar of full-revaluation P&L, in %. */
  varswapCapturePct: varswap ? varswap.taylor_capture * 100 : NaN,
  autocallCapturePct: autocall ? autocall.taylor_capture * 100 : NaN,
  autocallMaxGap: autocall?.max_gap,
  /** How many times faster the shortcut runs than full revaluation. */
  speedup: varData.runtime.full_seconds / varData.runtime.taylor_seconds,
  worstDay: varData.worst_days[0],
  seriesDays: varData.series.length,
  proxyNames: varData.proxy.names.map((n) => ({ ...n, label: VOL_PCA_NAMES[n.underlying] ?? n.underlying })),
};
