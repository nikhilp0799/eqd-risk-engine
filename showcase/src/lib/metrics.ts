// Every headline figure on the site is derived here from the exported JSON, so
// the homepage and the detail pages cannot drift apart or carry a hardcoded
// number that no longer matches the data.
import aiAgentJson from "@/data/ai_agent.json";
import deepHedgingJson from "@/data/deep_hedging.json";
import pnlExplainJson from "@/data/pnl_explain.json";
import volSurfaceJson from "@/data/vol_surface.json";
import type {
  AiAgentData,
  DeepHedgeData,
  DeepHedgeRow,
  PnlExplainData,
  VolSurfaceData,
} from "@/lib/types";

export const aiAgent = aiAgentJson as AiAgentData;
export const deepHedging = deepHedgingJson as DeepHedgeData;
export const pnlExplain = pnlExplainJson as PnlExplainData;
export const volSurface = volSurfaceJson as VolSurfaceData;

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
const breaches = pnlExplain.residual_series.filter(
  (p) => Math.abs(p.residual_bp) > pnlExplain.residual_alert_threshold_bp,
);

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

/** Percent improvement in the average of the worst 5% of outcomes. */
export function tailImprovementPct(r: DeepHedgeRow): number {
  return (100 * (r.learned_cvar - r.baseline_cvar)) / Math.abs(r.baseline_cvar);
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
  toolCalls: aiAgent.trace.length,
  reprice: reprice && {
    positionLabel: pnlExplain.positions[reprice.position_id] ?? reprice.position_id,
    spotShockPct: 100 * reprice.spot_shock_pct,
    volShockPct: 100 * reprice.vol_shock_pct,
    basePrice: reprice.base_price,
    priceDelta: reprice.price_delta,
    priceDeltaPct: (100 * reprice.price_delta) / reprice.base_price,
  },
};

// ---- Pricing / vol surface -------------------------------------------------

const rmses = volSurface.pillars.map((p) => p.rmse_vol_points).sort((a, b) => a - b);

export const pricing = {
  maturities: volSurface.pillars.length,
  quotes: sum(volSurface.pillars.map((p) => p.n_points)),
  butterflyViolations: sum(volSurface.pillars.map((p) => p.butterfly_violations)),
  medianFitErrorVolPts:
    rmses.length % 2
      ? rmses[(rmses.length - 1) / 2]
      : (rmses[rmses.length / 2 - 1] + rmses[rmses.length / 2]) / 2,
  usedJointModel: volSurface.pillars.every((p) => p.model === "SSVI"),
};
