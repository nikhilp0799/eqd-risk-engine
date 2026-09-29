import type { Metadata } from "next";
import { GroupedBars } from "@/components/charts/GroupedBars";
import { Badge, Card, DemoNote, Methodology, PageHeader, Takeaway } from "@/components/ui";
import { SERIES } from "@/lib/chartTheme";
import { fmtDate, pct, signedPct } from "@/lib/format";
import {
  INSTRUMENT_LABELS,
  INSTRUMENT_SHORT_LABELS,
  OBJECTIVE_LABELS,
  deepHedging,
  hedging,
  tailImprovementPct,
} from "@/lib/metrics";
import type { DeepHedgeRow } from "@/lib/types";

export const metadata: Metadata = {
  title: "Hedging Strategy Lab — EQD Risk Engine",
  description:
    "AI-trained hedging strategies tested head-to-head against the benchmark hedge on fresh simulated markets, for an option, a structured note and a barrier option.",
};

const INSTRUMENT_ORDER: DeepHedgeRow["instrument"][] = ["barrier", "autocall", "vanilla"];
const OBJECTIVES: DeepHedgeRow["loss_type"][] = ["variance", "cvar", "cost"];

function chartData(metric: (r: DeepHedgeRow) => number) {
  return INSTRUMENT_ORDER.map((inst) => {
    const entry: Record<string, string | number> = {
      instrument: INSTRUMENT_LABELS[inst],
      short: INSTRUMENT_SHORT_LABELS[inst],
    };
    for (const obj of OBJECTIVES) {
      const r = deepHedging.rows.find((x) => x.instrument === inst && x.loss_type === obj);
      if (r) entry[obj] = metric(r);
    }
    return entry;
  });
}

type RangedMetric = "std_reduction_pct" | "cvar_improvement_pct";

function find(inst: DeepHedgeRow["instrument"], obj: DeepHedgeRow["loss_type"]) {
  return deepHedging.rows.find((x) => x.instrument === inst && x.loss_type === obj);
}

function ObjectiveHeader() {
  return (
    <>
      {OBJECTIVES.map((obj, i) => (
        <th key={obj} className="py-2 pr-4 text-right font-medium">
          <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: SERIES[i] }} />
          {OBJECTIVE_LABELS[obj]}
        </th>
      ))}
    </>
  );
}

/** Seed mean, with the min-to-max range across seeds underneath. */
function ResultsTable({ metric }: { metric: RangedMetric }) {
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-muted">
            <th className="py-2 pr-4 font-medium">Product</th>
            <ObjectiveHeader />
          </tr>
        </thead>
        <tbody>
          {INSTRUMENT_ORDER.map((inst) => (
            <tr key={inst} className="border-b border-border/60 last:border-0">
              <td className="py-2 pr-4 text-ink">{INSTRUMENT_LABELS[inst]}</td>
              {OBJECTIVES.map((obj) => {
                const r = find(inst, obj);
                if (!r) return <td key={obj} className="py-2 pr-4 text-right">—</td>;
                const v = r[metric];
                return (
                  <td key={obj} className="py-2 pr-4 text-right">
                    <span className={v < 0 ? "text-critical" : "text-ink"}>{signedPct(v, 1)}</span>
                    <span className="block text-xs text-muted">
                      {signedPct(r[`${metric}_min`], 1)} to {signedPct(r[`${metric}_max`], 1)}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const shares = (x: number) =>
  x >= 100 ? Math.round(x).toLocaleString("en-US") : x.toFixed(2);

/** Average shares traded per position, benchmark next to each strategy. */
function TradingTable() {
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-muted">
            <th className="py-2 pr-4 font-medium">Product</th>
            <th className="py-2 pr-4 text-right font-medium">Benchmark</th>
            <ObjectiveHeader />
          </tr>
        </thead>
        <tbody>
          {INSTRUMENT_ORDER.map((inst) => (
            <tr key={inst} className="border-b border-border/60 last:border-0">
              <td className="py-2 pr-4 text-ink">{INSTRUMENT_LABELS[inst]}</td>
              <td className="py-2 pr-4 text-right text-ink-2">
                {shares(find(inst, "variance")?.baseline_turnover ?? NaN)}
              </td>
              {OBJECTIVES.map((obj) => {
                const r = find(inst, obj);
                return (
                  <td key={obj} className="py-2 pr-4 text-right text-ink">
                    {r ? shares(r.learned_turnover) : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const series = OBJECTIVES.map((obj, i) => ({ key: obj, label: OBJECTIVE_LABELS[obj], color: SERIES[i] }));

const VERDICTS: { inst: DeepHedgeRow["instrument"]; badge: string; tone: "good" | "brand" | "neutral"; body: string }[] = [
  {
    inst: "barrier",
    badge: "Clear win",
    tone: "good",
    body: "Every AI-trained strategy beat the benchmark on both measures. A barrier option's risk jumps when the index nears the barrier, which a hedge set once at the start can't follow.",
  },
  {
    inst: "autocall",
    badge: "Real trade-off",
    tone: "brand",
    body: "Stability cuts swings by about a third but makes the worst outcomes slightly worse. Tail protection improves the worst outcomes, but swings get bigger and it trades far less. Which is better depends on what the desk cares about more.",
  },
  {
    inst: "vanilla",
    badge: "Small, consistent win",
    tone: "neutral",
    body: "A plain option is already hedged well by the textbook method, so the gain is small, but it holds on every training run. The learned hedge also closely tracks the textbook delta, a strong sign it learned real finance.",
  },
];

export default function DeepHedgingPage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        section="Hedging Strategy Lab"
        title="Test a smarter hedge before you trade it"
        lede="The lab trains hedging strategies on thousands of simulated market scenarios built from the day's real market, then pits them against the benchmark hedge on fresh scenarios they have never seen."
        meta={`Market date: ${fmtDate(deepHedging.asof_date)}`}
      />

      <Takeaway>
        On the S&amp;P 500 barrier option, every AI-trained strategy beat the benchmark on both
        measures, cutting P&amp;L swings by up to <strong>{pct(hedging.bestBarrierSwingReduction)}</strong>.
        On the NVIDIA structured note there is a genuine trade-off: Stability cuts swings by{" "}
        {pct(hedging.noteStabilitySwingReduction)} but its worst cases get{" "}
        {pct(Math.abs(hedging.noteStabilityTail), 1)} worse, while Tail protection improves the worst
        cases by {pct(hedging.noteTailProtectionTail, 1)} at the cost of bigger swings. Even on a
        plain option, already well served by the textbook hedge, Stability is{" "}
        {pct(hedging.optionStabilitySwingReduction, 1)} steadier. Every figure is the average of{" "}
        {hedging.nSeeds} independent training runs, with the range shown.
      </Takeaway>

      <Card title="How to read the results">
        <div className="grid gap-6 text-sm leading-relaxed text-ink-2 md:grid-cols-2">
          <div className="flex flex-col gap-3">
            <p>
              <strong className="text-ink">P&amp;L swings</strong>: how much the hedged result varies
              from one scenario to the next. Lower means a steadier book.
            </p>
            <p>
              <strong className="text-ink">Worst-case loss</strong>: the average result across the
              worst 5% of scenarios. Better means smaller losses when markets go badly.
            </p>
          </div>
          <div className="flex flex-col gap-3">
            <p>Three strategies, each trained to care about something different:</p>
            <ul className="flex flex-col gap-1.5">
              <li><strong className="text-ink">Stability</strong>: keep day-to-day swings small.</li>
              <li><strong className="text-ink">Tail protection</strong>: limit the worst outcomes.</li>
              <li><strong className="text-ink">Cost-aware</strong>: stability, with trading costs priced in.</li>
            </ul>
          </div>
        </div>
      </Card>

      <Card
        title="Reduction in P&L swings vs. the benchmark hedge"
        subtitle="Higher is better. Measured on scenarios the strategies never trained on."
      >
        <GroupedBars
          data={chartData((r) => r.std_reduction_pct)}
          categoryKey="instrument"
          shortCategoryKey="short"
          series={series}
          format="signedPct1"
          labels={false}
        />
        <ResultsTable metric="std_reduction_pct" />
      </Card>

      <Card
        title="Improvement in worst-case loss vs. the benchmark hedge"
        subtitle="Higher is better. Average of the worst 5% of scenarios."
      >
        <GroupedBars
          data={chartData(tailImprovementPct)}
          categoryKey="instrument"
          shortCategoryKey="short"
          series={series}
          format="signedPct1"
          labels={false}
        />
        <ResultsTable metric="cvar_improvement_pct" />
      </Card>

      <Card
        title="Trading activity"
        subtitle={`Average shares traded per position. Cost-aware trades ${pct(Math.min(...hedging.costAwareTradesLessPct), 0)}-${pct(Math.max(...hedging.costAwareTradesLessPct), 0)} less than Stability for almost the same stability: at 5bp per trade, trading is cheap next to the risk, so pricing costs in barely moves the best hedge. The note and barrier benchmarks trade once and hold, so their figures are not comparable with a rebalancing strategy.`}
      >
        <TradingTable />
      </Card>

      <div className="grid gap-4 md:grid-cols-3">
        {VERDICTS.map((v) => (
          <div key={v.inst} className="rounded-2xl border border-border bg-surface p-6">
            <Badge tone={v.tone}>{v.badge}</Badge>
            <h3 className="mt-3 font-semibold text-ink">{INSTRUMENT_LABELS[v.inst]}</h3>
            <p className="mt-1 text-sm leading-relaxed text-ink-2">{v.body}</p>
          </div>
        ))}
      </div>

      <Methodology>
        <p>
          <strong className="text-ink">Training.</strong> Each strategy is a small PyTorch neural
          network choosing the hedge ratio at every rebalance, trained on 8,000 Sobol paths from the
          engine&apos;s own local-volatility Monte Carlo, calibrated to that day&apos;s market.
          Training stops early once a separate 4,000-path validation set stops improving (at most
          3,000 epochs; every run stopped before the cap). Results are measured on a third,
          differently-seeded set of 8,000 paths, and each combination is trained {hedging.nSeeds}{" "}
          times with different seeds. Trading costs of 5bp apply to every rebalance, for both the
          strategies and the benchmark.
        </p>
        <p>
          <strong className="text-ink">Objectives.</strong> Stability minimises the variance of
          hedged P&amp;L; Tail protection minimises CVaR at 95%; Cost-aware minimises the standard
          deviation of hedged P&amp;L plus the expected dollar trading cost (both in dollars, so the
          trade-off does not depend on position size).
        </p>
        <p>
          <strong className="text-ink">Benchmarks, and their limits.</strong> The option benchmark
          is a Black-Scholes delta hedge rebalanced on the same schedule. For the structured note and
          the barrier option, the benchmark is a Monte Carlo delta hedge set once at inception and
          never rebalanced, because re-computing Monte Carlo Greeks at every step is too costly for
          this comparison. A dynamically re-hedged benchmark would be harder to beat, so the wins on
          those two products partly reflect a weaker benchmark.
        </p>
        <p>
          <strong className="text-ink">Simplifications.</strong> The structured note is rebalanced
          quarterly, at its observation dates. Barrier monitoring is discrete, at the simulation
          grid.
        </p>
      </Methodology>

      <DemoNote />
    </div>
  );
}
