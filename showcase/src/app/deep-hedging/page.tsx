import type { Metadata } from "next";
import { GroupedBars } from "@/components/charts/GroupedBars";
import { Badge, Card, DemoNote, Methodology, PageHeader, Takeaway } from "@/components/ui";
import { SERIES } from "@/lib/chartTheme";
import { fmtDate, pct, signedPct } from "@/lib/format";
import {
  INSTRUMENT_LABELS,
  INSTRUMENT_SHORT_LABELS,
  OBJECTIVE_LABELS,
  SCENARIO_LABELS,
  SCENARIO_SHORT_LABELS,
  deepHedging,
  hedging,
  robust,
  robustness,
  robustnessHeadlines as rh,
  tailImprovementPct,
} from "@/lib/metrics";
import type { DeepHedgeRow } from "@/lib/types";

const INSTRUMENT_ORDER: DeepHedgeRow["instrument"][] = ["barrier", "autocall", "vanilla"];
const OBJECTIVES: DeepHedgeRow["loss_type"][] = ["variance", "cvar", "cost"];

function robustChartData(inst: DeepHedgeRow["instrument"]) {
  return robustness.scenarios.map((sc) => {
    const entry: Record<string, string | number> = {
      scenario: SCENARIO_LABELS[sc],
      short: SCENARIO_SHORT_LABELS[sc],
    };
    for (const obj of OBJECTIVES) entry[obj] = robust(inst, obj, sc).std_reduction_pct;
    return entry;
  });
}

/** Worst-case improvement by objective (rows) and scenario (columns). */
function RobustTailTable({ inst }: { inst: DeepHedgeRow["instrument"] }) {
  return (
    <div className="mt-4 overflow-x-auto">
      <p className="mb-2 text-sm font-medium text-ink">Improvement in worst-case loss</p>
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-muted">
            <th className="py-2 pr-4 font-medium">Strategy</th>
            {robustness.scenarios.map((sc) => (
              <th key={sc} className="py-2 pr-4 text-right font-medium">
                {SCENARIO_LABELS[sc]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {OBJECTIVES.map((obj, i) => (
            <tr key={obj} className="border-b border-border/60 last:border-0">
              <td className="py-2 pr-4 text-ink">
                <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: SERIES[i] }} />
                {OBJECTIVE_LABELS[obj]}
              </td>
              {robustness.scenarios.map((sc) => {
                const v = robust(inst, obj, sc).cvar_improvement_pct;
                return (
                  <td key={sc} className={`py-2 pr-4 text-right ${v < 0 ? "text-critical" : "text-ink"}`}>
                    {signedPct(v, 1)}
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

export const metadata: Metadata = {
  title: "Hedging Strategy Lab — EQD Risk Engine",
  description:
    "AI-trained hedging strategies tested head-to-head against the benchmark hedge on fresh simulated markets, for an option, a structured note and a barrier option.",
};


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
    body: "Every AI-trained strategy beat the benchmark on both measures, in the lab and on 20 years of real S&P 500 history. A barrier option's risk jumps when the index nears the barrier, which a hedge set once at the start can't follow.",
  },
  {
    inst: "autocall",
    badge: "Real trade-off",
    tone: "brand",
    body: "In the lab, Stability cuts swings by about a fifth, but that edge does not survive real NVIDIA history. Tail protection improves the worst outcomes in every test, real history included: the robust choice, at the price of bigger swings.",
  },
  {
    inst: "vanilla",
    badge: "Small, consistent win",
    tone: "neutral",
    body: "A plain option is already hedged well by the textbook method, so the gain is small, but it holds on every training run and on real history. The learned hedge also closely tracks the textbook delta, a strong sign it learned real finance.",
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
        {pct(hedging.optionStabilitySwingReduction, 1)} steadier. Out of the lab, on{" "}
        {rh.historyYears} years of real prices, the barrier&apos;s edge holds (
        {signedPct(rh.barrierHistory, 0)}) but the note&apos;s Stability edge does not; see below.
        Every figure is the average of {hedging.nSeeds} independent training runs.
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

      <Card
        title="Does it hold up outside the lab?"
        subtitle={`Each strategy was trained in the lab, on the engine's own market simulator. Then, with no retraining, it was run through markets that behave differently: volatility 25% higher or lower than priced, sudden crash-style drops, and ${rh.historyYears} years of real price history (2006 to 2026, through 2008 and 2020). The benchmark faces exactly the same markets.`}
      >
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-ink-2">
          <li>
            <strong className="text-ink">Barrier option: the edge holds everywhere.</strong> Still{" "}
            {pct(rh.barrierHistory)} steadier than the benchmark on real S&amp;P 500 history.
          </li>
          <li>
            <strong className="text-ink">Plain option: small, but it holds.</strong>{" "}
            {signedPct(rh.optionHistory, 1)} on real NVIDIA history.
          </li>
          <li>
            <strong className="text-ink">NVIDIA note: the lab favourite fails in reality.</strong>{" "}
            Stability is {signedPct(rh.noteStabilityLab, 0)} in the lab and survives every simulated
            stress, but on real history it is {pct(Math.abs(rh.noteStabilityHistory), 0)} worse than
            the benchmark. Tail protection improves the worst cases by{" "}
            {pct(rh.noteTailMin, 1)}-{pct(rh.noteTailMax, 1)} in every test. A strategy that looks
            best in the lab can be the wrong one in the market, which is why the lab tests it before
            anyone trades it.
          </li>
        </ul>
        <p className="mt-4 text-xs text-muted">
          Real-history windows overlap, so 20 years gives only about{" "}
          {Math.round(rh.noteIndependentWindows)} independent one-year samples for the note: strong
          evidence, not proof.
        </p>
      </Card>

      {INSTRUMENT_ORDER.map((inst) => (
        <Card
          key={inst}
          title={`${INSTRUMENT_LABELS[inst]}: reduction in P&L swings, by market`}
          subtitle="Higher is better. Same strategies, never retrained, in five different markets."
        >
          <GroupedBars
            data={robustChartData(inst)}
            categoryKey="scenario"
            shortCategoryKey="short"
            series={series}
            format="signedPct1"
            labels={false}
            height={260}
          />
          <RobustTailTable inst={inst} />
        </Card>
      ))}

      <Card title="Found and fixed by testing outside the lab">
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-ink-2">
          <li>
            <strong className="text-ink">A simulation error.</strong> The note&apos;s market paths
            were simulated one step per quarter, which made NVIDIA&apos;s first three months less
            than half as volatile as the market priced (17% vs 39%). Now simulated 16 steps per
            quarter. Together with the position limit below, this took the note&apos;s lab result from
            about 33% to 19%.
          </li>
          <li>
            <strong className="text-ink">A missing position limit.</strong> Without one, a strategy
            facing moves it had never seen traded about three times as much as its siblings and
            blew up on real history (its P&amp;L swings about 54 times the benchmark&apos;s). Every strategy now trades under a limit, as on any
            real desk.
          </li>
        </ul>
        <p className="mt-3 text-xs text-muted">Every number on this page is after both fixes.</p>
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
          <strong className="text-ink">Out-of-model tests.</strong> Saved policies are scored with
          no retraining on: local volatility scaled by 1.25 and 0.75; compensated Merton jumps
          (one a year on average, mean -10%, std 5%) overlaid on local-vol paths; and rolling
          windows of 20 years of daily closes (stride 5 days), rescaled to start at the calibration
          spot and sampled at the training grid. Both hedgers stay as calibrated on the as-of date.
          The lab scenario reproduces the stored training results exactly.
        </p>
        <p>
          <strong className="text-ink">Position limit.</strong> The policy&apos;s output passes
          through <code>L · tanh(raw / L)</code>: L = 1.5 shares per option for the plain option and
          barrier, 1.0 times notional for the note. Near-identity for normal hedges.
        </p>
        <p>
          <strong className="text-ink">Simplifications.</strong> The structured note is rebalanced
          quarterly, at its observation dates, with its price path simulated 16 steps per quarter.
          Barrier monitoring is discrete, at the simulation grid.
        </p>
      </Methodology>

      <DemoNote />
    </div>
  );
}
