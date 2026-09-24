import type { Metadata } from "next";
import { Panel, Pill } from "@/components/Panel";
import { DeepHedgeBars } from "@/components/charts/DeepHedgeBars";
import { fmtDate, fmtPct, fmtUsd } from "@/lib/format";
import deepHedging from "@/data/deep_hedging.json";
import type { DeepHedgeData, DeepHedgeRow } from "@/lib/types";

export const metadata: Metadata = {
  title: "Deep Hedging — EQD Risk Engine",
  description:
    "Neural-network hedging policies trained under variance/CVaR/cost-adjusted losses, benchmarked out-of-sample against classical Greeks across a vanilla, autocallable, and barrier option.",
};

const data = deepHedging as DeepHedgeData;

const INSTRUMENTS: {
  key: DeepHedgeRow["instrument"];
  title: string;
  underlying: string;
  description: string;
  finding: string;
}[] = [
  {
    key: "vanilla",
    title: "Vanilla option",
    underlying: "NVDA",
    description:
      "A single-maturity European call, the simplest instrument in the book — the vertical slice this project's deep-hedging work started from.",
    finding:
      "All three learned policies land close to the Black-Scholes delta baseline, as expected for an instrument this well-approximated by closed-form delta hedging — no dramatic win, and that itself is a sanity check that the training pipeline is behaving correctly.",
  },
  {
    key: "autocall",
    title: "Autocallable note",
    underlying: "NVDA",
    description:
      "A path-dependent structured note with quarterly observation dates and early-redemption risk — the book's largest, riskiest position.",
    finding:
      "A genuine tradeoff, not a clean win: the variance-loss policy cuts P&L standard deviation by about a third but leaves tail risk (CVaR) slightly worse than the baseline. The CVaR-loss policy does the opposite — better tail risk, far less variance reduction. No single objective wins on every metric, and that is reported here as a real finding, not hidden.",
  },
  {
    key: "barrier",
    title: "Down-and-in barrier put",
    underlying: "SPX",
    description:
      "A discretely-monitored knock-in put, hedged with a Brownian-bridge-corrected Monte Carlo baseline delta.",
    finding:
      "A clean sweep: every learned policy beats its static baseline on both variance and CVaR, for every loss objective. Unlike the autocallable, there is no tradeoff here — the static baseline is simply weak against a barrier's discontinuous risk profile.",
  },
];

export default function DeepHedgingPage() {
  return (
    <div className="flex flex-col gap-10">
      <header className="flex flex-col gap-3">
        <p className="font-num text-sm text-accent">deep-hedging / {fmtDate(data.asof_date)}</p>
        <h1 className="text-3xl font-semibold text-foreground">Deep hedging</h1>
        <p className="text-muted max-w-2xl">
          Neural-network hedging policies (PyTorch), trained against this project&apos;s own
          calibrated local-vol Monte Carlo simulator under three loss objectives — variance,
          CVaR (tail risk), and cost-adjusted — then evaluated out-of-sample against a classical
          Greeks-based baseline. A research extension beyond the engine&apos;s original 16-step
          plan, never wired into the daily pricing pipeline.
        </p>
      </header>

      {INSTRUMENTS.map((inst) => {
        const rows = data.rows.filter((r) => r.instrument === inst.key);
        if (rows.length === 0) return null;
        return (
          <Panel key={inst.key} title={`${inst.title} — ${inst.underlying}`} caption={inst.description}>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
              <DeepHedgeBars rows={rows} />
              <div className="overflow-x-auto">
                <table className="w-full text-sm font-num">
                  <thead>
                    <tr className="text-left text-muted border-b border-panel-border">
                      <th className="py-2 pr-4 font-normal">loss</th>
                      <th className="py-2 pr-4 font-normal">std reduction</th>
                      <th className="py-2 pr-4 font-normal">learned CVaR</th>
                      <th className="py-2 font-normal">baseline CVaR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.loss_type} className="border-b border-panel-border/50">
                        <td className="py-2 pr-4 text-foreground">{r.loss_type}</td>
                        <td className="py-2 pr-4">
                          <Pill tone={r.std_reduction_pct >= 0 ? "positive" : "negative"}>
                            {fmtPct(r.std_reduction_pct)}
                          </Pill>
                        </td>
                        <td className="py-2 pr-4">
                          {inst.key === "vanilla"
                            ? fmtUsd(r.learned_cvar, 2)
                            : fmtUsd(r.learned_cvar)}
                        </td>
                        <td className="py-2">
                          {inst.key === "vanilla"
                            ? fmtUsd(r.baseline_cvar, 2)
                            : fmtUsd(r.baseline_cvar)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <p className="text-sm text-muted mt-5 border-t border-panel-border pt-4">
              {inst.finding}
            </p>
          </Panel>
        );
      })}
    </div>
  );
}
