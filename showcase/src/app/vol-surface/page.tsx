import type { Metadata } from "next";
import { Panel, Pill } from "@/components/Panel";
import { SmileExplorer } from "@/components/charts/SmileExplorer";
import { GreeksLadderChart } from "@/components/charts/GreeksLadderChart";
import { fmtDate } from "@/lib/format";
import volSurface from "@/data/vol_surface.json";
import type { VolSurfaceData } from "@/lib/types";

export const metadata: Metadata = {
  title: "Volatility Surface & Greeks — EQD Risk Engine",
  description:
    "SVI/SSVI calibration on real SPX option chains, arbitrage-checked, driving a full closed-form Greek set validated against QuantLib.",
};

const data = volSurface as VolSurfaceData;
const totalButterflyViolations = data.pillars.reduce((s, p) => s + p.butterfly_violations, 0);

export default function VolSurfacePage() {
  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-3">
        <p className="font-num text-sm text-accent">
          vol-surface / {data.underlying} / {fmtDate(data.asof_date)}
        </p>
        <h1 className="text-3xl font-semibold text-foreground">
          Volatility surface &amp; Greeks
        </h1>
        <p className="text-muted max-w-2xl">
          {data.pillars.length} expiries calibrated from real {data.underlying} option-chain
          quotes, arbitrage-checked (butterfly and calendar), driving a closed-form Greek set
          validated against QuantLib to 1e-8 in price.
        </p>
        <div className="flex gap-2">
          <Pill tone={totalButterflyViolations === 0 ? "positive" : "negative"}>
            {totalButterflyViolations} butterfly violation(s)
          </Pill>
        </div>
      </header>

      <Panel
        title="Calibrated smile, fit vs. market"
        caption="Each curve is the already-calibrated SVI/SSVI parameters for that expiry, evaluated at a log-moneyness grid — not a live refit."
      >
        <SmileExplorer smiles={data.smiles} />
      </Panel>

      <Panel
        title="Delta and vega across strikes"
        caption={`Expiry ${data.greeks_ladder.expiry} — full Black-76 Greek set, cross-checked against QuantLib at calibration time.`}
      >
        <GreeksLadderChart ladder={data.greeks_ladder} />
      </Panel>

      <Panel title="Calibration quality by expiry">
        <div className="overflow-x-auto">
          <table className="w-full text-sm font-num">
            <thead>
              <tr className="text-left text-muted border-b border-panel-border">
                <th className="py-2 pr-4 font-normal">expiry</th>
                <th className="py-2 pr-4 font-normal">T</th>
                <th className="py-2 pr-4 font-normal">model</th>
                <th className="py-2 pr-4 font-normal">points</th>
                <th className="py-2 pr-4 font-normal">RMSE (vol pts)</th>
                <th className="py-2 font-normal">max err (vol pts)</th>
              </tr>
            </thead>
            <tbody>
              {data.pillars.map((p) => (
                <tr key={p.expiry} className="border-b border-panel-border/50">
                  <td className="py-2 pr-4 text-foreground">{p.expiry}</td>
                  <td className="py-2 pr-4">{p.T.toFixed(3)}</td>
                  <td className="py-2 pr-4">{p.model}</td>
                  <td className="py-2 pr-4">{p.n_points}</td>
                  <td className="py-2 pr-4">{p.rmse_vol_points.toFixed(4)}</td>
                  <td className="py-2">{p.max_abs_error_vol_points.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
