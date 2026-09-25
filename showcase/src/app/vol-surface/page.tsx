import type { Metadata } from "next";
import { SensitivityLine } from "@/components/charts/SensitivityLine";
import { SmileExplorer } from "@/components/charts/SmileExplorer";
import { Card, DemoNote, Methodology, PageHeader, StatTile, Takeaway } from "@/components/ui";
import { fmtDate, maturityLabel } from "@/lib/format";
import { pricing, volSurface } from "@/lib/metrics";

export const metadata: Metadata = {
  title: "Market-Consistent Pricing — EQD Risk Engine",
  description:
    "Live S&P 500 option quotes turned into one consistent pricing surface each day, the foundation for every valuation and risk number.",
};

const ladder = volSurface.greeks_ladder;
const ladderT = volSurface.pillars.find((p) => p.expiry === ladder.expiry)?.T;
const delta = ladder.strike.map((strike, i) => ({ strike, value: ladder.delta_spot[i] }));
// Stored vega is per 1.00 (100 vol points) change in volatility; shown per point.
const vega = ladder.strike.map((strike, i) => ({ strike, value: ladder.vega[i] / 100 }));

export default function VolSurfacePage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        section="Market-Consistent Pricing"
        title="Every price starts from the market"
        lede={`Each day the engine reads live ${volSurface.underlying === "SPX" ? "S&P 500" : volSurface.underlying} option quotes and fits them into one smooth, consistent pricing surface. Every valuation, sensitivity and stress test in the product is built on it.`}
        meta={`Market date: ${fmtDate(volSurface.asof_date)}`}
      />

      <Takeaway>
        <strong>{pricing.quotes.toLocaleString("en-US")}</strong> market quotes across{" "}
        {pricing.maturities} maturities, from next-day to one year, fitted with a median error of{" "}
        <strong>{pricing.medianFitErrorVolPts.toFixed(1)} volatility points</strong> and no
        strike-to-strike pricing inconsistencies. Option prices computed from it agree with QuantLib,
        the industry&apos;s benchmark library, to eight decimal places.
      </Takeaway>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Market quotes used" value={pricing.quotes.toLocaleString("en-US")} />
        <StatTile label="Maturities covered" value={String(pricing.maturities)} detail="Next day to one year" />
        <StatTile
          label="Median fit error"
          value={`${pricing.medianFitErrorVolPts.toFixed(1)} pts`}
          detail="Volatility points, fitted vs. quoted"
        />
        <StatTile
          label="Strike arbitrage found"
          value={String(pricing.butterflyViolations)}
          detail="Inconsistent prices across strikes"
          tone={pricing.butterflyViolations === 0 ? "good" : "critical"}
        />
      </div>

      <Card
        title="How the market prices risk across strikes"
        subtitle="Dots are real quotes; the line is the engine's fitted curve. Protection against a fall (strikes below 100%) costs more than upside, the market's familiar 'skew'. Pick a maturity to compare."
      >
        <SmileExplorer smiles={volSurface.smiles} />
      </Card>

      <Card
        title="Risk sensitivities, straight from the surface"
        subtitle={`Call options expiring in ${ladderT ? maturityLabel(ladderT) : ladder.expiry}, by strike. These feed the daily P&L explain and the hedging lab.`}
      >
        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <p className="text-sm font-medium text-ink">Delta</p>
            <p className="mb-3 text-sm text-muted">Change in option value per 1-point move in the index</p>
            <SensitivityLine data={delta} name="Delta" format="fixed2" />
          </div>
          <div>
            <p className="text-sm font-medium text-ink">Vega</p>
            <p className="mb-3 text-sm text-muted">Change in option value per 1-point rise in volatility</p>
            <SensitivityLine data={vega} name="Vega" format="fixed1" />
          </div>
        </div>
      </Card>

      <Methodology>
        <p>
          <strong className="text-ink">Fitting.</strong> Quotes pass a quality filter (stale, crossed
          and wide markets removed), are inverted to Black-76 implied volatilities against
          parity-implied forwards, and each maturity is fitted with SVI. If the per-maturity fits
          would be inconsistent across maturities (calendar arbitrage), the engine switches to a
          joint SSVI fit across all maturities.{" "}
          {pricing.usedJointModel && "On this date it did, which is why every maturity below shows SSVI."}
        </p>
        <p>
          <strong className="text-ink">Validation.</strong> Black-76 prices and Greeks are
          cross-checked against QuantLib: price to 1e-8, several Greeks to 1e-6.
        </p>
        <div className="overflow-x-auto">
          <table className="tabular w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-muted">
                <th className="py-2 pr-4 font-medium">Maturity</th>
                <th className="py-2 pr-4 font-medium">Expiry</th>
                <th className="py-2 pr-4 font-medium">Model</th>
                <th className="py-2 pr-4 text-right font-medium">Quotes</th>
                <th className="py-2 pr-4 text-right font-medium">RMSE (vol pts)</th>
                <th className="py-2 text-right font-medium">Max error (vol pts)</th>
              </tr>
            </thead>
            <tbody>
              {volSurface.pillars.map((p) => (
                <tr key={p.expiry} className="border-b border-border/60">
                  <td className="py-2 pr-4 text-ink">{maturityLabel(p.T)}</td>
                  <td className="py-2 pr-4">{p.expiry}</td>
                  <td className="py-2 pr-4">{p.model}</td>
                  <td className="py-2 pr-4 text-right">{p.n_points}</td>
                  <td className="py-2 pr-4 text-right">{p.rmse_vol_points.toFixed(2)}</td>
                  <td className="py-2 text-right">{p.max_abs_error_vol_points.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Methodology>

      <DemoNote />
    </div>
  );
}
