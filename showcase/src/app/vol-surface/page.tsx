import type { Metadata } from "next";
import { GroupedBars } from "@/components/charts/GroupedBars";
import { SensitivityLine } from "@/components/charts/SensitivityLine";
import { SmileExplorer } from "@/components/charts/SmileExplorer";
import { Card, DemoNote, Methodology, PageHeader, StatTile, Takeaway } from "@/components/ui";
import { SERIES } from "@/lib/chartTheme";
import { fmtDate, maturityLabel, pct } from "@/lib/format";
import { pricing, volMoves, volSurface } from "@/lib/metrics";

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

const patternBars = volMoves.names.map((n) => ({
  name: n.label,
  p1: n.explained_pct[0],
  p2: n.explained_pct[1],
  p3: n.explained_pct[2],
}));

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
        {pricing.maturities} maturities, from {pricing.maturityRange}, fitted with a median error of{" "}
        <strong>{pricing.medianFitErrorVolPts.toFixed(1)} volatility points</strong> and no
        strike-to-strike pricing inconsistencies. Option prices computed from it agree with QuantLib,
        the industry&apos;s benchmark library, to eight decimal places.
      </Takeaway>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Market quotes used" value={pricing.quotes.toLocaleString("en-US")} />
        <StatTile label="Maturities covered" value={String(pricing.maturities)} detail={pricing.maturityRange} />
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

      <Card
        title="How the surface moves from day to day"
        subtitle={`Each day's change across the surface, split into a few recurring patterns (principal components), over ${volMoves.minChanges} or more trading days of real history. For every name, three patterns account for at least ${pct(volMoves.minTop3Pct)} of the day-to-day movement.`}
      >
        <GroupedBars
          data={patternBars}
          categoryKey="name"
          series={[
            { key: "p1", label: "Pattern 1", color: SERIES[0] },
            { key: "p2", label: "Pattern 2", color: SERIES[1] },
            { key: "p3", label: "Pattern 3", color: SERIES[2] },
          ]}
          format="pct1"
          max={100}
          narrowLabels={false}
        />
        <div className="mt-6 flex flex-col gap-3 text-sm leading-relaxed text-ink-2">
          <p>
            <strong className="text-ink">Pattern 1</strong> is the whole surface rising or falling
            together, with short-dated options moving about{" "}
            {Math.round(volMoves.frontRatioLow)} to {Math.round(volMoves.frontRatioHigh)} times as
            much as one-year options. Its shape holds in every {volMoves.windowDays}-day window of
            the history.
          </p>
          <p>
            <strong className="text-ink">Patterns 2 and 3</strong> are smaller: mostly short-dated
            volatility moving against longer-dated and, for Apple and NVIDIA, downside strikes
            moving against upside. Their shapes still shift from window to window, so they need more history before they
            can be relied on.
          </p>
        </div>
        <div className="mt-5 overflow-x-auto">
          <table className="tabular w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-muted">
                <th className="py-2 pr-4 font-medium">Underlying</th>
                <th className="py-2 pr-4 text-right font-medium">Daily changes</th>
                <th className="py-2 pr-4 text-right font-medium">Surface points</th>
                <th className="py-2 pr-4 text-right font-medium">Pattern 1</th>
                <th className="py-2 pr-4 text-right font-medium">Pattern 2</th>
                <th className="py-2 pr-4 text-right font-medium">Pattern 3</th>
                <th className="py-2 pr-4 text-right font-medium">Top three</th>
                <th className="py-2 text-right font-medium">Pattern 1 shape match</th>
              </tr>
            </thead>
            <tbody>
              {volMoves.names.map((n) => (
                <tr key={n.underlying} className="border-b border-border/60">
                  <td className="py-2 pr-4 text-ink">{n.label}</td>
                  <td className="py-2 pr-4 text-right">{n.n_changes}</td>
                  <td className="py-2 pr-4 text-right">{n.n_nodes}</td>
                  {n.explained_pct.map((x, i) => (
                    <td key={i} className="py-2 pr-4 text-right">
                      {pct(x, 1)}
                    </td>
                  ))}
                  <td className="py-2 pr-4 text-right font-medium text-ink">{pct(n.top3Pct, 1)}</td>
                  <td className="py-2 text-right">{n.lowest_window_match.pc1.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-muted">
            Shape match: the lowest similarity (1 = identical) between pattern 1 fitted on any{" "}
            {volMoves.windowDays}-day window and on the full history.
          </p>
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
          <strong className="text-ink">Surface movement.</strong> Each day the surface is read on a
          fixed grid of strikes and maturities, and the patterns come from the day-to-day changes
          in volatility at those points. Changes are measured in volatility points, because in
          total variance the same move is about 24 times bigger at two years than at one month
          and the long end would swamp everything else. Only points inside the quoted strikes on
          at least 80% of days are used, since outside the quotes the surface is an extrapolation
          that moves with the fit, not the market. Only changes between consecutive trading days
          count.
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
