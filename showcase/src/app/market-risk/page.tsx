import type { Metadata } from "next";
import { LossHistogram } from "@/components/charts/LossHistogram";
import { RankedBars } from "@/components/charts/RankedBars";
import { VarHistory } from "@/components/charts/VarHistory";
import { Card, DemoNote, Methodology, PageHeader, StatTile, Takeaway } from "@/components/ui";
import { fmtDate, pct, signedUsdCompact, usdCompact } from "@/lib/format";
import { marketRisk, varData } from "@/lib/metrics";

export const metadata: Metadata = {
  title: "Market Risk — EQD Risk Engine",
  description:
    "How much the book could lose on a bad day, from years of real market moves replayed on today's positions, computed two ways.",
};

const factorBars = [
  { label: "Everything moves", value: varData.es10.all },
  { label: "Stock prices only", value: varData.es10.spot },
  { label: "Volatility only", value: varData.es10.vol },
  { label: "Single-stock volatility only", value: varData.es10.single_name_vol },
];

const positionRows = varData.positions.filter((p) => p.rms_full >= 1_000);

export default function MarketRiskPage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        section="Market Risk"
        title="How much could the book lose on a bad day?"
        lede={`Each day the engine replays the market moves of the last ${varData.window_days} trading days on today's positions and reprices everything. The worst of those outcomes is the book's Value at Risk.`}
        meta={`Book date: ${fmtDate(varData.asof_date)}`}
      />

      <Takeaway>
        On 99 of every 100 replayed days the book loses less than{" "}
        <strong>{usdCompact(marketRisk.var1)}</strong>. The faster shortcut most systems use,
        which estimates each loss from the positions&apos; sensitivities, says{" "}
        <strong>{usdCompact(marketRisk.var1Shortcut)}</strong>:{" "}
        {pct(marketRisk.shortcutShortfallPct)} too low, mostly because of one position, the S&amp;P
        500 variance swap.
      </Takeaway>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="1-day Value at Risk (99%)"
          value={usdCompact(marketRisk.var1)}
          detail="Full repricing of every position"
        />
        <StatTile
          label="Average of the worst 1%"
          value={usdCompact(marketRisk.es1)}
          detail="Expected Shortfall: how bad the bad days are"
        />
        <StatTile
          label="10-day Value at Risk (99%)"
          value={usdCompact(marketRisk.var10)}
          detail="From real 10-day market moves"
        />
        <StatTile
          label="Time to compute"
          value={`${Math.round(marketRisk.runtimeSeconds)} s`}
          detail={`${varData.window_days} full repricings of the book`}
        />
      </div>

      <Card
        title={`${varData.n_scenarios} past days, replayed on today's book`}
        subtitle={`Profit or loss today's positions would make if each trading day since ${fmtDate(varData.first_scenario_date)} happened again tomorrow. Red: the ranges past the 99% Value at Risk line. Not shown: the ${varData.histogram_hidden.count} days with gains above ${usdCompact(varData.histogram_hidden.above)}.`}
      >
        <LossHistogram bins={varData.histogram} varLoss={varData.var1.full} />
        <p className="mt-4 text-sm leading-relaxed text-ink-2">
          The worst replay is {fmtDate(marketRisk.worstDay.date)}:{" "}
          {signedUsdCompact(marketRisk.worstDay.full)}. Without the variance swap the 1-day
          Value at Risk would be {usdCompact(marketRisk.var1ExVarswap)}: it is the book&apos;s
          largest volatility position, and the worst days for it are the days fear drains out of
          the market.
        </p>
      </Card>

      <Card
        title="Where the sensitivity shortcut breaks"
        subtitle={`Banks often estimate each scenario's loss from a position's sensitivities instead of repricing it: here it ran about ${Math.round(marketRisk.speedup).toLocaleString("en-US")} times faster. This is how much of each position's real profit or loss that shortcut captures.`}
      >
        <div className="overflow-x-auto">
          <table className="tabular w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-muted">
                <th className="py-2 pr-4 font-medium">Position</th>
                <th className="py-2 pr-4 text-right font-medium">Typical daily move</th>
                <th className="py-2 pr-4 text-right font-medium">Shortcut error</th>
                <th className="py-2 text-right font-medium">Share captured</th>
              </tr>
            </thead>
            <tbody>
              {positionRows.map((p) => (
                <tr key={p.id} className="border-b border-border/60">
                  <td className="py-2 pr-4 text-ink">{p.label}</td>
                  <td className="py-2 pr-4 text-right">{usdCompact(p.rms_full)}</td>
                  <td className="py-2 pr-4 text-right">{usdCompact(p.rms_gap)}</td>
                  <td className="py-2 text-right">{pct(p.taylor_capture * 100)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-muted">
            Typical move and error: root-mean-square over the {varData.n_scenarios} replayed days.
            Share captured: the shortcut&apos;s profit or loss per dollar of the real one.
          </p>
        </div>
        <div className="mt-5 flex flex-col gap-3 text-sm leading-relaxed text-ink-2">
          <p>
            <strong className="text-ink">The variance swap</strong>: the shortcut captures{" "}
            {pct(marketRisk.varswapCapturePct)}. It reads the volatility move at one strike, but
            the swap is priced off the whole smile, and when fear rises the downside strikes move
            more than the middle.
          </p>
          {marketRisk.autocallMaxGap && (
            <p>
              <strong className="text-ink">The autocallable note</strong>: the shortcut captures{" "}
              {pct(marketRisk.autocallCapturePct)} on average, but not on the largest moves, where
              its payoff bends. Its worst replay for the shortcut is{" "}
              {fmtDate(marketRisk.autocallMaxGap.date)}: the note would make{" "}
              {signedUsdCompact(marketRisk.autocallMaxGap.full)}, the shortcut says{" "}
              {signedUsdCompact(marketRisk.autocallMaxGap.taylor)}.
            </p>
          )}
        </div>
      </Card>

      <Card
        title="Value at Risk, day by day"
        subtitle={`The 1-day 99% figure for each of the last ${marketRisk.seriesDays} stored days, both ways. The gap between the lines is the shortcut's blind spot.`}
      >
        <VarHistory data={varData.series} />
      </Card>

      <Card
        title="Where the risk comes from"
        subtitle="Expected Shortfall over 10 days at 97.5%, the regulatory standard, with only some market factors allowed to move."
      >
        <RankedBars data={factorBars} highlight="Everything moves" format="usdCompact" />
        <div className="mt-5 flex flex-col gap-3 text-sm leading-relaxed text-ink-2">
          <p>
            Volatility moves alone cost more than everything moving together (
            {usdCompact(varData.es10.vol)} against {usdCompact(varData.es10.all)}): on the
            worst volatility days, the stock moves that came with them partly offset the loss.
          </p>
          <p>
            Regulators also ask how long each risk would take to exit. Allowing 20 days for
            single-stock volatility and 60 for the structured notes raises the figure from{" "}
            {usdCompact(varData.es10.all)} to {usdCompact(varData.es10_lh_adjusted)}.
          </p>
          <p>
            Ten-day losses come from real 10-day moves. Scaling the 1-day figure by &radic;10, a
            common shortcut, gives {usdCompact(marketRisk.var10Sqrt)}, {pct(marketRisk.sqrt10OverPct)}{" "}
            more: the worst single days were partly reversed within two weeks, which that rule
            assumes never happens.
          </p>
        </div>
      </Card>

      <Card
        title="Turning a past day into today's market"
        subtitle={`The engine has stored a full volatility surface only since August 2026, not ${varData.window_days} days. For each past day it uses the published volatility index of each name, which goes back decades, and maps it onto today's surface point by point.`}
      >
        <div className="overflow-x-auto">
          <table className="tabular w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-muted">
                <th className="py-2 pr-4 font-medium">Underlying</th>
                <th className="py-2 pr-4 font-medium">Volatility index</th>
                <th className="py-2 pr-4 text-right font-medium">Stored days fitted</th>
                <th className="py-2 pr-4 text-right font-medium">1-month sensitivity</th>
                <th className="py-2 text-right font-medium">Share of moves explained</th>
              </tr>
            </thead>
            <tbody>
              {marketRisk.proxyNames.map((n) => (
                <tr key={n.underlying} className="border-b border-border/60">
                  <td className="py-2 pr-4 text-ink">{n.label}</td>
                  <td className="py-2 pr-4">{n.index}</td>
                  <td className="py-2 pr-4 text-right">{n.n_changes}</td>
                  <td className="py-2 pr-4 text-right">
                    {n.atm_1m_beta === null ? "n/a" : n.atm_1m_beta.toFixed(2)}
                  </td>
                  <td className="py-2 text-right">{pct(n.r2_median * 100)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-muted">
            1-month sensitivity: at-the-money vol points per index point. Share explained:
            median R&sup2; across the surface points used. NVIDIA has no
            volatility index of its own; the Nasdaq-100&apos;s (VXN) stands in.
          </p>
        </div>
        <p className="mt-5 text-sm leading-relaxed text-ink-2">
          What the index leaves out is measured, not ignored: replaying the {varData.proxy.n_days}{" "}
          stored days both with their real surface moves and with the index-based ones, the
          index version misses {usdCompact(varData.proxy.rms_error)} a day on average (root mean
          square), against a typical daily swing of {usdCompact(varData.proxy.true_sd)}. Nearly
          all of it is the variance swap again ({usdCompact(varData.proxy.varswap_rms_error)}).
        </p>
      </Card>

      <Methodology>
        <p>
          <strong className="text-ink">Scenarios.</strong> Historical simulation over{" "}
          {varData.window_days} trading days, also run over{" "}
          {varData.windows.map((w) => w.window).join(", ")} days for comparison (
          {varData.windows.map((w) => `${w.window}: ${usdCompact(w.var_full)}`).join(", ")}). Each
          scenario applies one past day&apos;s stock returns, and its volatility-index change
          rescaled to today&apos;s index level, to today&apos;s market. Rates and dividends stay
          at today&apos;s values.
        </p>
        <p>
          <strong className="text-ink">Volatility moves.</strong> Each surface point&apos;s daily
          move is regressed on its index&apos;s move over the stored days, using only points
          inside the quoted strikes on at least 80% of days. A past index move times those
          sensitivities gives that day&apos;s move at every point.
        </p>
        <p>
          <strong className="text-ink">Two methods.</strong> Full revaluation reprices every
          position, the simulated notes with the same random paths in every scenario so
          simulation noise cancels. The shortcut uses delta, gamma, vega, vanna and volga from the
          day&apos;s marks. Value at Risk is the 5th worst of 500 outcomes; Expected Shortfall the
          average of the worst.
        </p>
        <p>
          <strong className="text-ink">Not yet.</strong> Backtesting these figures against the
          profit and loss that followed needs about 250 days of daily Value at Risk; there are{" "}
          {marketRisk.seriesDays} so far.
        </p>
      </Methodology>

      <DemoNote />
    </div>
  );
}
