import type { Metadata } from "next";
import { AlertHistory } from "@/components/charts/AlertHistory";
import { GroupedBars } from "@/components/charts/GroupedBars";
import { RankedBars } from "@/components/charts/RankedBars";
import { Card, DemoNote, Methodology, NextLink, PageHeader, StatTile, Takeaway } from "@/components/ui";
import { SERIES } from "@/lib/chartTheme";
import { fmtDate, fmtShortDate, pct, signedUsdCompact, usdCompact } from "@/lib/format";
import { DRIVER_LABELS, DRIVER_SHORT_LABELS, pnl, pnlExplain } from "@/lib/metrics";

export const metadata: Metadata = {
  title: "Daily P&L Explain — EQD Risk Engine",
  description:
    "See what made or lost money each day, driver by driver, and which position is behind anything the risk model can't explain.",
};

const TOP_N = 5;

const drivers = pnlExplain.waterfall.map((s) => ({
  driver: DRIVER_LABELS[s.step],
  short: DRIVER_SHORT_LABELS[s.step],
  expected: s.explained_pnl,
  actual: s.actual_pnl,
}));

const ranked = pnlExplain.by_position.map((p) => ({
  label: pnlExplain.positions[p.position_id] ?? p.position_id,
  value: p.residual,
}));
const rest = ranked.slice(TOP_N);
const positionBars = [
  ...ranked.slice(0, TOP_N),
  ...(rest.length
    ? [{ label: `Other positions (${rest.length})`, value: rest.reduce((a, b) => a + b.value, 0) }]
    : []),
];

const history = pnlExplain.residual_series.map((p) => ({
  date: p.asof_date,
  label: fmtShortDate(p.asof_date),
  pct: p.residual_bp / 100,
}));

// The driver with the largest gap on the featured day, stated from the data.
const biggestGapDriver = [...pnlExplain.waterfall].sort(
  (a, b) => Math.abs(b.residual) - Math.abs(a.residual),
)[0];

export default function PnlExplainPage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        section="Daily P&L Explain"
        title="Where the day's P&L came from"
        lede="Each evening the engine splits the book's profit and loss into its drivers and checks the total against what the risk model predicted. Anything it can't account for is flagged and traced to the position behind it."
        meta={`Case study: ${fmtDate(pnlExplain.day0)} to ${fmtDate(pnlExplain.asof_date)}`}
      />

      <Takeaway>
        On {fmtDate(pnlExplain.asof_date)} the engine flagged{" "}
        <strong>{usdCompact(pnl.thenUnexplained)}</strong> of P&amp;L it could not explain,{" "}
        {pct(pnl.thenUnexplainedPctOfBook, 1)} of book value, {pct(pnl.thenTopPositionShare)} of it
        from one position: the {pnl.topPositionLabel}. The investigation traced the gap to the
        engine&apos;s own inputs and model, not the market. After {pnl.findings.length} fixes and a rebuild of the
        full history, the same day shows a gap of <strong>{usdCompact(Math.abs(pnl.unexplained))}</strong>{" "}
        ({pct(Math.abs(pnl.unexplainedPctOfBook), 2)} of book), and the average daily gap is{" "}
        {pct(pnl.meanGapReduction)} smaller.
      </Takeaway>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Unexplained, as first flagged"
          value={usdCompact(pnl.thenUnexplained)}
          detail={`${pct(pnl.thenUnexplainedPctOfBook, 1)} of book value`}
          tone="critical"
        />
        <StatTile
          label="Same day, after the fixes"
          value={usdCompact(Math.abs(pnl.unexplained))}
          detail={`${pct(Math.abs(pnl.unexplainedPctOfBook), 2)} of book value`}
        />
        <StatTile
          label="Average daily gap"
          value={`${pct(pnl.meanGapBeforePct, 2)} to ${pct(pnl.meanGapAfterPct, 2)}`}
          detail="Of book value, on the same 16 trading days before and after"
        />
        <StatTile
          label="Days still alerting"
          value={`${pnl.daysBreached} of ${pnl.daysRecorded}`}
          detail={`${pnl.daysInconclusive} within simulation noise, ${pnl.daysWithinThreshold} clear`}
        />
      </div>

      <Card
        title="What the investigation found"
        subtitle={`The first explanation, missing risk sensitivities, did not hold up when tested on six weeks of history. These ${pnl.findings.length} causes did, and each was fixed and measured on every stored day.`}
      >
        <div className="grid gap-4 sm:grid-cols-2">
          {pnl.findings.map((f) => (
            <div key={f.title} className="rounded-xl bg-subtle px-4 py-3">
              <p className="font-medium text-ink">{f.title}</p>
              <p className="mt-1 text-sm text-ink-2">{f.body}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card
        title="Expected vs. actual, by driver, after the fixes"
        subtitle={`On the case-study day the model expected ${signedUsdCompact(pnl.expected)} and the book made ${signedUsdCompact(pnl.actual)}. The largest remaining difference is in ${DRIVER_LABELS[biggestGapDriver.step].toLowerCase()} (${signedUsdCompact(biggestGapDriver.residual)}).`}
      >
        <GroupedBars
          data={drivers}
          categoryKey="driver"
          shortCategoryKey="short"
          series={[
            { key: "expected", label: "Expected by risk model", color: SERIES[0] },
            { key: "actual", label: "Actual", color: SERIES[1] },
          ]}
          format="usdCompact"
        />
      </Card>

      <Card
        title="Unexplained P&L by position, after the fixes"
        subtitle={`The ${pnl.topPositionLabel} is still the largest contributor; the other ${ranked.length - 1} positions together make up ${pct(Math.max(0, 100 - pnl.topPositionShare))}.`}
      >
        <RankedBars data={positionBars} highlight={pnl.topPositionLabel} format="usdCompact" />
      </Card>

      <Card
        title="Alert history"
        subtitle={`Unexplained P&L as a share of book value, one bar per trading day, after the fixes. A day alerts only when the gap is wider than the ±${pct(pnl.alertPctOfBook, 2)} alert level even after allowing for simulation noise: ${pnl.daysBreached} of ${pnl.daysRecorded} days alert, ${pnl.daysInconclusive} are within noise and ${pnl.daysWithinThreshold} are clear. The largest remaining gaps come from the structured note on days with big NVIDIA moves or changes in the shape of its volatility curve, a known limit of sensitivity-based explanations for this product.`}
      >
        <AlertHistory data={history} highlightDate={pnlExplain.asof_date} />
        <p className="mt-3 text-xs text-muted">
          Days where either day&apos;s valuation is missing (market holidays, missed runs, lost data)
          are left out rather than shown as zero.
        </p>
      </Card>

      <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-border bg-surface px-6 py-5">
        <p className="text-ink-2">What did the AI Risk Analyst make of this day?</p>
        <NextLink href="/ai-agent">Read its investigation</NextLink>
      </div>

      <Methodology>
        <p>
          <strong className="text-ink">Attribution.</strong> Explained P&amp;L is a Greeks-based
          Taylor expansion applied sequentially: time (theta), rates and dividends (rho, dividend
          rho), spot (delta, gamma) and volatility (vega, measured where the position&apos;s
          exposure sits). Actual P&amp;L is full revaluation on each day&apos;s calibrated market.
          The residual is actual minus explained.
        </p>
        <p>
          <strong className="text-ink">Alert level and noise.</strong>{" "}
          {pnlExplain.residual_alert_threshold_bp}bp of NAV per day. Monte Carlo-priced positions
          carry a standard error from 16 independently scrambled sub-runs; a day alerts only if the
          residual exceeds the threshold by more than two standard errors, and is inconclusive when
          the noise band alone is wider than the threshold.
        </p>
        <p>
          <strong className="text-ink">What changed.</strong> Better-conditioned Greeks for the
          Monte Carlo exotics; single-name forwards fitted with the rates curve&apos;s discount
          factor and one reliability rule for surface and pricing; flat implied vol and continued
          carry past the last expiry; short gaps in long-dated quotes filled from the previous day;
          a fixed local-vol smoothing strength; observation dates simulated exactly; and the structured note
          priced against its issue-date fixing, with expiries settled at the expiry close and expired
          options rolled to the next quarter. The full
          write-up, including the original diagnosis that turned out to be wrong, is in the
          incident report in the code repository.
        </p>
        <p>
          <strong className="text-ink">The figures as first flagged</strong> come from the original
          run on that day; the stored history has since been recomputed with the fixes, so they are
          kept as fixed values.
        </p>
      </Methodology>

      <DemoNote />
    </div>
  );
}
