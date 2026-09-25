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
const vol = pnlExplain.waterfall.find((s) => s.step === "vol");

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

export default function PnlExplainPage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        section="Daily P&L Explain"
        title="Where the day's P&L came from"
        lede="Each evening the engine splits the book's profit and loss into its drivers and checks the total against what the risk model predicted. Anything it can't account for is flagged and traced to the position behind it."
        meta={`Featured day: ${fmtDate(pnlExplain.day0)} to ${fmtDate(pnlExplain.asof_date)}`}
      />

      <Takeaway>
        The book made <strong>{signedUsdCompact(pnl.actual)}</strong>, but the risk model expected{" "}
        <strong>{signedUsdCompact(pnl.expected)}</strong>. That <strong>{usdCompact(pnl.unexplained)}</strong>{" "}
        gap is {pct(pnl.unexplainedPctOfBook, 1)} of book value, far above the{" "}
        {pct(pnl.alertPctOfBook, 2)} alert level, and <strong>{pct(pnl.topPositionShare)}</strong> of
        it came from one position: the {pnl.topPositionLabel}. The engine flagged it the same evening
        and handed it to the AI Risk Analyst.
      </Takeaway>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Actual P&L" value={signedUsdCompact(pnl.actual)} />
        <StatTile label="Expected by risk model" value={signedUsdCompact(pnl.expected)} />
        <StatTile
          label="Unexplained"
          value={usdCompact(pnl.unexplained)}
          detail={`${pct(pnl.unexplainedPctOfBook, 1)} of book value`}
          tone="critical"
        />
        <StatTile
          label="Traced to one position"
          value={pct(pnl.topPositionShare)}
          detail={pnl.topPositionLabel}
        />
      </div>

      <Card
        title="Expected vs. actual, by driver"
        subtitle={
          vol
            ? `Almost all of the gap sits in volatility: the model expected the volatility move to cost ${usdCompact(-vol.explained_pnl)}, while the book actually gained ${usdCompact(vol.actual_pnl)} from it.`
            : undefined
        }
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
        title="Unexplained P&L by position"
        subtitle={`One position accounts for nearly all of it. The other ${ranked.length - 1} positions together make up ${pct(100 - pnl.topPositionShare)}.`}
      >
        <RankedBars data={positionBars} highlight={pnl.topPositionLabel} format="usdCompact" />
      </Card>

      <Card
        title="Alert history"
        subtitle={`Unexplained P&L as a share of book value, one bar per trading day. The alert level is ±${pct(pnl.alertPctOfBook, 2)}, too small to see at this scale, and every bar is well past it: the alert fired on ${pnl.daysBreached} of ${pnl.daysRecorded} recorded days, and on every day with a position breakdown (${pnl.daysWithBreakdown} of them) the same note was the biggest contributor. That points to a known modelling limit for this product type, not random noise, and it stays flagged until it is fixed.`}
      >
        <AlertHistory data={history} highlightDate={pnlExplain.asof_date} />
        <p className="mt-3 text-xs text-muted">
          Days where either day&apos;s valuation is missing (market holidays, missed runs) are left out
          rather than shown as zero.
        </p>
      </Card>

      <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-border bg-surface px-6 py-5">
        <p className="text-ink-2">
          What did the AI Risk Analyst make of this gap?
        </p>
        <NextLink href="/ai-agent">Read its investigation</NextLink>
      </div>

      <Methodology>
        <p>
          <strong className="text-ink">Attribution.</strong> Explained P&amp;L is a Greeks-based
          Taylor expansion applied sequentially: time (theta), rates and dividends (rho, dividend
          rho), spot (delta, gamma) and volatility (vega, plus vanna and volga for Monte Carlo-priced
          exotics). Actual P&amp;L is full revaluation on each day&apos;s calibrated market. The
          residual is actual minus explained.
        </p>
        <p>
          <strong className="text-ink">Alert level.</strong> {pnlExplain.residual_alert_threshold_bp}bp
          of NAV per day, applied to the book total.
        </p>
        <p>
          <strong className="text-ink">Why the autocallable note dominates.</strong> The note is
          priced by local-volatility Monte Carlo and its Greeks come from bump-and-revalue. Its value
          is path-dependent and strongly non-linear in spot and vol near its barriers, so a
          second-order Greek expansion misses a large part of a big daily move. Adding vanna and volga
          cut the residual by about 13% but did not close it; a full-revaluation step for this
          position is the fix.
        </p>
        <p>
          <strong className="text-ink">Data handling.</strong> Early days predate the stored NAV
          column; for those, NAV is the day&apos;s stored book value from the portfolio marks. Day
          pairs without marks on both sides are excluded.
        </p>
      </Methodology>

      <DemoNote />
    </div>
  );
}
