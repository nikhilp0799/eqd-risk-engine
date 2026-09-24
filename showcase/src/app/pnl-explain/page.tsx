import Link from "next/link";
import type { Metadata } from "next";
import { Panel, Pill } from "@/components/Panel";
import { WaterfallChart } from "@/components/charts/WaterfallChart";
import { ResidualSeriesChart } from "@/components/charts/ResidualSeriesChart";
import { ByPositionChart } from "@/components/charts/ByPositionChart";
import { fmtBp, fmtDate, fmtUsd } from "@/lib/format";
import pnlExplain from "@/data/pnl_explain.json";
import type { PnlExplainData } from "@/lib/types";

export const metadata: Metadata = {
  title: "P&L Explain — EQD Risk Engine",
  description:
    "A Greeks-based P&L attribution waterfall that traced 99.6% of a real 670bp residual to a single mispriced exotic position.",
};

const data = pnlExplain as PnlExplainData;
const topResidual = data.by_position[0];
const breach = Math.abs(data.residual_bp) > data.residual_alert_threshold_bp;

export default function PnlExplainPage() {
  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-3">
        <p className="font-num text-sm text-accent">
          pnl-explain / {fmtDate(data.day0)} &rarr; {fmtDate(data.asof_date)}
        </p>
        <h1 className="text-3xl font-semibold text-foreground">P&amp;L explain</h1>
        <p className="text-muted max-w-2xl">
          A Greeks-based attribution waterfall — time decay, rates/dividends, spot, and vol —
          decomposes the book&apos;s real day-over-day P&amp;L. What is left over (the residual)
          is the part the Greeks alone don&apos;t explain, monitored against a{" "}
          {data.residual_alert_threshold_bp}bp-of-NAV alert threshold below.
        </p>
        <div className="flex gap-2">
          <Pill tone={breach ? "negative" : "positive"}>
            {fmtBp(data.residual_bp)} of NAV {breach ? "— threshold breached" : ""}
          </Pill>
        </div>
      </header>

      <Panel
        title={`Waterfall — ${fmtDate(data.day0)} to ${fmtDate(data.asof_date)}`}
        caption={`NAV ${fmtUsd(data.nav)}`}
      >
        <WaterfallChart steps={data.waterfall} />
      </Panel>

      <Panel
        title="Residual by position"
        caption={`${topResidual?.position_id ?? "—"} accounts for the large majority of this day's residual — traced, not hidden.`}
      >
        <ByPositionChart positions={data.by_position} />
      </Panel>

      <Panel
        title="Residual over time"
        caption="Basis points of NAV, with the alert threshold — most days sit well inside the band; this incident day is the visible exception."
      >
        <ResidualSeriesChart
          series={data.residual_series}
          thresholdBp={data.residual_alert_threshold_bp}
        />
      </Panel>

      <p className="text-sm text-muted">
        This is the same real incident investigated by the{" "}
        <Link href="/ai-agent" className="text-accent hover:underline">
          AI investigation agent
        </Link>{" "}
        — read what it found.
      </p>
    </div>
  );
}
