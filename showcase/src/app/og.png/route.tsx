import { ImageResponse } from "next/og";
import { hedging, pnl, pricing } from "@/lib/metrics";

// The link-preview card (LinkedIn, X, Slack), served as /og.png. A route handler
// rather than the opengraph-image convention: the static export names the file
// after this folder, so hosts serve it as a PNG by its extension. Built once at
// export time from the same data as the pages, so its figures cannot drift.
export const dynamic = "force-static";

const size = { width: 1200, height: 630 };

const INK = "#0e1a2b";
const INK_2 = "#475467";
const MUTED = "#7a8699";
const BRAND = "#1c5cab";
const BORDER = "#e3e6eb";
const BG = "#f5f7fa";

const stats = [
  { value: `${Math.round(pnl.meanGapReduction)}%`, label: "smaller daily P&L gap after the engine fixed its own errors" },
  { value: `${Math.round(hedging.bestBarrierSwingReduction)}%`, label: "smaller P&L swings from AI-trained hedging" },
  { value: pricing.quotes.toLocaleString("en-US"), label: "live S&P 500 quotes in one pricing surface" },
  { value: "8 decimals", label: "agreement with QuantLib on option prices" },
];

export function GET() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: BG,
          padding: "56px 64px",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 52,
              height: 52,
              borderRadius: 12,
              background: BRAND,
              color: "white",
              fontSize: 22,
              fontWeight: 700,
            }}
          >
            EQ
          </div>
          <div style={{ display: "flex", fontSize: 30, fontWeight: 600, color: INK }}>EQD Risk Engine</div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ display: "flex", fontSize: 24, color: BRAND }}>
            Risk and P&amp;L intelligence for equity derivatives
          </div>
          <div style={{ display: "flex", fontSize: 64, fontWeight: 700, color: INK, lineHeight: 1.1 }}>
            Know why your P&amp;L moved, before the desk asks.
          </div>
        </div>

        <div style={{ display: "flex", gap: 20 }}>
          {stats.map((s) => (
            <div
              key={s.value}
              style={{
                display: "flex",
                flexDirection: "column",
                flex: 1,
                gap: 8,
                background: "white",
                border: `1px solid ${BORDER}`,
                borderRadius: 16,
                padding: "20px 22px",
              }}
            >
              <div style={{ display: "flex", fontSize: 40, fontWeight: 700, color: INK }}>{s.value}</div>
              <div style={{ display: "flex", fontSize: 18, color: INK_2, lineHeight: 1.3 }}>{s.label}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", fontSize: 18, color: MUTED }}>
          Demo book priced on real market data. Figures from actual daily runs.
        </div>
      </div>
    ),
    size,
  );
}
