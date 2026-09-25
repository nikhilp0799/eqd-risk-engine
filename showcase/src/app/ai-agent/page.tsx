import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Badge, Card, DemoNote, Methodology, NextLink, PageHeader, StatTile, Takeaway } from "@/components/ui";
import { fmtDate, pct, usdCompact } from "@/lib/format";
import { ai, aiAgent, pnl } from "@/lib/metrics";

export const metadata: Metadata = {
  title: "AI Risk Analyst — EQD Risk Engine",
  description:
    "An AI analyst that investigates unexplained P&L by running its own what-if tests on the pricing engine, then recommends a fix for a person to approve.",
};

function prettyJson(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <li className="relative flex gap-5 pb-8 last:pb-0">
      <div className="flex flex-col items-center">
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-brand text-sm font-semibold text-white">
          {n}
        </span>
        <span className="mt-2 w-px flex-1 bg-border" aria-hidden />
      </div>
      <div className="flex-1 pt-1">
        <h3 className="font-semibold text-ink">{title}</h3>
        <div className="mt-2 text-sm leading-relaxed text-ink-2">{children}</div>
      </div>
    </li>
  );
}

const r = ai.reprice;
const confidenceTone = aiAgent.confidence === "high" ? "good" : "neutral";

export default function AiAgentPage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        section="AI Risk Analyst"
        title="An analyst that tests its own hunch"
        lede="When the daily P&L check flags a gap, an AI analyst investigates it. It can ask the pricing engine to run what-if tests, then writes up the likely cause and a recommendation. It runs privately on the firm's own hardware and never changes anything by itself."
        meta={`Featured investigation: ${fmtDate(aiAgent.day0)} to ${fmtDate(aiAgent.asof_date)}`}
      />

      {r && (
        <Takeaway>
          Given the <strong>{usdCompact(pnl.unexplained)}</strong> gap, the analyst didn&apos;t just
          summarise the numbers. It ran its own stress test on the {r.positionLabel}: NVIDIA down{" "}
          {pct(-r.spotShockPct)} and volatility up by a quarter. That took{" "}
          <strong>{usdCompact(-r.priceDelta)}</strong> ({pct(-r.priceDeltaPct, 1)}) off the
          note&apos;s value, showing how sharply this position reacts to large moves. It then wrote up
          the likely cause and a recommendation for a person to review.
        </Takeaway>
      )}

      <div className="grid gap-4 sm:grid-cols-3">
        <StatTile
          label="Pricing-engine tests it chose to run"
          value={String(ai.toolCalls)}
          detail="Real calls to the valuation engine, not a canned script."
        />
        <StatTile
          label="Position it flagged"
          value={aiAgent.flagged_positions[0]?.position_id ?? "None"}
          detail={pnl.topPositionLabel}
        />
        <StatTile
          label="Where it runs"
          value="On-site"
          detail="An open-weight model on local hardware. No data leaves the firm; no per-query fees."
        />
      </div>

      <Card title="How the investigation went">
        <ol>
          <Step n={1} title="Spotted the outlier">
            Of the day&apos;s {usdCompact(pnl.unexplained)} unexplained P&amp;L, the largest share
            ({pct(pnl.topPositionShare)}) sat in the {pnl.topPositionLabel}. The analyst picked it
            for a closer look.
          </Step>
          {r && (
            <Step n={2} title="Ran its own what-if test">
              <p>
                It asked the pricing engine: what is this note worth if NVIDIA falls{" "}
                {pct(-r.spotShockPct)} and volatility rises by a quarter?
              </p>
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <div className="rounded-xl bg-subtle px-4 py-3">
                  <p className="text-xs text-muted">Value today</p>
                  <p className="mt-1 text-lg font-semibold text-ink">{usdCompact(r.basePrice)}</p>
                </div>
                <div className="rounded-xl bg-subtle px-4 py-3">
                  <p className="text-xs text-muted">Value in that scenario</p>
                  <p className="mt-1 text-lg font-semibold text-ink">
                    {usdCompact(r.basePrice + r.priceDelta)}
                  </p>
                </div>
                <div className="rounded-xl bg-critical-soft px-4 py-3">
                  <p className="text-xs text-critical">Change</p>
                  <p className="mt-1 text-lg font-semibold text-critical">
                    {usdCompact(r.priceDelta)} ({pct(r.priceDeltaPct, 1)})
                  </p>
                </div>
              </div>
            </Step>
          )}
          <Step n={3} title="Wrote up the likely cause">
            <blockquote className="rounded-xl border-l-4 border-brand bg-subtle px-4 py-3 text-ink">
              {aiAgent.root_cause_hypothesis}
            </blockquote>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Badge tone={confidenceTone}>AI confidence: {aiAgent.confidence}</Badge>
              <span className="text-xs text-muted">Written by the AI analyst, not independently verified.</span>
            </div>
          </Step>
          <Step n={4} title="Recommended a change, for a person to decide">
            {aiAgent.proposed_changes.map((c) => (
              <div key={c.parameter} className="rounded-xl border border-border px-4 py-3">
                <p className="font-medium capitalize text-ink">{c.parameter}</p>
                <p className="mt-1">
                  Today: {c.current_value}. Suggested: {c.suggested_value}.
                </p>
                <p className="mt-1 text-muted">{c.rationale}</p>
              </div>
            ))}
            <div className="mt-2">
              <Badge>Sent for review, not applied</Badge>
            </div>
          </Step>
        </ol>
      </Card>

      <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-border bg-surface px-6 py-5">
        <p className="text-ink-2">See the P&amp;L breakdown the analyst was working from.</p>
        <NextLink href="/pnl-explain">Open Daily P&amp;L Explain</NextLink>
      </div>

      <Methodology>
        <p>
          <strong className="text-ink">Model.</strong> {aiAgent.model}, an open-weight model served
          locally through Ollama. It receives the day&apos;s P&amp;L-explain output and can call
          tools through native function calling, up to three rounds per investigation: a
          what-if reprice of any position under a spot and relative-volatility shock (bounded
          inputs), and reads of the model documentation.
        </p>
        <p>
          <strong className="text-ink">Guardrails.</strong> Output is parsed into a fixed structure
          (summary, hypothesis, confidence, flagged positions, proposed changes) and stored with the
          full tool-call trace. Proposed changes are never applied automatically.
        </p>
        <p>
          <strong className="text-ink">Full summary as written by the model:</strong> {aiAgent.summary}
        </p>
        {aiAgent.trace.map((t) => (
          <div key={t.round} className="grid gap-3 sm:grid-cols-2">
            <div>
              <p className="mb-1 text-xs font-medium text-muted">
                Tool call {t.round}: {t.tool_name}, arguments
              </p>
              <pre className="overflow-x-auto rounded-lg border border-border bg-subtle p-3 text-xs text-ink">
                {prettyJson(t.arguments)}
              </pre>
            </div>
            <div>
              <p className="mb-1 text-xs font-medium text-muted">Returned by the pricing engine</p>
              <pre className="overflow-x-auto rounded-lg border border-border bg-subtle p-3 text-xs text-ink">
                {prettyJson(t.result_summary)}
              </pre>
            </div>
          </div>
        ))}
      </Methodology>

      <DemoNote />
    </div>
  );
}
