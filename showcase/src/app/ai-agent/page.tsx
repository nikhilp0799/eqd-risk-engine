import type { Metadata } from "next";
import { Panel, Pill } from "@/components/Panel";
import { fmtDate } from "@/lib/format";
import aiAgent from "@/data/ai_agent.json";
import type { AiAgentData } from "@/lib/types";

export const metadata: Metadata = {
  title: "AI Investigation Agent — EQD Risk Engine",
  description:
    "A local open-weight model with real tool access to the pricing engine — a genuine agentic tool call, not a claimed capability, investigating a real P&L residual.",
};

const data = aiAgent as AiAgentData;

function prettyJson(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

const confidenceTone =
  data.confidence === "high" ? "positive" : data.confidence === "low" ? "negative" : "neutral";

export default function AiAgentPage() {
  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-3">
        <p className="font-num text-sm text-accent">
          ai-agent / {fmtDate(data.day0)} &rarr; {fmtDate(data.asof_date)}
        </p>
        <h1 className="text-3xl font-semibold text-foreground">AI investigation agent</h1>
        <p className="text-muted max-w-2xl">
          A local open-weight model (<span className="font-num">{data.model}</span>, via Ollama —
          zero API cost, runs entirely on-machine) investigates the daily P&amp;L-explain
          residual with real tool access to this project&apos;s own pricing engine. This is the
          real day it caught a genuine attribution gap.
        </p>
        <div>
          <Pill tone={confidenceTone as "positive" | "negative" | "neutral"}>
            confidence: {data.confidence}
          </Pill>
        </div>
      </header>

      <Panel title="What the model concluded">
        <div className="flex flex-col gap-4 text-sm text-foreground/90 leading-relaxed">
          <p>{data.summary}</p>
          <p className="text-muted">
            <span className="text-foreground font-medium">Root cause hypothesis: </span>
            {data.root_cause_hypothesis}
          </p>
        </div>
      </Panel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Flagged positions">
          <table className="w-full text-sm">
            <tbody>
              {data.flagged_positions.map((p) => (
                <tr key={p.position_id} className="border-b border-panel-border/50 last:border-0">
                  <td className="py-2 pr-4 font-num text-accent align-top">{p.position_id}</td>
                  <td className="py-2 text-muted">{p.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel
          title="Proposed changes"
          caption="Suggestions for a human to review — never auto-applied."
        >
          <div className="flex flex-col gap-4">
            {data.proposed_changes.map((c, i) => (
              <div key={i} className="text-sm">
                <p className="font-num text-foreground">{c.parameter}</p>
                <p className="text-muted">
                  <span className="font-num">{c.current_value}</span> &rarr;{" "}
                  <span className="font-num text-accent">{c.suggested_value}</span>
                </p>
                <p className="text-muted mt-1">{c.rationale}</p>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <Panel
        title="Investigation trace"
        caption="What the model actually called mid-investigation — real pricing-engine reprices, not a claimed capability."
      >
        <div className="flex flex-col gap-5">
          {data.trace.map((t, i) => (
            <div key={i} className="border-l-2 border-accent/40 pl-4">
              <p className="font-num text-sm text-accent mb-2">
                round {t.round} &middot; {t.tool_name}
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted mb-1">arguments</p>
                  <pre className="font-num text-xs bg-background/60 rounded-lg p-3 overflow-x-auto border border-panel-border">
                    {prettyJson(t.arguments)}
                  </pre>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted mb-1">result</p>
                  <pre className="font-num text-xs bg-background/60 rounded-lg p-3 overflow-x-auto border border-panel-border">
                    {prettyJson(t.result_summary)}
                  </pre>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
