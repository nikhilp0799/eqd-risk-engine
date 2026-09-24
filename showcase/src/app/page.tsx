import Link from "next/link";
import { StatCard } from "@/components/Panel";
import overview from "@/data/overview.json";
import type { OverviewData } from "@/lib/types";

const data = overview as OverviewData;

const DETAIL_PAGES = [
  {
    href: "/deep-hedging",
    title: "Deep Hedging",
    teaser:
      "PyTorch hedging policies trained against a calibrated local-vol Monte Carlo simulator, benchmarked out-of-sample against classical Greeks — across a vanilla, an autocallable, and a barrier option.",
  },
  {
    href: "/ai-agent",
    title: "AI Investigation Agent",
    teaser:
      "A local LLM that reaches for the pricing engine mid-investigation — a real tool call, real returned Greeks, used in its own written hypothesis.",
  },
  {
    href: "/vol-surface",
    title: "Volatility Surface & Greeks",
    teaser:
      "SVI/SSVI calibration on real SPX option chains, arbitrage-checked, driving a full closed-form Greek set validated against QuantLib.",
  },
  {
    href: "/pnl-explain",
    title: "P&L Explain",
    teaser:
      "A Greeks-based attribution waterfall that traced 99.6% of a real 670bp residual to a single mispriced exotic position — and said so.",
  },
];

export default function Home() {
  return (
    <div className="flex flex-col gap-14">
      <section className="flex flex-col gap-5 pt-6">
        <p className="font-num text-sm text-accent">$ eqdrisk --status</p>
        <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight text-foreground max-w-3xl">
          An equity derivatives risk engine, built from scratch.
        </h1>
        <p className="text-lg text-muted max-w-2xl">
          Vol surface calibration, exotics pricing, deep-hedging neural policies, and an
          agentic AI investigator — every number on this site is read from a real,
          reproducible pipeline run, not illustrative. No live backend: this is a static
          snapshot of real curated output.
        </p>
      </section>

      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {data.stats.map((stat) => (
          <StatCard key={stat.label} {...stat} />
        ))}
      </section>

      <section className="flex flex-col gap-4">
        <h2 className="text-sm uppercase tracking-wide text-muted">Explore</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {DETAIL_PAGES.map((page) => (
            <Link
              key={page.href}
              href={page.href}
              className="group rounded-xl border border-panel-border bg-panel/60 p-6 flex flex-col gap-2 transition-colors hover:border-accent"
            >
              <h3 className="text-lg font-semibold text-foreground group-hover:text-accent transition-colors">
                {page.title} <span className="font-num">&rarr;</span>
              </h3>
              <p className="text-sm text-muted">{page.teaser}</p>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
