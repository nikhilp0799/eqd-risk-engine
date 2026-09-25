import Link from "next/link";
import { Badge, DemoNote, StatTile } from "@/components/ui";
import { fmtDate, pct, signedUsdCompact, usdCompact } from "@/lib/format";
import { hedging, pnl, pnlExplain, pricing } from "@/lib/metrics";

// Not derived from exported data: the repo's own test count at the time of the
// last site update (`pytest --collect-only`).
const TEST_COUNT = "1,778";

const CAPABILITIES = [
  {
    href: "/pnl-explain",
    title: "Daily P&L Explain",
    body: "Every morning, see what made or lost money, split into time decay, rates, stock moves and volatility. Anything the numbers can't account for is flagged.",
  },
  {
    href: "/ai-agent",
    title: "AI Risk Analyst",
    body: "When a number doesn't add up, an AI analyst investigates. It runs its own stress tests on the suspect position and writes up the likely cause. It recommends; people decide.",
  },
  {
    href: "/deep-hedging",
    title: "Hedging Strategy Lab",
    body: "Hedging strategies trained on thousands of simulated markets, compared head-to-head with the benchmark hedge before anyone trades on them.",
  },
  {
    href: "/vol-surface",
    title: "Market-Consistent Pricing",
    body: "Every price starts from the market. Live option quotes become one consistent pricing surface, the foundation for every valuation and risk number.",
  },
];

const STEPS = [
  {
    title: "Collect",
    body: "Option prices, interest rates and dividends for every underlying, captured at the market close.",
  },
  {
    title: "Price",
    body: "The engine rebuilds the market's volatility surface and reprices every position, from plain options to autocallable notes.",
  },
  {
    title: "Explain and alert",
    body: `It attributes the day's P&L, flags any gap above ${pct(pnl.alertPctOfBook, 2)} of book value, and hands the gap to the AI analyst.`,
  },
];

const TRUST = [
  {
    title: "Benchmarked",
    body: "Option prices agree with QuantLib, the industry's open-source benchmark library, to eight decimal places.",
  },
  { title: "Tested", body: `${TEST_COUNT} automated tests run on every change.` },
  {
    title: "Private by design",
    body: "The AI analyst runs on the firm's own hardware. No position data is sent to an outside AI service.",
  },
  {
    title: "People stay in control",
    body: "The AI never changes a model or a trade by itself. Every suggestion goes to a person to approve.",
  },
];

function IncidentPreview() {
  return (
    <div className="rounded-2xl border border-border bg-surface p-6 shadow-[0_12px_40px_rgba(14,26,43,0.08)]">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-ink">Daily P&amp;L check</p>
          <p className="text-xs text-muted">{fmtDate(pnlExplain.asof_date)}</p>
        </div>
        <Badge tone="critical">Alert</Badge>
      </div>
      <dl className="mt-5 divide-y divide-border text-sm">
        <div className="flex justify-between py-2.5">
          <dt className="text-ink-2">Actual P&amp;L</dt>
          <dd className="tabular font-medium text-ink">{signedUsdCompact(pnl.actual)}</dd>
        </div>
        <div className="flex justify-between py-2.5">
          <dt className="text-ink-2">Expected by risk model</dt>
          <dd className="tabular font-medium text-ink">{signedUsdCompact(pnl.expected)}</dd>
        </div>
        <div className="flex justify-between py-2.5">
          <dt className="text-ink-2">Unexplained</dt>
          <dd className="tabular font-semibold text-critical">
            {usdCompact(pnl.unexplained)} ({pct(pnl.unexplainedPctOfBook, 1)} of book)
          </dd>
        </div>
      </dl>
      <div className="mt-4 rounded-xl bg-subtle px-4 py-3 text-sm">
        <p className="text-ink-2">Traced to</p>
        <p className="mt-0.5 font-medium text-ink">
          {pnl.topPositionLabel}: {pct(pnl.topPositionShare)} of the gap
        </p>
      </div>
      <Link
        href="/pnl-explain"
        className="mt-4 inline-flex text-sm font-medium text-brand hover:text-brand-strong"
      >
        Open the full breakdown &rarr;
      </Link>
    </div>
  );
}

export default function Home() {
  return (
    <div className="flex flex-col gap-24">
      <section className="grid items-center gap-12 pt-4 lg:grid-cols-[1.15fr_1fr]">
        <div className="flex flex-col gap-6">
          <p className="text-sm font-medium text-brand">
            Risk and P&amp;L intelligence for equity derivatives desks
          </p>
          <h1 className="text-4xl font-semibold leading-[1.1] tracking-tight text-ink sm:text-5xl">
            Know why your P&amp;L moved, before the desk asks.
          </h1>
          <p className="max-w-xl text-lg leading-relaxed text-ink-2">
            EQD Risk Engine prices every position in an options and structured-products book each
            day, explains what drove the profit and loss, flags anything that doesn&apos;t add up,
            and puts an AI risk analyst on the gaps.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/pnl-explain"
              className="rounded-xl bg-brand px-5 py-3 text-sm font-medium text-white shadow-sm hover:bg-brand-strong"
            >
              See a real incident
            </Link>
            <a
              href="#how-it-works"
              className="rounded-xl border border-border bg-surface px-5 py-3 text-sm font-medium text-ink hover:bg-subtle"
            >
              How it works
            </a>
          </div>
          <DemoNote />
        </div>
        <IncidentPreview />
      </section>

      <section className="flex flex-col gap-6">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">Results from real runs</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatTile
            label="Unexplained P&L caught in one day"
            value={usdCompact(pnl.unexplained)}
            detail={`${pct(pnl.topPositionShare)} traced to a single structured note.`}
          />
          <StatTile
            label="Smaller P&L swings when hedging a barrier option"
            value={pct(hedging.bestBarrierSwingReduction)}
            detail="AI-trained hedging versus the benchmark hedge, on scenarios it never trained on."
          />
          <StatTile
            label="Market quotes behind each day's pricing"
            value={pricing.quotes.toLocaleString("en-US")}
            detail={`S&P 500 option quotes across ${pricing.maturities} maturities, fitted into one consistent surface.`}
          />
          <StatTile
            label="Agreement with the industry benchmark"
            value="8 decimals"
            detail="Option prices match QuantLib to eight decimal places."
          />
        </div>
      </section>

      <section className="flex flex-col gap-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-ink">What it does</h2>
          <p className="mt-2 max-w-2xl text-ink-2">
            Four views of the same book, each one open with real output.
          </p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          {CAPABILITIES.map((c) => (
            <Link
              key={c.href}
              href={c.href}
              className="group flex flex-col gap-2 rounded-2xl border border-border bg-surface p-6 shadow-[0_1px_2px_rgba(14,26,43,0.04)] transition hover:border-brand/40 hover:shadow-[0_8px_24px_rgba(14,26,43,0.06)]"
            >
              <h3 className="font-semibold text-ink">{c.title}</h3>
              <p className="text-sm leading-relaxed text-ink-2">{c.body}</p>
              <span className="mt-2 text-sm font-medium text-brand">
                View <span aria-hidden>&rarr;</span>
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section id="how-it-works" className="scroll-mt-24 flex flex-col gap-6">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">How it works</h2>
        <ol className="grid gap-4 md:grid-cols-3">
          {STEPS.map((s, i) => (
            <li key={s.title} className="rounded-2xl border border-border bg-surface p-6">
              <span className="grid h-8 w-8 place-items-center rounded-full bg-brand-soft text-sm font-semibold text-brand">
                {i + 1}
              </span>
              <h3 className="mt-4 font-semibold text-ink">{s.title}</h3>
              <p className="mt-1 text-sm leading-relaxed text-ink-2">{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="rounded-3xl bg-ink px-8 py-10 text-white sm:px-10">
        <h2 className="text-2xl font-semibold tracking-tight">Built to be trusted with a real book</h2>
        <div className="mt-8 grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {TRUST.map((t) => (
            <div key={t.title}>
              <h3 className="font-semibold">{t.title}</h3>
              <p className="mt-1 text-sm leading-relaxed text-white/70">{t.body}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
