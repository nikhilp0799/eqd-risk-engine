import Link from "next/link";
import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-2xl border border-border bg-surface p-6 sm:p-7 shadow-[0_1px_2px_rgba(14,26,43,0.04)] ${className}`}
    >
      {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
      {subtitle && <p className="mt-1 text-sm text-ink-2 max-w-3xl">{subtitle}</p>}
      <div className={title || subtitle ? "mt-5" : ""}>{children}</div>
    </section>
  );
}

export function StatTile({
  label,
  value,
  detail,
  tone = "default",
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "default" | "good" | "critical";
}) {
  const valueColor =
    tone === "good" ? "text-good" : tone === "critical" ? "text-critical" : "text-ink";
  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-[0_1px_2px_rgba(14,26,43,0.04)]">
      <p className="text-sm text-ink-2">{label}</p>
      <p className={`mt-2 text-3xl font-semibold tracking-tight ${valueColor}`}>{value}</p>
      {detail && <p className="mt-2 text-sm text-muted leading-snug">{detail}</p>}
    </div>
  );
}

export function PageHeader({
  section,
  title,
  lede,
  meta,
}: {
  section: string;
  title: string;
  lede: string;
  meta?: string;
}) {
  return (
    <header className="flex flex-col gap-3">
      <p className="text-sm font-medium text-brand">{section}</p>
      <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink">{title}</h1>
      <p className="text-lg text-ink-2 max-w-3xl leading-relaxed">{lede}</p>
      {meta && <p className="text-sm text-muted">{meta}</p>}
    </header>
  );
}

export function Takeaway({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-brand/15 bg-brand-soft px-6 py-5">
      <p className="text-xs font-semibold uppercase tracking-wider text-brand">Key takeaway</p>
      <div className="mt-2 text-[17px] leading-relaxed text-ink">{children}</div>
    </div>
  );
}

export function Methodology({ children }: { children: ReactNode }) {
  return (
    <details className="group rounded-2xl border border-border bg-surface">
      <summary className="flex items-center justify-between gap-4 px-6 py-4">
        <span>
          <span className="text-sm font-semibold text-ink">Methodology</span>
          <span className="ml-2 text-sm text-muted">For quant and model-risk reviewers</span>
        </span>
        <span className="chevron text-muted transition-transform" aria-hidden>
          &#8250;
        </span>
      </summary>
      <div className="border-t border-border px-6 py-5 text-sm leading-relaxed text-ink-2 flex flex-col gap-3">
        {children}
      </div>
    </details>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "critical" | "brand";
}) {
  const cls = {
    neutral: "bg-subtle text-ink-2",
    good: "bg-good-soft text-good",
    critical: "bg-critical-soft text-critical",
    brand: "bg-brand-soft text-brand",
  }[tone];
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${cls}`}>
      {children}
    </span>
  );
}

export function NextLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link
      href={href}
      className="inline-flex items-center gap-1 text-sm font-medium text-brand hover:text-brand-strong"
    >
      {children} <span aria-hidden>&rarr;</span>
    </Link>
  );
}

export function DemoNote() {
  return (
    <p className="text-xs text-muted">
      Demo book: a model portfolio of 9 positions, priced on real market data. Figures come from
      actual daily runs.
    </p>
  );
}
