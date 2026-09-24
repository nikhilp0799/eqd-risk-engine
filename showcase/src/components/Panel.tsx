import type { ReactNode } from "react";

export function Panel({
  title,
  caption,
  children,
  className = "",
}: {
  title?: string;
  caption?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-panel-border bg-panel/60 p-6 ${className}`}
    >
      {title && (
        <h2 className="text-lg font-semibold text-foreground mb-1">{title}</h2>
      )}
      {caption && <p className="text-sm text-muted mb-4">{caption}</p>}
      {children}
    </section>
  );
}

export function StatCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="rounded-xl border border-panel-border bg-panel/60 p-5 flex flex-col gap-2">
      <p className="text-xs uppercase tracking-wide text-muted">{label}</p>
      <p className="font-num text-xl text-accent leading-snug">{value}</p>
      <p className="text-sm text-muted">{detail}</p>
    </div>
  );
}

export function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "positive" | "negative" }) {
  const toneClass =
    tone === "positive"
      ? "bg-positive/10 text-positive"
      : tone === "negative"
      ? "bg-negative/10 text-negative"
      : "bg-accent-soft text-accent";
  return (
    <span className={`font-num text-xs px-2 py-0.5 rounded ${toneClass}`}>
      {children}
    </span>
  );
}
