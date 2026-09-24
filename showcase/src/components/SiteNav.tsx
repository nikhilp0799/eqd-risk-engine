"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/deep-hedging", label: "Deep Hedging" },
  { href: "/ai-agent", label: "AI Agent" },
  { href: "/vol-surface", label: "Vol Surface" },
  { href: "/pnl-explain", label: "P&L Explain" },
];

export function SiteNav() {
  const pathname = usePathname();

  return (
    <header className="border-b border-panel-border bg-background/80 backdrop-blur sticky top-0 z-10">
      <div className="mx-auto max-w-6xl px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <Link href="/" className="flex items-baseline gap-2 font-num text-sm tracking-wide">
          <span className="text-accent">&gt;</span>
          <span className="text-foreground font-semibold">EQD Risk Engine</span>
        </Link>
        <nav className="flex flex-wrap gap-1 text-sm">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded px-3 py-1.5 transition-colors ${
                  active
                    ? "bg-accent-soft text-accent"
                    : "text-muted hover:text-foreground"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
