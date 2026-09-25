"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/pnl-explain", label: "P&L Explain" },
  { href: "/ai-agent", label: "AI Risk Analyst" },
  { href: "/deep-hedging", label: "Hedging" },
  { href: "/vol-surface", label: "Pricing" },
];

export function SiteNav() {
  const pathname = usePathname();
  const active = (href: string) => pathname === href || pathname === `${href}/`;

  return (
    <header className="sticky top-0 z-20 border-b border-border bg-surface/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-6 py-3.5">
        <Link href="/" className="flex shrink-0 items-center gap-2.5">
          <span
            aria-hidden
            className="grid h-7 w-7 place-items-center rounded-lg bg-brand text-xs font-bold text-white"
          >
            EQ
          </span>
          <span className="font-semibold tracking-tight text-ink">EQD Risk Engine</span>
        </Link>
        <nav className="-mr-2 flex min-w-0 gap-1 overflow-x-auto text-sm">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`whitespace-nowrap rounded-lg px-3 py-1.5 transition-colors ${
                active(link.href)
                  ? "bg-brand-soft font-medium text-brand"
                  : "text-ink-2 hover:bg-subtle hover:text-ink"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
