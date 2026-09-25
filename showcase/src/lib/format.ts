/** $1,234 / $86K / $4.2M — compact dollars for headlines and stat tiles. */
export function usdCompact(x: number): string {
  const sign = x < 0 ? "-" : "";
  const a = Math.abs(x);
  if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${sign}$${Math.round(a / 1e3)}K`;
  return `${sign}$${Math.round(a)}`;
}

export function signedUsdCompact(x: number): string {
  return `${x >= 0 ? "+" : ""}${usdCompact(x)}`;
}

export function usd(x: number, digits = 0): string {
  const sign = x < 0 ? "-" : "";
  return `${sign}$${Math.abs(x).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

export function pct(x: number, digits = 0): string {
  return `${x.toFixed(digits)}%`;
}

export function signedPct(x: number, digits = 0): string {
  return `${x > 0 ? "+" : ""}${x.toFixed(digits)}%`;
}

export function fmtDate(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function fmtShortDate(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

/** Plain-language maturity: "1 day", "2 weeks", "6 months", "1 year". */
export function maturityLabel(T: number): string {
  const days = Math.max(1, Math.round(T * 365));
  if (days < 14) return `${days} day${days === 1 ? "" : "s"}`;
  if (days < 60) return `${Math.round(days / 7)} weeks`;
  if (days < 330) return `${Math.round(days / 30)} months`;
  return `${Math.round(days / 365)} year`;
}

// Chart components are client components, and a server page can't hand them a
// function — so charts take one of these names instead.
export const FORMATTERS = {
  usdCompact,
  signedPct1: (x: number) => signedPct(x, 1),
  fixed1: (x: number) => x.toFixed(1),
  fixed2: (x: number) => x.toFixed(2),
} as const;
export type FormatName = keyof typeof FORMATTERS;
