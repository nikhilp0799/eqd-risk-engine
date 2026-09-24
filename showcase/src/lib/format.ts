export function fmtNum(x: number, digits = 2): string {
  return x.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtUsd(x: number, digits = 0): string {
  const sign = x < 0 ? "-" : "";
  return `${sign}$${Math.abs(x).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

export function fmtPct(x: number, digits = 1): string {
  return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}%`;
}

export function fmtBp(x: number, digits = 1): string {
  return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}bp`;
}

export function fmtDate(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}
