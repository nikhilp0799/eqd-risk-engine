"use client";

import { useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SERIES, axisProps, legendProps, tooltipProps } from "@/lib/chartTheme";
import { maturityLabel } from "@/lib/format";
import type { VolSmile } from "@/lib/types";

// Log-moneyness k -> strike as % of the forward price, the way a desk reads it.
const toPct = (k: number) => 100 * Math.exp(k);

export function SmileExplorer({ smiles }: { smiles: Record<string, VolSmile> }) {
  const expiries = Object.keys(smiles).sort();
  const [selected, setSelected] = useState(expiries[expiries.length - 1]);
  const smile = smiles[selected];

  // Keep the view where the market actually quotes, with a little margin.
  const lo = Math.min(...smile.market_k) - 0.05;
  const hi = Math.max(...smile.market_k) + 0.05;
  const fit = smile.k_grid
    .map((k, i) => ({ x: toPct(k), iv: 100 * smile.fit_iv[i] }))
    .filter((p) => p.x >= toPct(lo) && p.x <= toPct(hi));
  const market = smile.market_k.map((k, i) => ({ x: toPct(k), iv: 100 * smile.market_iv[i] }));

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Maturity">
        {expiries.map((exp) => (
          <button
            key={exp}
            role="tab"
            aria-selected={exp === selected}
            onClick={() => setSelected(exp)}
            className={`rounded-lg border px-3 py-1.5 text-sm transition-colors ${
              exp === selected
                ? "border-brand bg-brand-soft font-medium text-brand"
                : "border-border text-ink-2 hover:bg-subtle"
            }`}
          >
            {maturityLabel(smiles[exp].T)}
          </button>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart margin={{ top: 8, right: 8, left: 0, bottom: 16 }}>
          <CartesianGrid stroke="#eceef2" />
          <XAxis
            dataKey="x"
            type="number"
            domain={["dataMin", "dataMax"]}
            {...axisProps}
            tickFormatter={(v) => `${Math.round(Number(v))}%`}
            label={{ value: "Strike, % of current forward price", position: "insideBottom", offset: -12, fill: "#7a8699", fontSize: 12 }}
          />
          <YAxis
            dataKey="iv"
            type="number"
            {...axisProps}
            axisLine={false}
            width={48}
            domain={["auto", "auto"]}
            tickFormatter={(v) => `${Math.round(Number(v))}%`}
          />
          <Tooltip
            {...tooltipProps}
            formatter={(v) => `${Number(v).toFixed(1)}%`}
            labelFormatter={(v) => `Strike ${Number(v).toFixed(1)}% of forward`}
          />
          <Legend {...legendProps} verticalAlign="top" align="right" height={28} />
          <Scatter data={market} dataKey="iv" name="Market quotes" fill={SERIES[1]} r={3} isAnimationActive={false} />
          <Line
            data={fit}
            dataKey="iv"
            name="Engine's fitted curve"
            type="monotone"
            stroke={SERIES[0]}
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
