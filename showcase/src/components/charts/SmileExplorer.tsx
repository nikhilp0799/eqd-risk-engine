"use client";

import { useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { VolSmile } from "@/lib/types";

export function SmileExplorer({ smiles }: { smiles: Record<string, VolSmile> }) {
  const expiries = Object.keys(smiles).sort();
  const [selected, setSelected] = useState(expiries[expiries.length - 1] ?? expiries[0]);
  const smile = smiles[selected];
  if (!smile) return null;

  const fitData = smile.k_grid.map((k, i) => ({ k, iv: smile.fit_iv[i] }));
  const marketData = smile.market_k.map((k, i) => ({ k, iv: smile.market_iv[i] }));

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2">
        {expiries.map((exp) => (
          <button
            key={exp}
            onClick={() => setSelected(exp)}
            className={`font-num text-xs px-2.5 py-1 rounded transition-colors ${
              exp === selected
                ? "bg-accent-soft text-accent"
                : "text-muted hover:text-foreground border border-panel-border"
            }`}
          >
            {exp} (T={smiles[exp].T.toFixed(2)})
          </button>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2b3d" />
          <XAxis
            dataKey="k"
            type="number"
            domain={[-1, 1]}
            stroke="#7d8ba0"
            fontSize={12}
            label={{ value: "log-moneyness k", position: "insideBottom", offset: -4, fill: "#7d8ba0", fontSize: 12 }}
          />
          <YAxis
            dataKey="iv"
            type="number"
            stroke="#7d8ba0"
            fontSize={12}
            width={55}
            label={{ value: "IV", angle: -90, position: "insideLeft", fill: "#7d8ba0", fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{
              background: "#111826",
              border: "1px solid #1f2b3d",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#dbe4f0" }}
            formatter={(value) => Number(value).toFixed(4)}
          />
          <Line
            data={fitData}
            dataKey="iv"
            name={`${smile.model} fit`}
            type="monotone"
            stroke="#2dd4bf"
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
          <Scatter data={marketData} dataKey="iv" name="market (OK quotes)" fill="#f87171" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
