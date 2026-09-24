"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DeepHedgeRow } from "@/lib/types";

const CHART_COLORS = { learned: "#2dd4bf", baseline: "#475569" };

export function DeepHedgeBars({ rows }: { rows: DeepHedgeRow[] }) {
  const chartData = rows.map((r) => ({
    loss_type: r.loss_type,
    "learned policy": r.learned_std,
    "static baseline": r.baseline_std,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2b3d" />
        <XAxis dataKey="loss_type" stroke="#7d8ba0" fontSize={12} />
        <YAxis stroke="#7d8ba0" fontSize={12} width={70} />
        <Tooltip
          contentStyle={{
            background: "#111826",
            border: "1px solid #1f2b3d",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "#dbe4f0" }}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: "#7d8ba0" }} />
        <Bar dataKey="learned policy" fill={CHART_COLORS.learned} radius={[4, 4, 0, 0]} />
        <Bar dataKey="static baseline" fill={CHART_COLORS.baseline} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
