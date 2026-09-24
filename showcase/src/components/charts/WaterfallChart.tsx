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
import type { PnlWaterfallStep } from "@/lib/types";

export function WaterfallChart({ steps }: { steps: PnlWaterfallStep[] }) {
  const chartData = steps.map((s) => ({
    step: s.step,
    actual: s.actual_pnl,
    explained: s.explained_pnl,
    residual: s.residual,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2b3d" />
        <XAxis dataKey="step" stroke="#7d8ba0" fontSize={12} />
        <YAxis stroke="#7d8ba0" fontSize={12} width={70} />
        <Tooltip
          contentStyle={{
            background: "#111826",
            border: "1px solid #1f2b3d",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "#dbe4f0" }}
          formatter={(value) => `$${Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 })}`}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: "#7d8ba0" }} />
        <Bar dataKey="actual" fill="#38bdf8" radius={[4, 4, 0, 0]} />
        <Bar dataKey="explained" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
        <Bar dataKey="residual" fill="#f87171" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
