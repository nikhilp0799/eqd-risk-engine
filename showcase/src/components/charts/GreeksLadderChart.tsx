"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { GreeksLadder } from "@/lib/types";

export function GreeksLadderChart({ ladder }: { ladder: GreeksLadder }) {
  const chartData = ladder.strike.map((strike, i) => ({
    strike,
    delta: ladder.delta_spot[i],
    vega: ladder.vega[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2b3d" />
        <XAxis dataKey="strike" stroke="#7d8ba0" fontSize={12} />
        <YAxis yAxisId="delta" stroke="#2dd4bf" fontSize={12} width={50} domain={[0, 1]} />
        <YAxis yAxisId="vega" orientation="right" stroke="#38bdf8" fontSize={12} width={60} />
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
        <Line
          yAxisId="delta"
          type="monotone"
          dataKey="delta"
          name="delta (spot)"
          stroke="#2dd4bf"
          dot={false}
          strokeWidth={2}
          isAnimationActive={false}
        />
        <Line
          yAxisId="vega"
          type="monotone"
          dataKey="vega"
          name="vega"
          stroke="#38bdf8"
          dot={false}
          strokeWidth={2}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
