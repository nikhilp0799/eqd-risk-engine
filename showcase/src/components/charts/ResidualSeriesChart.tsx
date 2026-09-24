"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PnlResidualPoint } from "@/lib/types";

export function ResidualSeriesChart({
  series,
  thresholdBp,
}: {
  series: PnlResidualPoint[];
  thresholdBp: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={series} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2b3d" />
        <XAxis dataKey="asof_date" stroke="#7d8ba0" fontSize={11} />
        <YAxis stroke="#7d8ba0" fontSize={12} width={55} />
        <Tooltip
          contentStyle={{
            background: "#111826",
            border: "1px solid #1f2b3d",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "#dbe4f0" }}
          formatter={(value) => `${Number(value).toFixed(1)}bp`}
        />
        <ReferenceLine y={thresholdBp} stroke="#f87171" strokeDasharray="4 4" />
        <ReferenceLine y={-thresholdBp} stroke="#f87171" strokeDasharray="4 4" />
        <ReferenceLine y={0} stroke="#1f2b3d" />
        <Line
          type="monotone"
          dataKey="residual_bp"
          stroke="#2dd4bf"
          dot={{ r: 2 }}
          strokeWidth={2}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
