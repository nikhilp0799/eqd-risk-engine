"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PnlByPosition } from "@/lib/types";

export function ByPositionChart({ positions }: { positions: PnlByPosition[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={positions} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2b3d" />
        <XAxis dataKey="position_id" stroke="#7d8ba0" fontSize={12} />
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
        <Bar dataKey="residual" radius={[4, 4, 0, 0]}>
          {positions.map((p) => (
            <Cell key={p.position_id} fill={p.residual >= 0 ? "#f87171" : "#38bdf8"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
