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
import { fmtShortDate, usdCompact } from "@/lib/format";
import { GRID, SERIES, axisProps, legendProps, tooltipProps } from "@/lib/chartTheme";

/** Daily 1-day 99% VaR by both methods, one shared dollar axis. */
export function VarHistory({ data }: { data: { date: string; full: number; taylor: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="date" {...axisProps} tickFormatter={(v) => fmtShortDate(String(v))} minTickGap={24} />
        <YAxis
          {...axisProps}
          axisLine={false}
          width={52}
          domain={[0, "auto"]}
          tickFormatter={(v) => usdCompact(Number(v))}
        />
        <Tooltip
          {...tooltipProps}
          labelFormatter={(v) => fmtShortDate(String(v))}
          formatter={(v, name) => [usdCompact(Number(v)), name]}
        />
        <Legend {...legendProps} />
        <Line
          type="monotone"
          dataKey="full"
          name="Full revaluation"
          stroke={SERIES[0]}
          strokeWidth={2}
          dot={{ r: 3 }}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="taylor"
          name="Greeks shortcut"
          stroke={SERIES[1]}
          strokeWidth={2}
          dot={{ r: 3 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
