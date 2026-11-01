"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { usdCompact } from "@/lib/format";
import { AXIS, CRITICAL, GRID, INK_2, SERIES, axisProps, tooltipProps } from "@/lib/chartTheme";

/** How many replayed days landed in each P&L range. Ranges at or beyond the
 * VaR loss are drawn in the critical color; the caption names them, so the
 * color is never the only cue. */
export function LossHistogram({
  bins,
  varLoss,
}: {
  bins: { lo: number; hi: number; count: number }[];
  varLoss: number;
}) {
  const data = bins.map((b) => ({
    mid: (b.lo + b.hi) / 2,
    count: b.count,
    range: `${usdCompact(b.lo)} to ${usdCompact(b.hi)}`,
    tail: b.lo < -varLoss,
  }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 24, right: 8, left: 0, bottom: 0 }} barCategoryGap={2}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis
          dataKey="mid"
          type="number"
          domain={["dataMin", "dataMax"]}
          {...axisProps}
          tickFormatter={(v) => usdCompact(Number(v))}
        />
        <YAxis {...axisProps} axisLine={false} width={36} allowDecimals={false} />
        <ReferenceLine
          x={-varLoss}
          stroke={CRITICAL}
          strokeDasharray="4 3"
          label={{ value: "99% VaR", position: "top", fill: INK_2, fontSize: 12 }}
        />
        <ReferenceLine y={0} stroke={AXIS} />
        <Tooltip
          {...tooltipProps}
          labelFormatter={(_, payload) => String(payload?.[0]?.payload?.range ?? "")}
          formatter={(v) => [`${v} days`, "Replayed days"]}
        />
        <Bar dataKey="count" radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.mid} fill={d.tail ? CRITICAL : SERIES[0]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
