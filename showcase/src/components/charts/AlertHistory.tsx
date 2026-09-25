"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, GRID, INK_2, SERIES, axisProps, tooltipProps } from "@/lib/chartTheme";

/** Unexplained P&L per trading day as % of book value. The alert level is
 * stated in the caption, not drawn: at this scale a ±0.05% line sits on the
 * zero line and reads as the baseline. */
export function AlertHistory({
  data,
  highlightDate,
}: {
  data: { date: string; label: string; pct: number }[];
  highlightDate: string;
}) {
  const fmt = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 24, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="label" {...axisProps} />
        <YAxis {...axisProps} axisLine={false} width={52} tickFormatter={(v) => `${v}%`} />
        <ReferenceLine y={0} stroke={AXIS} />
        <Tooltip {...tooltipProps} formatter={(v) => [fmt(Number(v)), "Unexplained, % of book"]} />
        <Bar dataKey="pct" maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.date} fill={SERIES[0]} fillOpacity={d.date === highlightDate ? 1 : 0.45} />
          ))}
          <LabelList dataKey="pct" position="top" formatter={(v) => fmt(Number(v))} style={{ fill: INK_2, fontSize: 11 }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
