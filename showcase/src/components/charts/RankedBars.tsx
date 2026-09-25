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
import { FORMATTERS, type FormatName } from "@/lib/format";
import { useNarrow } from "@/lib/useNarrow";
import { AXIS, DEEMPHASIS, GRID, INK_2, SERIES, axisProps, tooltipProps } from "@/lib/chartTheme";

/** Horizontal single-series bars, one item highlighted and the rest in gray —
 * for "which one is the story" charts. */
export function RankedBars({
  data,
  highlight,
  format: formatName,
}: {
  data: { label: string; value: number }[];
  highlight: string;
  format: FormatName;
}) {
  const format = FORMATTERS[formatName];
  const narrow = useNarrow();
  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 44 + 20)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: narrow ? 48 : 64, left: 0, bottom: 0 }}>
        <CartesianGrid horizontal={false} stroke={GRID} />
        <XAxis type="number" {...axisProps} tickFormatter={(v) => format(Number(v))} />
        <YAxis
          type="category"
          dataKey="label"
          {...axisProps}
          axisLine={false}
          width={narrow ? 118 : 190}
          tick={{ fill: INK_2, fontSize: narrow ? 11 : 13 }}
        />
        <ReferenceLine x={0} stroke={AXIS} />
        <Tooltip {...tooltipProps} formatter={(v) => [format(Number(v)), "Unexplained"]} />
        <Bar dataKey="value" maxBarSize={22} radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.label} fill={d.label === highlight ? SERIES[0] : DEEMPHASIS} />
          ))}
          <LabelList
            dataKey="value"
            position="right"
            formatter={(v) => format(Number(v))}
            style={{ fill: INK_2, fontSize: 12 }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
