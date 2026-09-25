"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { FORMATTERS, type FormatName } from "@/lib/format";
import { GRID, SERIES, axisProps, tooltipProps } from "@/lib/chartTheme";

/** One sensitivity against strike — single series, its own axis. Two of these
 * side by side replace the old dual-axis delta/vega chart. */
export function SensitivityLine({
  data,
  name,
  format: formatName,
}: {
  data: { strike: number; value: number }[];
  name: string;
  format: FormatName;
}) {
  const format = FORMATTERS[formatName];
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis
          dataKey="strike"
          type="number"
          domain={["dataMin", "dataMax"]}
          {...axisProps}
          tickFormatter={(v) => Number(v).toLocaleString("en-US")}
        />
        <YAxis {...axisProps} axisLine={false} width={48} tickFormatter={(v) => format(Number(v))} />
        <Tooltip
          {...tooltipProps}
          formatter={(v) => [format(Number(v)), name]}
          labelFormatter={(v) => `Strike ${Number(v).toLocaleString("en-US")}`}
        />
        <Line type="monotone" dataKey="value" stroke={SERIES[0]} strokeWidth={2} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
