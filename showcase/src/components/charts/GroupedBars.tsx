"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FORMATTERS, type FormatName } from "@/lib/format";
import { useNarrow } from "@/lib/useNarrow";
import { AXIS, GRID, INK_2, axisProps, legendProps, tooltipProps } from "@/lib/chartTheme";

/** Round-number ticks (1/2/2.5/5 x 10^n steps) with headroom past the data on
 * both sides, so cap labels never collide with the axis labels. */
function niceAxis(min: number, max: number) {
  const span = Math.max(max - min, 1e-9);
  const raw = span / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const lo = min < 0 ? Math.floor((min - 0.1 * span) / step) * step : 0;
  const hi = max > 0 ? Math.ceil((max + 0.1 * span) / step) * step : 0;
  const ticks: number[] = [];
  for (let t = lo; t <= hi + step / 2; t += step) ticks.push(Number(t.toFixed(10)));
  return { domain: [lo, hi] as [number, number], ticks };
}

export interface BarSeries {
  key: string;
  label: string;
  color: string;
}

/** Vertical grouped columns on one shared axis. With `labels`, values are
 * printed on the caps, skipping bars too small to label cleanly (those stay in
 * the tooltip); without, the page pairs the chart with a table view. */
export function GroupedBars({
  data,
  categoryKey,
  shortCategoryKey,
  series,
  format: formatName,
  height = 300,
  labels = true,
}: {
  data: Record<string, string | number>[];
  categoryKey: string;
  /** Shorter category labels used on phone-width screens. */
  shortCategoryKey?: string;
  series: BarSeries[];
  format: FormatName;
  height?: number;
  labels?: boolean;
}) {
  const format = FORMATTERS[formatName];
  const narrow = useNarrow();
  const values = data.flatMap((d) => series.map((s) => Number(d[s.key] ?? 0)));
  const maxAbs = Math.max(...values.map(Math.abs), 1e-9);
  const { domain, ticks } = niceAxis(Math.min(0, ...values), Math.max(0, ...values));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 24, right: 8, left: 0, bottom: 0 }} barGap={2} barCategoryGap="24%">
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey={narrow && shortCategoryKey ? shortCategoryKey : categoryKey} {...axisProps} interval={0} />
        <YAxis
          {...axisProps}
          axisLine={false}
          width={56}
          domain={domain}
          ticks={ticks}
          tickFormatter={(v) => format(Number(v))}
        />
        <ReferenceLine y={0} stroke={AXIS} />
        <Tooltip {...tooltipProps} formatter={(v, name) => [format(Number(v)), name]} />
        <Legend {...legendProps} />
        {series.map((s) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false}>
            {labels && (
              <LabelList
                dataKey={s.key}
                position="top"
                formatter={(v) => (Math.abs(Number(v)) < 0.05 * maxAbs ? "" : format(Number(v)))}
                style={{ fill: INK_2, fontSize: 11 }}
              />
            )}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
