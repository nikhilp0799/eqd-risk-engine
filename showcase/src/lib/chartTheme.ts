// Chart palette: the dataviz reference categorical slots 1-3, validated
// all-pairs on the white card surface (#ffffff). Slot 3 sits below 3:1
// contrast, so every chart using it carries visible value labels.
export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a"] as const;

// De-emphasis gray for "everything that isn't the story" (baselines, other
// positions).
export const DEEMPHASIS = "#c5ccd6";
export const CRITICAL = "#c2352f";

export const INK = "#0e1a2b";
export const INK_2 = "#475467";
export const MUTED = "#7a8699";
export const GRID = "#eceef2";
export const AXIS = "#cfd4dc";

export const axisProps = {
  stroke: AXIS,
  tick: { fill: MUTED, fontSize: 12 },
  tickLine: false,
} as const;

export const tooltipProps = {
  contentStyle: {
    background: "#ffffff",
    border: "1px solid #e3e6eb",
    borderRadius: 8,
    boxShadow: "0 4px 16px rgba(14, 26, 43, 0.08)",
    fontSize: 12,
    color: INK,
  },
  labelStyle: { color: INK, fontWeight: 600, marginBottom: 4 },
  itemStyle: { color: INK_2 },
  cursor: { fill: "rgba(14, 26, 43, 0.04)" },
} as const;

export const legendProps = {
  iconType: "circle" as const,
  iconSize: 8,
  // Recharts sorts legend entries alphabetically by default; keep series order.
  itemSorter: null,
  wrapperStyle: { fontSize: 12, color: INK_2, paddingTop: 8 },
};
