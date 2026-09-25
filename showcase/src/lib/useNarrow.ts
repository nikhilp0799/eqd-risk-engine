import { useEffect, useState } from "react";

/** True below Tailwind's `sm` breakpoint, for charts that need a compact
 * layout on phones. False during static prerender (desktop-first). */
export function useNarrow(maxWidth = 639): boolean {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${maxWidth}px)`);
    const update = () => setNarrow(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, [maxWidth]);
  return narrow;
}
