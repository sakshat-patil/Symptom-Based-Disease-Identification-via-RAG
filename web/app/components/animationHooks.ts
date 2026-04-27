"use client";
import { useEffect, useRef, useState } from "react";

/**
 * Tween a number from 0 to `target` over `durationMs`.
 * Re-runs whenever `target` or `key` changes (the latter lets a caller
 * force a replay without touching the value).
 */
export function useCountUp(
  target: number,
  durationMs = 500,
  key: number | string = 0
): number {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      // ease-out-cubic
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(target * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, key]);
  return value;
}

/**
 * Reveal a string character-by-character at `cps` chars/sec.
 * Re-runs when `text` or `key` changes.
 */
export function useTypewriter(
  text: string,
  cps = 60,
  key: number | string = 0
): string {
  const [shown, setShown] = useState("");
  useEffect(() => {
    setShown("");
    if (!text) return;
    let cancelled = false;
    const stepMs = 1000 / cps;
    let i = 0;
    let timer: ReturnType<typeof setTimeout>;
    const next = () => {
      if (cancelled) return;
      i += 1;
      setShown(text.slice(0, i));
      if (i < text.length) timer = setTimeout(next, stepMs);
    };
    timer = setTimeout(next, stepMs);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [text, cps, key]);
  return shown;
}

/**
 * Returns an incrementing reveal counter that fires once per item until
 * `count` items are visible. Used to drive staggered list reveals where
 * each item's `data-revealed` flips true after `i * stepMs`.
 *
 * Usage:
 *   const visibleCount = useStaggeredReveal(items.length, 60, runKey);
 *   {items.map((it, i) => (
 *     <li data-revealed={i < visibleCount}>...</li>
 *   ))}
 */
export function useStaggeredReveal(
  count: number,
  stepMs = 60,
  key: number | string = 0
): number {
  const [revealed, setRevealed] = useState(0);
  const cancelRef = useRef(false);
  useEffect(() => {
    cancelRef.current = false;
    setRevealed(0);
    if (count <= 0) return;
    let i = 0;
    const tick = () => {
      if (cancelRef.current) return;
      i += 1;
      setRevealed(i);
      if (i < count) setTimeout(tick, stepMs);
    };
    const start = setTimeout(tick, stepMs);
    return () => {
      cancelRef.current = true;
      clearTimeout(start);
    };
  }, [count, stepMs, key]);
  return revealed;
}
