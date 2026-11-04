import { useState } from "react";

const KEY = "resonance-contrast";

function stored(): boolean {
  try {
    return localStorage.getItem(KEY) === "high";
  } catch {
    return false; // storage bloqueado (janela privada): o toggle funciona, só não persiste
  }
}

function apply(high: boolean) {
  document.documentElement.dataset.contrast = high ? "high" : "normal";
}

apply(stored()); // antes do primeiro render, sem flash

/** Alto contraste (WCAG AA reforçado) sobre a base monocromática. */
export default function ContrastToggle() {
  const [high, setHigh] = useState(stored);

  function toggle() {
    const next = !high;
    setHigh(next);
    apply(next);
    try {
      localStorage.setItem(KEY, next ? "high" : "normal");
    } catch {
      /* sem persistência */
    }
  }

  return (
    <button type="button" className="contrast-toggle ms-auto" aria-pressed={high} onClick={toggle}>
      Alto contraste
    </button>
  );
}
