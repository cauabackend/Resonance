import { useId, useState, type KeyboardEvent, type ReactNode } from "react";

export interface TabDef {
  id: string;
  label: string;
  content: ReactNode;
}

export default function Tabs({ tabs }: { tabs: TabDef[] }) {
  const [active, setActive] = useState(tabs[0].id);
  const uid = useId();

  const onKey = (e: KeyboardEvent, i: number) => {
    const next = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : null;
    if (next === null) return;
    const t = tabs[(next + tabs.length) % tabs.length];
    setActive(t.id);
    document.getElementById(`${uid}-tab-${t.id}`)?.focus();
  };

  return (
    <div>
      <ul className="nav nav-tabs mb-3" role="tablist">
        {tabs.map((t, i) => (
          <li className="nav-item" role="presentation" key={t.id}>
            <button
              id={`${uid}-tab-${t.id}`}
              role="tab"
              type="button"
              className={`nav-link ${active === t.id ? "active" : ""}`}
              aria-selected={active === t.id}
              aria-controls={`${uid}-panel-${t.id}`}
              tabIndex={active === t.id ? 0 : -1}
              onClick={() => setActive(t.id)}
              onKeyDown={(e) => onKey(e, i)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>
      {tabs.map((t) => (
        <div
          key={t.id}
          id={`${uid}-panel-${t.id}`}
          role="tabpanel"
          aria-labelledby={`${uid}-tab-${t.id}`}
          hidden={active !== t.id}
        >
          {active === t.id && t.content}
        </div>
      ))}
    </div>
  );
}
