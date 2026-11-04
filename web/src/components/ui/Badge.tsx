// Diferencia por peso, caixa e contraste, nunca por matiz (tema monocromático).
export function VerdictBadge({ isHit }: { isHit: boolean }) {
  return isHit ? (
    <span className="verdict verdict--hit">Potencial de hit</span>
  ) : (
    <span className="verdict verdict--miss">Baixo potencial</span>
  );
}

export function Chip({ children }: { children: string }) {
  return <span className="chip">{children}</span>;
}
