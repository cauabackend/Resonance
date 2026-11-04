/** Erro recuperável: mensagem + botão de nova tentativa (a API pode ter caído só por um instante). */
export default function ErrorPanel({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="panel d-flex flex-wrap align-items-center justify-content-between gap-3">
      <span>{message}</span>
      {onRetry && (
        <button type="button" className="btn btn-outline-light btn-sm" onClick={onRetry}>
          Tentar de novo
        </button>
      )}
    </div>
  );
}
