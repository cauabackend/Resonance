import { useEffect, useState } from "react";
import { searchTracks } from "../api/client";
import type { TrackOut } from "../api/types";
import ErrorPanel from "../components/ui/ErrorPanel";
import SearchBar from "../features/search/SearchBar";
import SearchResultsList from "../features/search/SearchResultsList";

interface Result {
  term: string;
  tracks?: TrackOut[];
  error?: string;
}

export default function SearchPage() {
  const [q, setQ] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [attempt, setAttempt] = useState(0);
  const term = q.trim();

  useEffect(() => {
    if (term.length < 2) return;
    const ctrl = new AbortController();
    const timer = setTimeout(() => {
      searchTracks(term, ctrl.signal)
        .then((tracks) => setResult({ term, tracks }))
        .catch((e: unknown) => {
          if (e instanceof DOMException && e.name === "AbortError") return;
          setResult({ term, error: e instanceof Error ? e.message : "Erro inesperado." });
        });
    }, 300);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [term, attempt]);

  const idle = term.length < 2;
  const loading = !idle && result?.term !== term;

  return (
    <>
      <h1 className="page-title mb-3">Esse som tem cara de hit?</h1>
      <p className="lede mb-5">
        Escolha uma música e veja como o perfil de áudio dela se compara ao dos hits. É um filtro para priorizar a
        escuta, não uma decisão final.
      </p>
      <div className="mb-4" style={{ maxWidth: "36rem" }}>
        <SearchBar value={q} onChange={setQ} />
      </div>
      <div aria-live="polite">
        {idle && <p className="text-muted-r">Digite ao menos 2 letras.</p>}
        {loading && (
          <div className="d-grid gap-2" aria-label="Carregando">
            {[0, 1, 2].map((i) => (
              <div key={i} className="skeleton" style={{ height: "4rem" }} />
            ))}
          </div>
        )}
        {!idle && !loading && result?.error && (
          <ErrorPanel
            message={`${result.error} Verifique se a API está no ar.`}
            onRetry={() => {
              setResult(null);
              setAttempt((a) => a + 1);
            }}
          />
        )}
        {!idle &&
          !loading &&
          result?.tracks &&
          (result.tracks.length === 0 ? (
            <p className="text-muted-r">Nenhuma música encontrada para “{term}”. Tente outro termo.</p>
          ) : (
            <SearchResultsList tracks={result.tracks} />
          ))}
      </div>
    </>
  );
}
