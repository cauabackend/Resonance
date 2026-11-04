import type { CSSProperties } from "react";
import { Link } from "react-router-dom";
import type { TrackOut } from "../../api/types";
import { Chip } from "../../components/ui/Badge";

export default function SearchResultsList({ tracks }: { tracks: TrackOut[] }) {
  return (
    <ul className="list-unstyled d-grid gap-2 mb-0" aria-label="Resultados da busca">
      {tracks.map((t, i) => (
        <li key={t.track_id}>
          <Link to={`/track/${t.track_id}`} className="track-row" style={{ "--i": i } as CSSProperties}>
            <span className="min-w-0">
              <span className="d-block fw-semibold text-truncate">{t.name}</span>
              <span className="d-block text-muted-r text-truncate">{t.artist}</span>
            </span>
            <span className="d-flex align-items-center gap-3 flex-shrink-0">
              <Chip>{t.features.track_genre}</Chip>
              <span className="pop-score" title="Popularidade no Spotify">
                {t.popularity}
              </span>
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
