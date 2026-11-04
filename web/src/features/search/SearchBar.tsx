export default function SearchBar({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <label htmlFor="q" className="form-label text-muted-r">
        Nome da música ou do artista
      </label>
      <input
        id="q"
        type="search"
        className="form-control form-control-lg"
        placeholder="Ex.: Bohemian Rhapsody"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        autoComplete="off"
        autoFocus
      />
    </div>
  );
}
