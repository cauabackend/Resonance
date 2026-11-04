import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Link, NavLink, Route, Routes } from "react-router-dom";
import "./styles/theme.scss";
import "./styles/app.scss";
import ContrastToggle from "./components/ContrastToggle";
import BatchRankingPage from "./pages/BatchRankingPage";
import ModelCardPage from "./pages/ModelCardPage";
import SearchPage from "./pages/SearchPage";
import TrackDetailPage from "./pages/TrackDetailPage";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <div className="app-shell">
        <header className="mb-5 d-flex flex-wrap align-items-baseline gap-4">
          <Link to="/" className="wordmark fs-5">
            Resonance
          </Link>
          <nav aria-label="Principal" className="d-flex gap-3">
            <NavLink to="/" end className="nav-top">
              Busca
            </NavLink>
            <NavLink to="/ranking" className="nav-top">
              Ranking de demos
            </NavLink>
            <NavLink to="/model-card" className="nav-top">
              Model card
            </NavLink>
          </nav>
          <ContrastToggle />
        </header>
        <main>
          <Routes>
            <Route path="/" element={<SearchPage />} />
            <Route path="/ranking" element={<BatchRankingPage />} />
            <Route path="/model-card" element={<ModelCardPage />} />
            <Route path="/track/:id" element={<TrackDetailPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  </StrictMode>,
);
