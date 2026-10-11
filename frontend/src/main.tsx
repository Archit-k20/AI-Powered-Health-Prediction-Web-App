import React, { lazy, Suspense, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  NavLink,
  useLocation,
} from "react-router-dom";
import "@fontsource/inter/latin-400.css";
import "@fontsource/inter/latin-500.css";
import "./style.css";
import { Providers } from "./state";
import Home from "./Home";
import { Icon } from "./Icon";
const Workspace = lazy(() => import("./Workspace"));
const Information = lazy(() => import("./Information"));
export function Header() {
  return (
    <header className="site-header">
      <Link to="/" className="brand" aria-label="The Shape of Health home">
        <span className="brand-mark">
          <Icon name="heart" />
        </span>
        <span>
          The Shape of Health<small>by Archit</small>
        </span>
      </Link>
      <nav aria-label="Primary">
        <NavLink to="/" end>
          Home
        </NavLink>
        <Link to="/workspace">Symptoms</Link>
        <Link to="/workspace?tab=reports">Reports</Link>
        <NavLink to="/methods">How it works</NavLink>
      </nav>
      <Link to="/workspace" className="header-action">
        Get started <Icon name="arrow" />
      </Link>
    </header>
  );
}
function ResetScroll() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
    document.title =
      pathname === "/"
        ? "The Shape of Health — by Archit"
        : `${pathname === "/workspace" ? "The workspace" : pathname.slice(1)[0].toUpperCase() + pathname.slice(2)} — The Shape of Health`;
  }, [pathname]);
  return null;
}
function App() {
  return (
    <BrowserRouter>
      <Providers>
        <ResetScroll />
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <Header />
        <Suspense
          fallback={
            <main id="main" className="loading">
              Opening your health tools…
            </main>
          }
        >
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/workspace" element={<Workspace />} />
            {["methods", "sources", "about", "privacy"].map((page) => (
              <Route
                key={page}
                path={`/${page}`}
                element={<Information page={page} />}
              />
            ))}
            <Route
              path="*"
              element={
                <main id="main" className="information">
                  <h1>We couldn’t find this page.</h1>
                  <Link to="/">Return home</Link>
                </main>
              }
            />
          </Routes>
        </Suspense>
      </Providers>
    </BrowserRouter>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
