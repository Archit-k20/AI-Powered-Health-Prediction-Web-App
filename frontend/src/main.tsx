import React, { lazy, Suspense, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  useLocation,
} from "react-router-dom";
import "@fontsource/inter/latin-400.css";
import "@fontsource/inter/latin-500.css";
import "./style.css";
import { Providers } from "./state";
import Story from "./Story";
const Workspace = lazy(() => import("./Workspace"));
const Information = lazy(() => import("./Information"));
export function Mark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 40 48" className="mark">
      <path d="M9 2C31 10-5 38 17 46M19 2C41 10 5 38 27 46M29 2C51 10 15 38 37 46" />
    </svg>
  );
}
export function Header() {
  return (
    <header className="site-header">
      <Link to="/" className="brand">
        <Mark />
        <span>
          The Shape of Health<small>by Archit</small>
        </span>
      </Link>
      <nav aria-label="Primary">
        <Link to="/methods">Method</Link>
        <Link to="/sources">Sources</Link>
        <Link to="/about">About</Link>
        <Link to="/workspace" className="demo-link">
          Try the demo <span>↗</span>
        </Link>
      </nav>
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
              Opening the exhibition…
            </main>
          }
        >
          <Routes>
            <Route path="/" element={<Story />} />
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
                  <h1>
                    This page has
                    <br />
                    changed shape.
                  </h1>
                  <Link to="/">Return to the exhibition ↗</Link>
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
