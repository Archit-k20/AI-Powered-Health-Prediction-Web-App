import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import Home from "./Home";
import Workspace from "./Workspace";
import { Providers } from "./state";
vi.mock("./api", () => ({ API: "", request: vi.fn() }));
afterEach(cleanup);
function setup() {
  render(
    <MemoryRouter>
      <Providers>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/workspace" element={<Workspace />} />
        </Routes>
      </Providers>
    </MemoryRouter>,
  );
  return userEvent.setup();
}
describe("simple healthcare entry points", () => {
  it("opens the report tool directly from the homepage", async () => {
    const user = setup();
    await user.click(screen.getByRole("link", { name: "Read a report" }));
    expect(screen.getByRole("tab", { name: "02Reports" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      screen.getByRole("heading", { name: "Understand what your report says" }),
    ).toBeVisible();
  });
  it("opens an offline example with technical explanations collapsed", async () => {
    const user = setup();
    await user.click(screen.getByRole("link", { name: /Cold-like symptoms/ }));
    expect(
      screen.getByText("Common cold", { selector: "strong" }),
    ).toBeVisible();
    const explanation = screen.getAllByText(/Largest SHAP contributions/)[0];
    expect(explanation).not.toBeVisible();
    await user.click(screen.getAllByText("Why these symptoms matched")[0]);
    expect(explanation).toBeVisible();
  });
});
