import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { cleanup, render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { Providers } from "./state";
import Story from "./Story";
vi.mock("./Sculpture", () => ({
  default: () => <div role="img" aria-label="Sculpture" />,
}));
vi.mock("gsap", () => ({
  gsap: { registerPlugin: vi.fn(), context: () => ({ revert: vi.fn() }) },
}));
vi.mock("gsap/ScrollTrigger", () => ({ ScrollTrigger: {} }));
beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  HTMLDialogElement.prototype.showModal = function () {
    this.setAttribute("open", "");
  };
  HTMLDialogElement.prototype.close = function () {
    this.removeAttribute("open");
  };
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  localStorage.clear();
});
describe("exhibition controls", () => {
  it("opens all eight chapters and navigates without motion", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <Providers>
          <Story />
        </Providers>
      </MemoryRouter>,
    );
    await user.click(screen.getByRole("switch", { name: "Motion on" }));
    await user.click(screen.getByRole("button", { name: /Chapters/ }));
    expect(screen.getByRole("dialog")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "07Limits↗" }));
    expect(Element.prototype.scrollIntoView).toHaveBeenCalledWith({
      behavior: "instant",
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(document.documentElement.dataset.motion).toBe("off");
    expect(screen.getByRole("switch", { name: "╎ Sound off" })).toHaveAttribute(
      "aria-checked",
      "false",
    );
  });
  it("reshapes via the accessible slider and reveals an editable report", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <Providers>
          <Story />
        </Providers>
      </MemoryRouter>,
    );
    fireEvent.change(
      screen.getByRole("slider", { name: "Reshape the sculpture" }),
      { target: { value: "2" } },
    );
    expect(
      screen.getByText("A different perspective changes what we notice."),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Reveal the structure ↗" }),
    );
    expect(screen.getByLabelText("Fictional hemoglobin value")).toHaveValue(
      "13.8",
    );
  });
});
