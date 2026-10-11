import { it, expect, vi } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { Providers } from "./state";
import Sculpture from "./Sculpture";
vi.mock("three", () => ({
  WebGLRenderer: class {
    constructor() {
      throw new Error("WebGL unavailable");
    }
  },
}));
it("preserves an original static graphic when WebGL is unavailable", () => {
  const { container } = render(
    <Providers>
      <Sculpture shape={1} onShape={() => {}} />
    </Providers>,
  );
  expect(screen.getByRole("img")).toBeInTheDocument();
  expect(container.querySelectorAll(".static-sculpture i")).toHaveLength(22);
  cleanup();
});
