import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })),
});

window.scrollTo = vi.fn();
