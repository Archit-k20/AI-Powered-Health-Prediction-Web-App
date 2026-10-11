import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  render,
  screen,
  within,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { Providers } from "./state";
import Workspace from "./Workspace";
import { request } from "./api";
import samples from "../../shared/samples.json";
vi.mock("./api", () => ({ API: "", request: vi.fn() }));
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
function setup(route = "/workspace") {
  render(
    <MemoryRouter initialEntries={[route]}>
      <Providers>
        <Workspace />
      </Providers>
    </MemoryRouter>,
  );
  return userEvent.setup();
}
describe("health workspace", () => {
  it("runs every fictional walkthrough without the backend", async () => {
    const user = setup();
    await user.click(
      screen.getByText("Just exploring? Try a fictional example"),
    );
    for (const s of samples) {
      await user.click(
        screen.getByRole("button", {
          name: new RegExp(
            (
              {
                cold: "Cold-like symptoms",
                migraine: "Headache symptoms",
                skin: "Skin symptoms",
              } as Record<string, string>
            )[s.id],
          ),
        }),
      );
      expect(
        screen.getByText("Fictional walkthrough · precomputed model output"),
      ).toBeInTheDocument();
      expect(
        screen.getByText(s.assessment.matches[0].label, { selector: "strong" }),
      ).toBeInTheDocument();
    }
    expect(request).not.toHaveBeenCalled();
  });
  it("resolves aliases, removes duplicates and requires confirmation for a suggested feature", async () => {
    const user = setup();
    const search = screen.getByLabelText("Search symptoms");
    await user.type(search, "shortness of breath");
    await user.click(
      screen.getByRole("button", { name: "Shortness of breath+" }),
    );
    await user.clear(search);
    await user.type(search, "fever");
    await user.click(screen.getByRole("button", { name: "High fever+" }));
    expect(screen.getByText("2 selected")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Remove High fever" }));
    expect(screen.getByText("1 selected")).toBeInTheDocument();
    await user.selectOptions(
      screen.getByLabelText("Have you also noticed a high fever?"),
      "Yes",
    );
    expect(
      screen.getByRole("button", { name: "See possible matches ↗" }),
    ).toBeDisabled();
    await user.click(
      screen.getByRole("checkbox", { name: /Confirm adding high fever/ }),
    );
    expect(
      screen.getByRole("button", { name: "See possible matches ↗" }),
    ).toBeEnabled();
    await user.selectOptions(
      screen.getByLabelText("Have you also noticed a high fever?"),
      "No",
    );
    expect(screen.getByText("1 selected")).toBeInTheDocument();
  });
  it("shows a truthful error when live assessment fails", async () => {
    vi.mocked(request).mockRejectedValue(
      new Error("The backend is unavailable."),
    );
    const user = setup("/workspace?sample=skin");
    await user.click(
      screen.getByRole("button", { name: "See possible matches ↗" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The backend is unavailable.",
    );
  });
  it("invalidates a sample result when its inputs change", async () => {
    const user = setup("/workspace?sample=skin");
    await user.click(screen.getByRole("button", { name: "Remove Itching" }));
    expect(
      screen.queryByText("Fictional walkthrough · precomputed model output"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText(/Your results will appear here/),
    ).toBeInTheDocument();
  });
  it("keeps fictional report values editable and reviewable", async () => {
    const user = setup("/workspace?tab=reports");
    await user.click(
      screen.getByRole("button", { name: "Use a fictional report instead ↗" }),
    );
    const value = screen.getByLabelText(
      "Hemoglobin value in Fictional lab report",
    );
    expect(
      screen.getAllByRole("checkbox", {
        name: "Needs review against the original",
      })[0],
    ).not.toBeChecked();
    await user.clear(value);
    await user.type(value, "14.2");
    expect(value).toHaveValue("14.2");
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes[0]).toBeChecked();
    await user.click(checkboxes[0]);
    expect(checkboxes[0]).not.toBeChecked();
    expect(
      screen.getByLabelText(
        "Platelets reference range in Fictional lab report",
      ),
    ).toBeInTheDocument();
  });
  it("processes a three-file queue sequentially and preserves partial failure", async () => {
    let inFlight = 0,
      maxFlight = 0;
    vi.mocked(request).mockImplementation(async (path, options) => {
      if (path != "/lab-reports/extract") throw new Error("Unexpected route");
      inFlight++;
      maxFlight = Math.max(maxFlight, inFlight);
      const file = (options!.body as FormData).get("file") as File;
      await new Promise((r) => setTimeout(r, 5));
      inFlight--;
      if (file.name === "bad.pdf") throw new Error("Unreadable report");
      return {
        findings: [],
        warnings: ["No rows found."],
        partial: true,
        pages: 1,
        notice: "Review.",
      } as never;
    });
    const user = setup("/workspace?tab=reports");
    const input = document.querySelector(
      "input[type=file]",
    ) as HTMLInputElement;
    await user.upload(input, [
      new File(["%PDF"], "first.pdf", { type: "application/pdf" }),
      new File(["%PDF"], "bad.pdf", { type: "application/pdf" }),
      new File(["%PDF"], "third.pdf", { type: "application/pdf" }),
    ]);
    await waitFor(() =>
      expect(
        screen.getAllByText("0 findings · review against your original"),
      ).toHaveLength(2),
    );
    expect(screen.getByText("Unreadable report")).toBeInTheDocument();
    expect(maxFlight).toBe(1);
    expect(request).toHaveBeenCalledTimes(3);
  });
  it("limits the file queue to three and supports keyboard tab navigation", async () => {
    const user = setup("/workspace?tab=reports");
    const input = document.querySelector(
      "input[type=file]",
    ) as HTMLInputElement;
    await user.upload(
      input,
      Array.from(
        { length: 4 },
        (_, i) => new File(["x"], `${i}.pdf`, { type: "application/pdf" }),
      ),
    );
    expect(screen.getByRole("alert")).toHaveTextContent("up to three");
    screen.getByRole("tab", { name: "02Reports" }).focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "03Summary" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      screen.getByRole("button", { name: "Download PDF summary ↓" }),
    ).toBeInTheDocument();
  });
  it("starts with common symptoms while keeping the full vocabulary searchable", async () => {
    const user = setup();
    const symptoms = screen.getByRole("generic", {
      name: "Available symptoms",
    });
    expect(within(symptoms).getAllByRole("button")).toHaveLength(12);
    await user.click(
      screen.getByRole("button", { name: "Browse all 131 symptoms" }),
    );
    expect(within(symptoms).getAllByRole("button")).toHaveLength(131);
    await user.click(
      screen.getByRole("button", { name: "Show common symptoms" }),
    );
    await user.type(screen.getByLabelText("Search symptoms"), "scaly skin");
    expect(
      within(symptoms).getByRole("button", { name: "Scaly skin+" }),
    ).toBeVisible();
  });
  it("does not persist health selections in local storage", async () => {
    const spy = vi.spyOn(Storage.prototype, "setItem");
    const user = setup("/workspace?sample=skin");
    await user.click(screen.getByRole("button", { name: "Clear session ↺" }));
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });
});
