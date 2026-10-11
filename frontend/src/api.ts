export const API = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
export async function request<T>(
  path: string,
  options: RequestInit = {},
  timeout = 25000,
): Promise<T> {
  if (!API)
    throw new Error(
      "Live tools are not connected on this deployment. The fictional walkthroughs are ready to explore.",
    );
  const controller = new AbortController();
  const abort = () => controller.abort();
  options.signal?.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(abort, timeout);
  try {
    const response = await fetch(`${API}/api/v1${path}`, {
      ...options,
      signal: controller.signal,
    });
    const body = await response.json();
    if (!response.ok)
      throw new Error(
        body.error?.message ||
          "The service could not complete this request. Please try again.",
      );
    return body as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw new Error(
        "Request cancelled or timed out. The server may be starting. Try again, or open a fictional example.",
      );
    if (error instanceof TypeError)
      throw new Error(
        "The backend is unavailable. Try again shortly, or explore a fictional example.",
      );
    throw error;
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", abort);
  }
}
