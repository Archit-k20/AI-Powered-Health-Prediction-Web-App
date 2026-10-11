import { useEffect, useRef, useState } from "react";
import {
  getDocument,
  GlobalWorkerOptions,
  type PDFDocumentLoadingTask,
  type RenderTask,
} from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = workerUrl;
export default function PdfPreview({
  file,
  page,
  highlight,
}: {
  file: File;
  page: number;
  highlight?: number[];
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState("Reading original page…");
  useEffect(() => {
    let active = true;
    let loading: PDFDocumentLoadingTask | undefined;
    let rendering: RenderTask | undefined;
    setStatus("Reading original page…");
    void (async () => {
      try {
        const data = new Uint8Array(await file.arrayBuffer());
        if (!active) return;
        loading = getDocument({
          data,
          verbosity: 0,
          useSystemFonts: true,
          useWasm: false,
          stopAtErrors: true,
          maxImageSize: 20_000_000,
        });
        const document = await loading.promise;
        const source = await document.getPage(page);
        if (!active || !canvas.current) return;
        const base = source.getViewport({ scale: 1 });
        const viewport = source.getViewport({
          scale: Math.min(
            1400 / base.width,
            Math.sqrt(4_000_000 / (base.width * base.height)),
          ),
        });
        canvas.current.width = Math.ceil(viewport.width);
        canvas.current.height = Math.ceil(viewport.height);
        rendering = source.render({ canvas: canvas.current, viewport });
        await rendering.promise;
        if (active) setStatus("");
      } catch {
        if (active)
          setStatus(
            "Preview unavailable. Open the original file using the link below; the extracted text is still available.",
          );
      }
    })();
    return () => {
      active = false;
      rendering?.cancel();
      void loading?.destroy().catch(() => {});
    };
  }, [file, page]);
  return (
    <div className="pdf-preview">
      {status && <p role="status">{status}</p>}
      <div className="pdf-page" hidden={!!status}>
        <canvas
          ref={canvas}
          role="img"
          aria-label={`Original report page ${page}. A reading transcript is available below.`}
        />
        {highlight && (
          <div
            className="evidence-highlight"
            aria-hidden="true"
            style={{
              left: `${highlight[0] * 100}%`,
              top: `${highlight[1] * 100}%`,
              width: `${highlight[2] * 100}%`,
              height: `${highlight[3] * 100}%`,
            }}
          />
        )}
      </div>
    </div>
  );
}
