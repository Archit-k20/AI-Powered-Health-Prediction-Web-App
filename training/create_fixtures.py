"""Generate clearly fictional report files for API and browser verification."""

from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
import pypdfium2 as pdfium

root = Path(__file__).resolve().parents[1] / "backend/tests/fixtures"
root.mkdir(exist_ok=True)
pdf = root / "fictional-text.pdf"
c = Canvas(str(pdf))
c.setFont("Helvetica", 18)
c.drawString(40, 790, "FICTIONAL REPORT - NO PATIENT DATA")
c.setFont("Helvetica", 13)
for i, line in enumerate(
    [
        "Hemoglobin 13.8 g/dL 12.0-16.0",
        "Glucose 92 mg/dL 70-99",
        "Platelets 250 10^3/uL 150-450",
        "TSH pending",
    ]
):
    c.drawString(40, 735 - i * 35, line)
c.save()
doc = pdfium.PdfDocument(str(pdf))
page = doc[0]
bitmap = page.render(scale=2)
image = bitmap.to_pil()
image.save(root / "fictional-image.png")
image.convert("RGB").save(root / "fictional-image.jpg")
c = Canvas(str(root / "fictional-scanned.pdf"))
c.drawImage(ImageReader(image), 0, 0, width=595, height=842)
c.save()
bitmap.close()
page.close()
doc.close()
(root / "malformed.pdf").write_bytes(b"%PDF-malformed fictional file")
print(root)
