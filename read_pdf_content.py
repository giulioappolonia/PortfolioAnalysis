
from pypdf import PdfReader
import sys

# Set stdout to utf-8 just in case, though writing to file is safer
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = "C:/Users/ppln_/OneDrive/Documenti/Antigravity/PortfolioAnalisys-main/PortfolioAnalisys-main/An Alternative Portfolio Theory.pdf"
output_path = "C:/Users/ppln_/OneDrive/Documenti/Antigravity/PortfolioAnalisys-main/PortfolioAnalisys-main/pdf_content.txt"

try:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Successfully wrote {len(text)} characters to {output_path}")

except Exception as e:
    print(f"Error reading PDF: {e}")
