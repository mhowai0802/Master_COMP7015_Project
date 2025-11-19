#!/usr/bin/env python3
"""
Merge Model_Technical_Documentation.pdf into Project_Report.pdf as an appendix.
"""

import sys
from pathlib import Path

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("Error: PyPDF2 is required. Install with: pip install PyPDF2")
    sys.exit(1)


def merge_pdfs(main_pdf_path, appendix_pdf_path, output_pdf_path):
    """Merge appendix PDF into main PDF."""
    
    main_pdf = Path(main_pdf_path)
    appendix_pdf = Path(appendix_pdf_path)
    output_pdf = Path(output_pdf_path)
    
    if not main_pdf.exists():
        print(f"Error: Main PDF not found: {main_pdf}")
        sys.exit(1)
    
    if not appendix_pdf.exists():
        print(f"Error: Appendix PDF not found: {appendix_pdf}")
        sys.exit(1)
    
    # Create PDF writer
    writer = PdfWriter()
    
    # Read main PDF
    print(f"Reading main PDF: {main_pdf}")
    main_reader = PdfReader(str(main_pdf))
    for page in main_reader.pages:
        writer.add_page(page)
    
    # Read appendix PDF
    print(f"Reading appendix PDF: {appendix_pdf}")
    appendix_reader = PdfReader(str(appendix_pdf))
    for page in appendix_reader.pages:
        writer.add_page(page)
    
    # Write merged PDF
    print(f"Writing merged PDF: {output_pdf}")
    with open(output_pdf, 'wb') as output_file:
        writer.write(output_file)
    
    print(f"✅ Successfully merged PDFs!")
    print(f"   Main PDF: {len(main_reader.pages)} pages")
    print(f"   Appendix PDF: {len(appendix_reader.pages)} pages")
    print(f"   Total pages: {len(main_reader.pages) + len(appendix_reader.pages)} pages")


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parent.parent
    main_pdf = project_root / 'submission' / 'Project_Report.pdf'
    appendix_pdf = project_root / 'docs' / 'Model_Technical_Documentation.pdf'
    output_pdf = project_root / 'submission' / 'Project_Report.pdf'
    
    try:
        merge_pdfs(main_pdf, appendix_pdf, output_pdf)
    except Exception as e:
        print(f"Error merging PDFs: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

