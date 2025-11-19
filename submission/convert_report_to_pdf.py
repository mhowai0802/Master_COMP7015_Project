#!/usr/bin/env python3
"""
Convert Project_Report.md to PDF using weasyprint
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path

def convert_markdown_to_pdf(md_file='Project_Report.md', pdf_file='Project_Report.pdf'):
    """Convert markdown file to PDF."""
    
    # Read markdown file
    md_path = Path(md_file)
    if not md_path.exists():
        print(f"Error: {md_file} not found!")
        return False
    
    print(f"Reading {md_file}...")
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    print("Converting markdown to HTML...")
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite', 'extra']
    )
    
    # Add CSS styling for better PDF formatting
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 1in;
            }}
            body {{
                font-family: 'Times New Roman', Times, serif;
                font-size: 11pt;
                line-height: 1.5;
                color: #000;
            }}
            h1 {{
                font-size: 18pt;
                font-weight: bold;
                margin-top: 1em;
                margin-bottom: 0.5em;
                page-break-after: avoid;
            }}
            h2 {{
                font-size: 14pt;
                font-weight: bold;
                margin-top: 0.8em;
                margin-bottom: 0.4em;
                page-break-after: avoid;
            }}
            h3 {{
                font-size: 12pt;
                font-weight: bold;
                margin-top: 0.6em;
                margin-bottom: 0.3em;
            }}
            p {{
                margin-bottom: 0.5em;
                text-align: justify;
            }}
            ul, ol {{
                margin-bottom: 0.5em;
                padding-left: 2em;
            }}
            li {{
                margin-bottom: 0.3em;
            }}
            code {{
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 9pt;
                page-break-inside: avoid;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1em 0;
                page-break-inside: avoid;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            strong {{
                font-weight: bold;
            }}
            em {{
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF
    print("Generating PDF...")
    try:
        HTML(string=html_template).write_pdf(pdf_file)
        print(f"✅ Successfully created {pdf_file}")
        
        # Check file size
        pdf_path = Path(pdf_file)
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            print(f"   File size: {size_kb:.1f} KB")
        
        return True
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

if __name__ == '__main__':
    success = convert_markdown_to_pdf()
    if success:
        print("\n✅ Conversion complete! Please verify the PDF looks correct.")
        print("   The PDF should be ≤ 5 pages as required.")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")

