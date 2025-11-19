#!/usr/bin/env python3
"""
Convert Project_Report.md to PDF using Python
Tries multiple methods: markdown-pdf, xhtml2pdf, reportlab, etc.
"""

import sys
import os
from pathlib import Path

def method_weasyprint():
    """Method 1: Using weasyprint (requires system libraries)"""
    try:
        import markdown
        from weasyprint import HTML
        
        print("Trying weasyprint method...")
        
        md_file = Path('Project_Report.md')
        if not md_file.exists():
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'codehilite', 'extra']
        )
        
        # Add CSS styling
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
                h1 {{ font-size: 18pt; font-weight: bold; margin-top: 1em; margin-bottom: 0.5em; }}
                h2 {{ font-size: 14pt; font-weight: bold; margin-top: 0.8em; margin-bottom: 0.4em; }}
                h3 {{ font-size: 12pt; font-weight: bold; margin-top: 0.6em; margin-bottom: 0.3em; }}
                p {{ margin-bottom: 0.5em; text-align: justify; }}
                ul, ol {{ margin-bottom: 0.5em; padding-left: 2em; }}
                li {{ margin-bottom: 0.3em; }}
                code {{ font-family: 'Courier New', monospace; font-size: 10pt; background-color: #f4f4f4; padding: 2px 4px; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-size: 9pt; }}
                table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        HTML(string=html_template).write_pdf('Project_Report.pdf')
        print("✅ Success! Created Project_Report.pdf using weasyprint")
        return True
        
    except ImportError as e:
        print(f"❌ weasyprint not available: {e}")
        print("   Install with: pip install weasyprint")
        print("   Note: May require system libraries (libgobject, etc.)")
        return False
    except Exception as e:
        print(f"❌ weasyprint failed: {e}")
        return False


def method_xhtml2pdf():
    """Method 2: Using xhtml2pdf (pisa)"""
    try:
        import markdown
        from xhtml2pdf import pisa
        
        print("Trying xhtml2pdf method...")
        
        md_file = Path('Project_Report.md')
        if not md_file.exists():
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'extra']
        )
        
        # Add basic styling
        html_template = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Times; font-size: 11pt; margin: 1in; }}
                h1 {{ font-size: 18pt; font-weight: bold; }}
                h2 {{ font-size: 14pt; font-weight: bold; }}
                h3 {{ font-size: 12pt; font-weight: bold; }}
                p {{ margin-bottom: 0.5em; }}
                ul, ol {{ margin-left: 2em; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        with open('Project_Report.pdf', 'wb') as pdf_file:
            pisa.CreatePDF(html_template, dest=pdf_file)
        
        if Path('Project_Report.pdf').exists():
            print("✅ Success! Created Project_Report.pdf using xhtml2pdf")
            return True
        else:
            print("❌ xhtml2pdf failed to create PDF")
            return False
            
    except ImportError as e:
        print(f"❌ xhtml2pdf not available: {e}")
        print("   Install with: pip install xhtml2pdf")
        return False
    except Exception as e:
        print(f"❌ xhtml2pdf failed: {e}")
        return False


def method_pdfkit():
    """Method 3: Using pdfkit (requires wkhtmltopdf)"""
    try:
        import markdown
        import pdfkit
        
        print("Trying pdfkit method...")
        
        md_file = Path('Project_Report.md')
        if not md_file.exists():
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'extra']
        )
        
        # Add styling
        html_template = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Times; font-size: 11pt; margin: 1in; }}
                h1 {{ font-size: 18pt; font-weight: bold; }}
                h2 {{ font-size: 14pt; font-weight: bold; }}
                h3 {{ font-size: 12pt; font-weight: bold; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        pdfkit.from_string(html_template, 'Project_Report.pdf', options={
            'page-size': 'A4',
            'margin-top': '1in',
            'margin-right': '1in',
            'margin-bottom': '1in',
            'margin-left': '1in',
        })
        
        if Path('Project_Report.pdf').exists():
            print("✅ Success! Created Project_Report.pdf using pdfkit")
            return True
        else:
            print("❌ pdfkit failed to create PDF")
            return False
            
    except ImportError as e:
        print(f"❌ pdfkit not available: {e}")
        print("   Install with: pip install pdfkit")
        print("   Also need: brew install wkhtmltopdf")
        return False
    except Exception as e:
        print(f"❌ pdfkit failed: {e}")
        if "No wkhtmltopdf executable found" in str(e):
            print("   Install wkhtmltopdf: brew install wkhtmltopdf")
        return False


def method_reportlab():
    """Method 4: Using reportlab (basic text extraction)"""
    try:
        import markdown
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
        from reportlab.lib import colors
        import re
        
        print("Trying reportlab method...")
        
        md_file = Path('Project_Report.md')
        if not md_file.exists():
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML first
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Create PDF
        pdf = SimpleDocTemplate("Project_Report.pdf", pagesize=A4,
                               rightMargin=1*inch, leftMargin=1*inch,
                               topMargin=1*inch, bottomMargin=1*inch)
        styles = getSampleStyleSheet()
        
        # Custom styles with appropriate font sizes
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            leading=12,
            alignment=TA_JUSTIFY,
        )
        
        # Table cell style - left aligned, no justification
        table_cell_style = ParagraphStyle(
            'TableCellStyle',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            alignment=TA_LEFT,  # Left align, don't justify
            leftIndent=0,
            rightIndent=0,
        )
        
        # Table header style
        table_header_style = ParagraphStyle(
            'TableHeaderStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=12,
            alignment=TA_LEFT,  # Left align headers too
            leftIndent=0,
            rightIndent=0,
        )
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            leading=24,
            spaceAfter=12,
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=16,
            leading=20,
            spaceAfter=10,
        )
        
        heading3_style = ParagraphStyle(
            'CustomHeading3',
            parent=styles['Heading3'],
            fontSize=14,
            leading=18,
            spaceAfter=8,
        )
        
        heading4_style = ParagraphStyle(
            'CustomHeading4',
            parent=styles['Heading3'],
            fontSize=12,
            leading=16,
            spaceAfter=6,
        )
        
        story = []
        first_section = True  # Track first major section
        
        # Add title page with author name and student number
        title_page_style = ParagraphStyle(
            'TitlePage',
            parent=styles['Normal'],
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=30,
        )
        author_style = ParagraphStyle(
            'AuthorStyle',
            parent=styles['Normal'],
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
        
        # Extract title, author, and student number from first lines
        title_text = "AI Stocks: Multi-Model Stock Price Prediction System"
        author_text = "Mak Ho Wai Winson"
        student_text = "Student Number: 24465828"
        
        story.append(Spacer(1, 2*inch))  # Top margin
        story.append(Paragraph(title_text, title_page_style))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(author_text, author_style))
        story.append(Paragraph(student_text, author_style))
        story.append(Spacer(1, 1*inch))
        story.append(PageBreak())  # Start content on new page
        
        def format_text(text_line):
            """Helper function to format text with markdown conversion"""
            # Convert inline code `text` to monospace font (ReportLab format)
            text = re.sub(r'`([^`]+)`', r'<font name="Courier" size="10">\1</font>', text_line)
            # Convert bold and italic markdown to HTML
            text = re.sub(r'\*\*([^\*]+?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'<i>\1</i>', text)
            # Escape HTML but preserve formatting tags
            text = text.replace('<b>', '___BOLD_START___')
            text = text.replace('</b>', '___BOLD_END___')
            text = text.replace('<i>', '___ITALIC_START___')
            text = text.replace('</i>', '___ITALIC_END___')
            text = text.replace('<font name="Courier" size="10">', '___FONT_START___')
            text = text.replace('</font>', '___FONT_END___')
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            text = text.replace('___BOLD_START___', '<b>').replace('___BOLD_END___', '</b>')
            text = text.replace('___ITALIC_START___', '<i>').replace('___ITALIC_END___', '</i>')
            text = text.replace('___FONT_START___', '<font name="Courier" size="10">').replace('___FONT_END___', '</font>')
            return text
        
        def parse_table(lines, start_idx):
            """Parse a markdown table starting at start_idx. Returns (table_data, end_idx)"""
            table_data = []
            i = start_idx
            
            # Read header row
            if i >= len(lines) or not lines[i].strip().startswith('|'):
                return None, start_idx
            header_line = lines[i].strip()
            headers = [cell.strip() for cell in header_line.split('|')[1:-1]]  # Remove empty first/last
            table_data.append(headers)
            i += 1
            
            # Read separator row (should be |---|---|)
            if i >= len(lines) or not lines[i].strip().startswith('|'):
                return None, start_idx
            i += 1  # Skip separator
            
            # Read data rows
            while i < len(lines):
                line = lines[i].strip()
                if not line.startswith('|'):
                    break
                row = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
                if len(row) == len(headers):  # Only add if row matches header length
                    table_data.append(row)
                i += 1
            
            return table_data, i
        
        def create_table(table_data):
            """Create a ReportLab Table from table_data (list of rows)"""
            if not table_data or len(table_data) < 2:
                return None
            
            # Convert table data to Paragraph objects
            table_paragraphs = []
            for row_idx, row in enumerate(table_data):
                formatted_row = []
                for cell in row:
                    # Format cell text (handle bold, italic, etc.)
                    formatted_cell = format_text(cell)
                    # Use appropriate style: header style for first row, cell style for others
                    if row_idx == 0:
                        formatted_cell = f"<b>{formatted_cell}</b>"
                        formatted_row.append(Paragraph(formatted_cell, table_header_style))
                    else:
                        formatted_row.append(Paragraph(formatted_cell, table_cell_style))
                table_paragraphs.append(formatted_row)
            
            # Create table with dynamic column widths
            num_cols = len(table_paragraphs[0])
            # Calculate available width: A4 width (8.27") minus left and right margins (1" each)
            # This matches the text width exactly
            available_width = (8.27 - 2) * inch  # A4 width minus 2*1inch margins = 6.27 inches
            
            if num_cols == 3:
                # For 3 columns: Parameter (1.8"), Value (1.8"), Rationale (2.67")
                # Adjusted for better readability and to match text width
                col_widths = [1.8*inch, 1.8*inch, 2.67*inch]
            elif num_cols == 2:
                # For 2 columns: proportional split
                col_widths = [2.5*inch, 3.77*inch]
            else:
                # Equal width for other cases
                col_width = available_width / num_cols
                col_widths = [col_width] * num_cols
            
            # Ensure column widths sum to available width exactly
            total_width = sum(col_widths)
            if abs(total_width - available_width) > 0.01 * inch:
                # Adjust proportionally if there's a small discrepancy
                scale_factor = available_width / total_width
                col_widths = [w * scale_factor for w in col_widths]
            
            table = Table(table_paragraphs, colWidths=col_widths)
            
            # Style the table
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Header background
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header text color
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ])
            table.setStyle(table_style)
            return table
        
        # Parse markdown lines and group content by sections
        lines = md_content.split('\n')
        sections = []  # List of (heading_level, heading_text, content_list)
        current_section = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip author/student number lines (already on title page)
            # Check for various formats: **Author:**, Author:, etc.
            author_patterns = [
                '**Author:**', '**author:**', 'Author:', 'author:',
                '**Student Number:**', '**student number:**', 
                'Student Number:', 'student number:', 'Student No.:', 'student no.:',
                '24465828', 'Mak Ho Wai Winson', 'Winson Mak'
            ]
            if any(pattern.lower() in line.lower() for pattern in author_patterns) or line == '---':
                i += 1
                continue
            
            # Detect headings
            if line.startswith('# '):
                # Skip the main title heading (already on title page)
                # Check for exact match or variations
                title_text = line[2:].strip()
                title_variations = [
                    "AI Stocks: Multi-Model Stock Price Prediction System",
                    "AI Stocks",
                    "Multi-Model Stock Price Prediction System"
                ]
                if any(title_text == var or title_text.startswith(var.split(':')[0]) for var in title_variations):
                    i += 1
                    continue
                if current_section:
                    sections.append(current_section)
                current_section = (1, title_text, [])
                i += 1
                continue
            elif line.startswith('## '):
                if current_section:
                    sections.append(current_section)
                current_section = (2, line[3:].strip(), [])
                i += 1
                continue
            elif line.startswith('### '):
                if current_section:
                    sections.append(current_section)
                current_section = (3, line[4:].strip(), [])
                i += 1
                continue
            elif line.startswith('#### '):
                if current_section:
                    sections.append(current_section)
                current_section = (4, line[5:].strip(), [])
                i += 1
                continue
            
            # Detect markdown tables
            if line.startswith('|'):
                table_data, end_idx = parse_table(lines, i)
                if table_data and len(table_data) > 1:  # Has header + at least one row
                    if current_section:
                        current_section[2].append(('table', table_data))
                    else:
                        # Handle table before first heading
                        table = create_table(table_data)
                        if table:
                            story.append(Spacer(1, 0.1*inch))
                            story.append(table)
                            story.append(Spacer(1, 0.2*inch))
                    i = end_idx
                    continue
            
            # Add content to current section
            if current_section:
                if not line:
                    current_section[2].append(('spacer', None))
                elif line.startswith('- ') or line.startswith('* '):
                    text = format_text(line[2:].strip())
                    current_section[2].append(('bullet', text))
                else:
                    # Check if this is "Sentiment Models:" text that needs a page break
                    if '**Sentiment Models**' in line or 'Sentiment Models:' in line:
                        current_section[2].append(('pagebreak', None))
                    text = format_text(line)
                    current_section[2].append(('text', text))
            else:
                # Content before first heading
                if not line:
                    story.append(Spacer(1, 0.2*inch))
                elif line.startswith('- ') or line.startswith('* '):
                    text = format_text(line[2:].strip())
                    story.append(Paragraph(f"• {text}", normal_style))
                else:
                    text = format_text(line)
                    story.append(Paragraph(text, normal_style))
            i += 1
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        # Process sections and add to story with KeepTogether
        for idx, (heading_level, heading_text, content_list) in enumerate(sections):
            # Add page breaks BEFORE the section (outside KeepTogether)
            if heading_level == 2:
                # Add page break before major sections (numbered sections like "1.", "2.", etc.)
                if heading_text and heading_text[0].isdigit():
                    if not first_section:
                        story.append(PageBreak())
                    first_section = False
                    # Special case: 2.2 Feature Engineering should start on new page
                    if '2.2' in heading_text or 'Feature Engineering' in heading_text:
                        story.append(PageBreak())
            elif heading_level == 3:
                # Add page break before "2.3.1 Detailed Model Configuration Rationale"
                if '2.3.1' in heading_text or 'Detailed Model Configuration Rationale' in heading_text:
                    story.append(PageBreak())
            elif heading_level == 4:
                # Add page break before major model configuration sections
                if 'Model Configuration' in heading_text:
                    story.append(PageBreak())
            
            # Determine heading style
            if heading_level == 1:
                heading_para = Paragraph(heading_text, title_style)
            elif heading_level == 2:
                heading_para = Paragraph(heading_text, heading2_style)
            elif heading_level == 3:
                heading_para = Paragraph(heading_text, heading3_style)
            elif heading_level == 4:
                heading_para = Paragraph(heading_text, heading4_style)
            
            # Build section content
            # Check if we need to split at "Sentiment Models:"
            split_index = None
            for idx, (content_type, content_text) in enumerate(content_list):
                if content_type == 'pagebreak':
                    split_index = idx
                    break
            
            if split_index is not None:
                # Split the section: content before pagebreak, then pagebreak, then content after
                content_before = content_list[:split_index]
                content_after = content_list[split_index+1:]  # Skip the pagebreak marker
                
                # Build first part with heading
                section_content_before = [heading_para]
                for content_type, content_text in content_before:
                    if content_type == 'spacer':
                        section_content_before.append(Spacer(1, 0.2*inch))
                    elif content_type == 'bullet':
                        section_content_before.append(Paragraph(f"• {content_text}", normal_style))
                    elif content_type == 'text':
                        section_content_before.append(Paragraph(content_text, normal_style))
                    elif content_type == 'table':
                        table = create_table(content_text)
                        if table:
                            section_content_before.append(Spacer(1, 0.1*inch))
                            section_content_before.append(table)
                            section_content_before.append(Spacer(1, 0.2*inch))
                
                # Add first part (keep together)
                if section_content_before:
                    story.append(KeepTogether(section_content_before))
                
                # Add page break
                story.append(PageBreak())
                
                # Build second part (content after "Sentiment Models:")
                section_content_after = []
                for content_type, content_text in content_after:
                    if content_type == 'spacer':
                        section_content_after.append(Spacer(1, 0.2*inch))
                    elif content_type == 'bullet':
                        section_content_after.append(Paragraph(f"• {content_text}", normal_style))
                    elif content_type == 'text':
                        section_content_after.append(Paragraph(content_text, normal_style))
                    elif content_type == 'table':
                        table = create_table(content_text)
                        if table:
                            section_content_after.append(Spacer(1, 0.1*inch))
                            section_content_after.append(table)
                            section_content_after.append(Spacer(1, 0.2*inch))
                
                # Add second part (keep together)
                if section_content_after:
                    story.append(KeepTogether(section_content_after))
            else:
                # No split needed, process normally
                section_content = [heading_para]
                for content_type, content_text in content_list:
                    if content_type == 'spacer':
                        section_content.append(Spacer(1, 0.2*inch))
                    elif content_type == 'bullet':
                        section_content.append(Paragraph(f"• {content_text}", normal_style))
                    elif content_type == 'text':
                        section_content.append(Paragraph(content_text, normal_style))
                    elif content_type == 'table':
                        table = create_table(content_text)
                        if table:
                            section_content.append(Spacer(1, 0.1*inch))
                            section_content.append(table)
                            section_content.append(Spacer(1, 0.2*inch))
                
                # Wrap section in KeepTogether to prevent splitting across pages
                # KeepTogether ensures that if the section doesn't fit on current page,
                # it will start on a new page. This is especially important for Configuration sections.
                story.append(KeepTogether(section_content))
        
        pdf.build(story)
        
        if Path('Project_Report.pdf').exists():
            print("✅ Success! Created Project_Report.pdf using reportlab")
            return True
        else:
            print("❌ reportlab failed to create PDF")
            return False
            
    except ImportError as e:
        print(f"❌ reportlab not available: {e}")
        print("   Install with: pip install reportlab")
        return False
    except Exception as e:
        print(f"❌ reportlab failed: {e}")
        return False


def main():
    """Try all methods in order"""
    print("=" * 60)
    print("Python PDF Conversion - Trying Multiple Methods")
    print("=" * 60)
    print()
    
    methods = [
        ("xhtml2pdf", method_xhtml2pdf),
        ("pdfkit", method_pdfkit),
        ("reportlab", method_reportlab),
        ("weasyprint", method_weasyprint),
    ]
    
    for name, method in methods:
        print(f"\n--- Trying {name} ---")
        if method():
            print(f"\n✅ Successfully converted using {name}!")
            print(f"📄 Output: Project_Report.pdf")
            return True
        print()
    
    print("=" * 60)
    print("❌ All Python methods failed!")
    print("=" * 60)
    print("\n💡 Installation options:")
    print("\n   Option 1: Install xhtml2pdf (easiest)")
    print("   → pip install xhtml2pdf markdown")
    print("   → python3 convert_with_python.py")
    print()
    print("   Option 2: Install pdfkit")
    print("   → pip install pdfkit markdown")
    print("   → brew install wkhtmltopdf")
    print("   → python3 convert_with_python.py")
    print()
    print("   Option 3: Install reportlab")
    print("   → pip install reportlab markdown")
    print("   → python3 convert_with_python.py")
    print()
    print("   Option 4: Use online converter (no installation)")
    print("   → https://www.markdowntopdf.com/")
    print()
    
    return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

