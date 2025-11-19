#!/usr/bin/env python3
"""
Convert markdown to PDF using Python libraries.
"""

import re
import sys
from pathlib import Path

try:
    import markdown
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
except ImportError as e:
    print(f"Error: Missing required library. Please install: pip install markdown reportlab")
    print(f"Missing: {e}")
    sys.exit(1)


def markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF."""
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML first (for code extraction)
    html = markdown.markdown(md_content, extensions=['fenced_code', 'tables', 'toc'])
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#1a1a1a'),
        spaceAfter=12,
        alignment=TA_LEFT,
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=10,
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=HexColor('#34495e'),
        spaceAfter=6,
        spaceBefore=8,
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        leftIndent=20,
        rightIndent=20,
        backColor=HexColor('#f5f5f5'),
        borderColor=HexColor('#cccccc'),
        borderWidth=1,
        borderPadding=5,
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
    )
    
    bold_style = ParagraphStyle(
        'Bold',
        parent=normal_style,
        fontName='Helvetica-Bold',
    )
    
    # Parse markdown content
    story = []
    
    lines = md_content.split('\n')
    i = 0
    in_code_block = False
    code_lines = []
    code_language = ''
    first_section = True  # Track if we're at the first major section
    
    while i < len(lines):
        line = lines[i]
        
        # Handle code blocks
        if line.startswith('```'):
            if in_code_block:
                # End of code block
                if code_lines:
                    code_text = '\n'.join(code_lines)
                    # Escape HTML for code
                    code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(f'<font face="Courier" size="9">{code_text}</font>', code_style))
                    story.append(Spacer(1, 6))
                code_lines = []
                in_code_block = False
            else:
                # Start of code block
                code_language = line[3:].strip()
                in_code_block = True
            i += 1
            continue
        
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
        
        # Handle headers
        if line.startswith('# '):
            text = line[2:].strip()
            if text.startswith('**') and text.endswith('**'):
                text = text[2:-2]
            story.append(Paragraph(text, title_style if i < 5 else h1_style))
            story.append(Spacer(1, 6))
        elif line.startswith('## '):
            text = line[3:].strip()
            # Add page break before major sections (numbered sections like "1.", "2.", etc.)
            # But skip page break for the first section
            if text and text[0].isdigit():
                if not first_section:
                    story.append(PageBreak())
                first_section = False
            story.append(Paragraph(text, h1_style))
            story.append(Spacer(1, 6))
        elif line.startswith('### '):
            text = line[4:].strip()
            story.append(Paragraph(text, h2_style))
            story.append(Spacer(1, 4))
        elif line.startswith('#### '):
            text = line[5:].strip()
            story.append(Paragraph(text, h3_style))
            story.append(Spacer(1, 3))
        # Handle horizontal rules
        elif line.strip() == '---':
            story.append(Spacer(1, 12))
        # Handle empty lines
        elif line.strip() == '':
            story.append(Spacer(1, 6))
        # Handle inline code
        elif '`' in line:
            # Simple handling of inline code
            parts = re.split(r'`([^`]+)`', line)
            para_text = ''
            for j, part in enumerate(parts):
                if j % 2 == 0:
                    para_text += part.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                else:
                    para_text += f'<font face="Courier" size="9">{part}</font>'
            if para_text.strip():
                story.append(Paragraph(para_text, normal_style))
                story.append(Spacer(1, 6))
        # Handle regular text
        elif line.strip():
            # Clean up markdown formatting
            text = line
            # Remove markdown links but keep text
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            
            # Handle bold - use non-greedy regex to match **text** including colons, dashes, etc.
            # Convert **text** to <b>text</b> HTML format (ReportLab supports this)
            text = re.sub(r'\*\*([^\*]+?)\*\*', r'<b>\1</b>', text)
            
            # Handle italic (single asterisk, but not if it's part of bold)
            text = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'<i>\1</i>', text)
            
            # Escape HTML special characters BUT preserve <b>, </b>, <i>, </i> tags
            # Use placeholders to protect formatting tags
            text = text.replace('<b>', '___BOLD_START___')
            text = text.replace('</b>', '___BOLD_END___')
            text = text.replace('<i>', '___ITALIC_START___')
            text = text.replace('</i>', '___ITALIC_END___')
            
            # Escape HTML
            text = text.replace('&', '&amp;')
            text = text.replace('<', '&lt;')
            text = text.replace('>', '&gt;')
            
            # Restore formatting tags (they should be valid HTML for ReportLab)
            text = text.replace('___BOLD_START___', '<b>')
            text = text.replace('___BOLD_END___', '</b>')
            text = text.replace('___ITALIC_START___', '<i>')
            text = text.replace('___ITALIC_END___', '</i>')
            
            if text.strip():
                story.append(Paragraph(text, normal_style))
                story.append(Spacer(1, 6))
        
        i += 1
    
    # Build PDF
    doc.build(story)
    print(f"PDF created successfully: {pdf_file}")


if __name__ == '__main__':
    md_file = Path(__file__).parent / 'Model_Training_Guide.md'
    pdf_file = Path(__file__).parent / 'Model_Training_Guide.pdf'
    
    if not md_file.exists():
        print(f"Error: Markdown file not found: {md_file}")
        sys.exit(1)
    
    try:
        markdown_to_pdf(md_file, pdf_file)
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

