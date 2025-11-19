#!/usr/bin/env python3
"""
Convert docs/Model_Training_Guide.md to PDF with improved layout
"""

import sys
import os
from pathlib import Path

def convert_training_guide():
    """Convert Model Training Guide markdown to PDF with better layout"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.lib import colors
        from reportlab.platypus.flowables import HRFlowable
        import re
        
        print("Converting docs/Model_Training_Guide.md to PDF with improved layout...")
        
        md_file = Path('docs/Model_Training_Guide.md')
        if not md_file.exists():
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Create PDF with better margins
        pdf = SimpleDocTemplate(
            "docs/Model_Training_Guide.pdf",
            pagesize=A4,
            rightMargin=1*inch,
            leftMargin=1*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        styles = getSampleStyleSheet()
        
        # Improved styles
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=8,
            leftIndent=0,
            rightIndent=0,
        )
        
        code_style = ParagraphStyle(
            'Code',
            parent=styles['Code'],
            fontSize=8.5,
            leading=11,
            fontName='Courier',
            leftIndent=12,
            rightIndent=12,
            backColor=colors.Color(0.96, 0.96, 0.96),
            borderColor=colors.Color(0.8, 0.8, 0.8),
            borderWidth=1,
            borderPadding=6,
            spaceAfter=12,
            spaceBefore=8,
        )
        
        code_inline_style = ParagraphStyle(
            'CodeInline',
            parent=styles['Normal'],
            fontSize=9,
            fontName='Courier',
            backColor=colors.Color(0.96, 0.96, 0.96),
            leftIndent=2,
            rightIndent=2,
        )
        
        # Main title - much larger and more prominent
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=28,
            leading=34,
            spaceAfter=12,
            spaceBefore=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a1a'),
            fontName='Helvetica-Bold',
        )
        
        # Subtitle - clearly visible but distinct
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=14,
            leading=18,
            spaceAfter=24,
            spaceBefore=0,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a5568'),
            fontName='Helvetica',
        )
        
        # H1 - Section headings (e.g., "Model 1: Baseline Model")
        heading1_style = ParagraphStyle(
            'Heading1',
            parent=styles['Heading1'],
            fontSize=20,
            leading=26,
            spaceAfter=14,
            spaceBefore=24,
            textColor=colors.HexColor('#1a202c'),
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderColor=colors.HexColor('#e2e8f0'),
            borderPadding=0,
        )
        
        # H2 - Major subsections
        heading2_style = ParagraphStyle(
            'Heading2',
            parent=styles['Heading2'],
            fontSize=16,
            leading=22,
            spaceAfter=10,
            spaceBefore=18,
            textColor=colors.HexColor('#2d3748'),
            fontName='Helvetica-Bold',
        )
        
        # H3 - Minor subsections
        heading3_style = ParagraphStyle(
            'Heading3',
            parent=styles['Heading3'],
            fontSize=13,
            leading=18,
            spaceAfter=8,
            spaceBefore=14,
            textColor=colors.HexColor('#4a5568'),
            fontName='Helvetica-Bold',
        )
        
        # H4 - Smallest headings
        heading4_style = ParagraphStyle(
            'Heading4',
            parent=styles['Heading3'],
            fontSize=11,
            leading=15,
            spaceAfter=6,
            spaceBefore=10,
            textColor=colors.HexColor('#718096'),
            fontName='Helvetica-Bold',
        )
        
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=normal_style,
            leftIndent=24,
            bulletIndent=12,
            spaceAfter=5,
        )
        
        numbered_style = ParagraphStyle(
            'Numbered',
            parent=normal_style,
            leftIndent=24,
            bulletIndent=12,
            spaceAfter=5,
        )
        
        story = []
        last_was_heading = False
        is_first_heading = True  # Track if this is the main title
        
        def format_text(text_line, preserve_newlines=False):
            """Helper function to format text with markdown conversion"""
            if not text_line:
                return ""
            
            # Handle emojis and special characters first
            text = text_line
            text = text.replace('✅', '✓')
            text = text.replace('⚠️', '⚠')
            text = text.replace('📋', '')
            text = text.replace('🎯', '')
            text = text.replace('🏗️', '')
            text = text.replace('🚀', '')
            text = text.replace('💻', '')
            text = text.replace('📊', '')
            text = text.replace('📝', '')
            
            # First, protect code blocks from HTML escaping
            code_blocks = []
            def replace_code(match):
                code_blocks.append(match.group(1))
                return f"__CODE_BLOCK_{len(code_blocks)-1}__"
            
            text = re.sub(r'`([^`]+)`', replace_code, text)
            
            # Convert markdown links [text](url) to just text
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            
            # Convert bold markdown
            text = re.sub(r'\*\*([^\*]+?)\*\*', r'<b>\1</b>', text)
            
            # Convert italic markdown (but not bold)
            text = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'<i>\1</i>', text)
            
            # Protect formatting tags before escaping
            text = text.replace('<b>', '___BOLD_START___')
            text = text.replace('</b>', '___BOLD_END___')
            text = text.replace('<i>', '___ITALIC_START___')
            text = text.replace('</i>', '___ITALIC_END___')
            
            # Escape HTML characters
            text = text.replace('&', '&amp;')
            text = text.replace('<', '&lt;')
            text = text.replace('>', '&gt;')
            
            # Restore formatting tags (these are valid HTML for ReportLab Paragraph)
            # The placeholders don't have angle brackets, so restore them directly
            text = text.replace('___BOLD_START___', '<b>')
            text = text.replace('___BOLD_END___', '</b>')
            text = text.replace('___ITALIC_START___', '<i>')
            text = text.replace('___ITALIC_END___', '</i>')
            
            # Restore code blocks with proper font tag (in reverse order to avoid conflicts)
            # Font tags are created AFTER escaping, so they remain unescaped (which is correct for ReportLab)
            for idx in range(len(code_blocks) - 1, -1, -1):
                code = code_blocks[idx]
                # Escape the code content itself
                escaped_code = code.replace('&amp;', '&')  # Unescape if already escaped
                escaped_code = escaped_code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Create font tag - this HTML will NOT be escaped (correct for ReportLab)
                code_font_tag = f'<font name="Courier" size="9" color="#c7254e">{escaped_code}</font>'
                text = text.replace(f"__CODE_BLOCK_{idx}__", code_font_tag)
            
            if preserve_newlines:
                text = text.replace('\n', '<br/>')
            
            return text
        
        def parse_code_block(lines, start_idx):
            """Parse a code block starting at start_idx. Returns (code_text, end_idx)"""
            code_lines = []
            i = start_idx + 1  # Skip opening ```
            while i < len(lines):
                line = lines[i]
                if line.strip().startswith('```'):
                    return '\n'.join(code_lines), i + 1
                code_lines.append(line.rstrip())
                i += 1
            return '\n'.join(code_lines), i
        
        def parse_table(lines, start_idx):
            """Parse a markdown table starting at start_idx. Returns (table_data, end_idx)"""
            table_data = []
            i = start_idx
            
            # Read header row
            if i >= len(lines) or not lines[i].strip().startswith('|'):
                return None, start_idx
            header_line = lines[i].strip()
            headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
            table_data.append(headers)
            i += 1
            
            # Read separator row
            if i >= len(lines) or not lines[i].strip().startswith('|'):
                return None, start_idx
            i += 1  # Skip separator
            
            # Read data rows
            while i < len(lines):
                line = lines[i].strip()
                if not line.startswith('|'):
                    break
                row = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(row) == len(headers):
                    table_data.append(row)
                i += 1
            
            return table_data, i
        
        def create_table(table_data):
            """Create a ReportLab Table from table_data with better formatting"""
            if not table_data or len(table_data) < 2:
                return None
            
            num_cols = len(table_data[0])
            num_rows = len(table_data)
            
            # Calculate available width
            available_width = (8.27 - 2) * inch  # A4 width minus margins
            
            # Smart column width allocation
            if num_cols == 2:
                col_widths = [2.5*inch, 3.77*inch]
            elif num_cols == 3:
                col_widths = [1.8*inch, 1.8*inch, 2.67*inch]
            elif num_cols == 4:
                col_width = available_width / 4
                col_widths = [col_width] * 4
            elif num_cols == 5:
                col_width = available_width / 5
                col_widths = [col_width] * 5
            elif num_cols == 6:
                col_width = available_width / 6
                col_widths = [col_width] * 6
            elif num_cols == 7:
                col_width = available_width / 7
                col_widths = [col_width] * 7
            elif num_cols == 8:
                col_width = available_width / 8
                col_widths = [col_width] * 8
            else:
                col_width = available_width / num_cols
                col_widths = [col_width] * num_cols
            
            table_paragraphs = []
            for row_idx, row in enumerate(table_data):
                formatted_row = []
                for col_idx, cell in enumerate(row):
                    # Format the cell content - this handles markdown like **bold**, `code`, etc.
                    formatted_cell = format_text(cell)
                    
                    if row_idx == 0:
                        # Header row - ensure it's bold
                        # Check if already has bold tags
                        if '<b>' not in formatted_cell:
                            formatted_cell = f"<b>{formatted_cell}</b>"
                        cell_style = ParagraphStyle(
                            'TableHeader',
                            parent=styles['Normal'],
                            fontSize=10,
                            leading=13,
                            alignment=TA_LEFT,
                            textColor=colors.white,
                            fontName='Helvetica-Bold',
                        )
                    else:
                        # Data row - allow wrapping for long content
                        cell_style = ParagraphStyle(
                            'TableCell',
                            parent=styles['Normal'],
                            fontSize=9,
                            leading=12,
                            alignment=TA_LEFT,
                            wordWrap='CJK',  # Enable word wrapping
                        )
                    
                    # Create paragraph with formatted content
                    try:
                        para = Paragraph(formatted_cell, cell_style)
                        formatted_row.append(para)
                    except Exception as e:
                        # Fallback: use plain text if formatting fails
                        print(f"Warning: Failed to format table cell '{cell[:50]}...': {e}")
                        formatted_row.append(Paragraph(cell, cell_style))
                table_paragraphs.append(formatted_row)
            
            table = Table(table_paragraphs, colWidths=col_widths, repeatRows=1)
            
            # Better table styling
            table_style = TableStyle([
                # Header styling - darker, more prominent
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                
                # Data row styling
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                
                # Alternating row colors for better readability
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
                
                # Borders - clearer separation
                ('GRID', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e0')),
                ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#1a202c')),  # Thicker line below header
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ])
            table.setStyle(table_style)
            return table
        
        # Parse markdown
        lines = md_content.split('\n')
        i = 0
        in_code_block = False
        code_language = None
        last_was_heading = False
        model_section_count = 0
        
        # Expected sections in order (for verification)
        expected_sections = [
            "Table of Contents",
            "Quick Start",
            "Model 1: Baseline Model",
            "Model 2: MLP Model",
            "Model 3: Transformer Model",
            "Model 4: Sentiment LSTM Model",
            "Model 5: Sentiment BERT Model",
            "Model Comparison"
        ]
        section_order = []
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check for code blocks
            if stripped.startswith('```'):
                if in_code_block:
                    # End of code block
                    in_code_block = False
                    code_language = None
                    # Spacing is already added after code block, so skip here
                    i += 1
                    continue
                else:
                    # Start of code block
                    in_code_block = True
                    code_language = stripped[3:].strip()
                    code_text, end_idx = parse_code_block(lines, i)
                    
                    # Use Paragraph with monospace font for code display (allows better page splitting)
                    # Add spacing before code block
                    story.append(Spacer(1, 0.1*inch))
                    # Escape code text and convert newlines to <br/>
                    escaped_code = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    # Convert newlines to HTML breaks
                    code_html = escaped_code.replace('\n', '<br/>')
                    # Wrap in monospace font tag
                    code_para = Paragraph(
                        f'<font name="Courier" size="8.5" color="#333333">{code_html}</font>',
                        code_style
                    )
                    story.append(code_para)
                    story.append(Spacer(1, 0.1*inch))
                    i = end_idx
                    continue
            
            if in_code_block:
                i += 1
                continue
            
            # Check for headings
            # IMPORTANT: Only process headings if we're NOT in a code block
            # Also, avoid treating code comments (lines starting with # followed by code-like content) as headings
            if line.startswith('# '):
                # Check if this looks like a code comment rather than a heading
                # Code comments often have patterns like "# [", "# '", "# \"", or are very short
                heading_text = line[2:].strip()
                is_likely_code_comment = (
                    heading_text.startswith('[') or 
                    heading_text.startswith("'") or 
                    heading_text.startswith('"') or
                    (len(heading_text) < 5 and not heading_text[0].isupper()) or
                    heading_text.startswith('#')  # Double comment
                )
                
                if is_likely_code_comment:
                    # Treat as regular text, not a heading
                    para = Paragraph(format_text(line), normal_style)
                    story.append(para)
                    last_was_heading = False
                else:
                    # Main title - first H1 gets special treatment
                    if is_first_heading:
                        is_first_heading = False
                        title_text = format_text(heading_text)
                        story.append(Paragraph(title_text, title_style))
                        # Check if next line is subtitle (bold text on its own line)
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if next_line.startswith('**') and next_line.endswith('**'):
                                # Extract subtitle text (remove ** markers)
                                subtitle_text = next_line.replace('**', '').strip()
                                story.append(Paragraph(format_text(subtitle_text), subtitle_style))
                                i += 1  # Skip the subtitle line
                    else:
                        # Regular H1 sections - add page break before new model sections
                        if model_section_count > 0:
                            story.append(PageBreak())
                        model_section_count += 1
                        story.append(Paragraph(format_text(heading_text), heading1_style))
                    last_was_heading = True
            elif line.startswith('## '):
                section_title = line[3:].strip()
                # Track section order for verification - track ALL matching H2 sections
                if section_title in expected_sections:
                    # Only add if not already in list (avoid duplicates)
                    if section_title not in section_order:
                        section_order.append(section_title)
                if last_was_heading:
                    story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(format_text(section_title), heading2_style))
                last_was_heading = True
            elif line.startswith('### '):
                story.append(Paragraph(format_text(line[4:].strip()), heading3_style))
                last_was_heading = True
            elif line.startswith('#### '):
                story.append(Paragraph(format_text(line[5:].strip()), heading4_style))
                last_was_heading = True
            # Check for horizontal rules
            elif stripped == '---':
                story.append(HRFlowable(width="100%", thickness=1, lineCap='round', 
                                       color=colors.HexColor('#dee2e6'), 
                                       spaceBefore=12, spaceAfter=12))
                last_was_heading = False
            # Check for tables
            elif stripped.startswith('|'):
                table_data, end_idx = parse_table(lines, i)
                if table_data and len(table_data) > 1:
                    table = create_table(table_data)
                    if table:
                        story.append(Spacer(1, 0.1*inch))
                        story.append(table)
                        story.append(Spacer(1, 0.15*inch))
                    i = end_idx
                    last_was_heading = False
                    continue
            # Check for numbered lists (e.g., "1. ", "2. ")
            elif re.match(r'^\d+\.\s+', stripped):
                list_text = format_text(re.sub(r'^\d+\.\s+', '', stripped))
                # Extract the number
                match = re.match(r'^(\d+)\.\s+', stripped)
                list_num = match.group(1) if match else "1"
                story.append(Paragraph(f"{list_num}. {list_text}", numbered_style))
                last_was_heading = False
            # Check for bullet points
            elif stripped.startswith('- ') or stripped.startswith('* '):
                bullet_text = format_text(stripped[2:].strip())
                story.append(Paragraph(f"• {bullet_text}", bullet_style))
                last_was_heading = False
            # Regular text
            elif stripped:
                # Skip subtitle if it was already processed
                if i > 0 and lines[i-1].strip().startswith('# ') and stripped.startswith('**') and stripped.endswith('**'):
                    # This subtitle was already handled above, skip it
                    pass
                else:
                    para = Paragraph(format_text(stripped), normal_style)
                    story.append(para)
                last_was_heading = False
            else:
                # Empty line - add small spacer
                if not last_was_heading:
                    story.append(Spacer(1, 0.08*inch))
                last_was_heading = False
            
            i += 1
        
        pdf.build(story)
        
        pdf_path = Path('docs/Model_Training_Guide.pdf')
        if pdf_path.exists():
            print("✅ Success! Created docs/Model_Training_Guide.pdf with improved layout")
            # Note: All sections are processed sequentially from the markdown file,
            # so they appear in the correct order matching the TOC
            return True
        else:
            print("❌ Failed to create PDF")
            return False
            
    except ImportError as e:
        print(f"❌ reportlab not available: {e}")
        print("   Install with: pip install reportlab")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = convert_training_guide()
    sys.exit(0 if success else 1)
