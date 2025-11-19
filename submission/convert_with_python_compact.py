#!/usr/bin/env python3
"""
Convert Project_Report.md to PDF using Python with compact formatting
Optimized to fit within 5 pages
"""

import sys
import os
from pathlib import Path

def convert_compact():
    """Convert with compact formatting to fit 5 pages"""
    try:
        import markdown
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_JUSTIFY
        import re
        
        print("Converting Project_Report.md to PDF (compact format)...")
        
        md_file = Path('Project_Report.md')
        if not md_file.exists():
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Create PDF with smaller margins
        pdf = SimpleDocTemplate(
            "Project_Report.pdf",
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        styles = getSampleStyleSheet()
        
        # Compact styles
        normal_style = ParagraphStyle(
            'CompactNormal',
            parent=styles['Normal'],
            fontSize=10,  # Smaller font
            leading=12,  # Tighter line spacing
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        )
        
        title_style = ParagraphStyle(
            'CompactTitle',
            parent=styles['Heading1'],
            fontSize=16,  # Smaller title
            leading=18,
            spaceAfter=8,
        )
        
        heading2_style = ParagraphStyle(
            'CompactHeading2',
            parent=styles['Heading2'],
            fontSize=12,  # Smaller heading
            leading=14,
            spaceAfter=6,
        )
        
        heading3_style = ParagraphStyle(
            'CompactHeading3',
            parent=styles['Heading3'],
            fontSize=11,
            leading=13,
            spaceAfter=4,
        )
        
        story = []
        
        # Parse markdown lines
        lines = md_content.split('\n')
        prev_empty = False
        
        for line in lines:
            line = line.strip()
            
            # Skip multiple empty lines
            if not line:
                if not prev_empty:
                    story.append(Spacer(1, 0.1*inch))
                    prev_empty = True
                continue
            
            prev_empty = False
            
            # Headers
            if line.startswith('# '):
                text = line[2:].strip()
                story.append(Paragraph(text, title_style))
            elif line.startswith('## '):
                text = line[3:].strip()
                story.append(Paragraph(text, heading2_style))
            elif line.startswith('### '):
                text = line[4:].strip()
                story.append(Paragraph(text, heading3_style))
            # Bullet points - compact
            elif line.startswith('- ') or line.startswith('* '):
                text = line[2:].strip()
                # Remove bold markers for cleaner look
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                story.append(Paragraph(f"• {text}", normal_style))
            # Numbered lists
            elif re.match(r'^\d+\.\s', line):
                text = re.sub(r'^\d+\.\s', '', line)
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                story.append(Paragraph(text, normal_style))
            # Regular text
            else:
                # Clean up markdown formatting
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
                # Remove code blocks markers
                text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
                story.append(Paragraph(text, normal_style))
        
        pdf.build(story)
        
        if Path('Project_Report.pdf').exists():
            print("✅ Success! Created Project_Report.pdf (compact format)")
            return True
        else:
            print("❌ Failed to create PDF")
            return False
            
    except ImportError as e:
        print(f"❌ reportlab not available: {e}")
        print("   Install with: pip install reportlab markdown")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = convert_compact()
    if success:
        print("\n📄 PDF created: Project_Report.pdf")
        print("   Please verify it's ≤ 5 pages")
    sys.exit(0 if success else 1)


