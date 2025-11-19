#!/usr/bin/env python3
"""
Convert submission/Model_Technical_Documentation.md to PDF with one page per model
"""

import sys
import os
from pathlib import Path

def convert_to_pdf():
    """Convert Model Technical Documentation to PDF with one page per model"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
        import re
        
        print("Converting docs/Model_Technical_Documentation.md to PDF...")
        
        md_file = Path('docs/Model_Technical_Documentation.md')
        if not md_file.exists():
            print(f"❌ Error: {md_file} not found!")
            return False
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Create PDF with compact margins
        pdf = SimpleDocTemplate(
            "docs/Model_Technical_Documentation.pdf",
            pagesize=A4,
            rightMargin=0.6*inch,
            leftMargin=0.6*inch,
            topMargin=0.6*inch,
            bottomMargin=0.6*inch
        )
        
        styles = getSampleStyleSheet()
        
        # Define styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=20,
            leading=24,
            spaceAfter=12,
            alignment=TA_CENTER,
        )
        
        heading1_style = ParagraphStyle(
            'Heading1',
            parent=styles['Heading1'],
            fontSize=14,
            leading=16,
            spaceAfter=6,
            spaceBefore=8,
        )
        
        heading2_style = ParagraphStyle(
            'Heading2',
            parent=styles['Heading2'],
            fontSize=11,
            leading=13,
            spaceAfter=4,
            spaceBefore=5,
        )
        
        heading3_style = ParagraphStyle(
            'Heading3',
            parent=styles['Heading3'],
            fontSize=12,
            leading=16,
            spaceAfter=6,
            spaceBefore=8,
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=8.5,
            leading=10,
            alignment=TA_JUSTIFY,
            spaceAfter=2,
        )
        
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontSize=8.5,
            leading=10,
            alignment=TA_LEFT,
            leftIndent=15,
            spaceAfter=2,
        )
        
        code_style = ParagraphStyle(
            'Code',
            parent=styles['Code'],
            fontSize=7.5,
            leading=8.5,
            alignment=TA_LEFT,
            leftIndent=10,
            rightIndent=10,
            spaceAfter=2,
            spaceBefore=1,
            fontName='Courier',
        )
        
        example_step_style = ParagraphStyle(
            'ExampleStep',
            parent=styles['Normal'],
            fontSize=7.5,
            leading=9,
            alignment=TA_LEFT,
            leftIndent=0,
            spaceAfter=1.5,
            spaceBefore=1,
            fontName='Helvetica-Bold',
        )
        
        example_bullet_style = ParagraphStyle(
            'ExampleBullet',
            parent=styles['Normal'],
            fontSize=7.5,
            leading=9,
            alignment=TA_LEFT,
            leftIndent=12,
            spaceAfter=0.8,
            spaceBefore=0.4,
        )
        
        example_sub_bullet_style = ParagraphStyle(
            'ExampleSubBullet',
            parent=styles['Normal'],
            fontSize=7.5,
            leading=9,
            alignment=TA_LEFT,
            leftIndent=20,
            spaceAfter=0.6,
            spaceBefore=0.2,
        )
        
        def format_text(text_line):
            """Helper function to format text with markdown conversion"""
            # Convert bold text **text** to bold font
            text = re.sub(r'\*\*([^\*]+)\*\*', r'<b>\1</b>', text_line)
            # Convert inline code `text` to monospace font
            text = re.sub(r'`([^`]+)`', r'<font name="Courier" size="9">\1</font>', text)
            # Escape HTML but preserve formatting tags
            text = text.replace('<b>', '___BOLD_START___')
            text = text.replace('</b>', '___BOLD_END___')
            text = text.replace('<i>', '___ITALIC_START___')
            text = text.replace('</i>', '___ITALIC_END___')
            text = text.replace('<font name="Courier" size="9">', '___FONT_START___')
            text = text.replace('</font>', '___FONT_END___')
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            text = text.replace('___BOLD_START___', '<b>').replace('___BOLD_END___', '</b>')
            text = text.replace('___ITALIC_START___', '<i>').replace('___ITALIC_END___', '</i>')
            text = text.replace('___FONT_START___', '<font name="Courier" size="9">').replace('___FONT_END___', '</font>')
            return text
        
        story = []
        
        # Parse markdown content
        lines = md_content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines at the start
            if not line and i == 0:
                i += 1
                continue
            
            # Main title
            if line.startswith('# ') and 'Project Overview' not in line:
                title_text = line[2:].strip()
                story.append(Paragraph(title_text, title_style))
                story.append(Spacer(1, 0.3*inch))
                i += 1
                continue
            
            # Project Overview section
            if line.startswith('## Project Overview'):
                story.append(Paragraph("Project Overview", heading1_style))
                i += 1
                # Read content until next ## heading
                content_lines = []
                while i < len(lines):
                    if lines[i].strip().startswith('## '):
                        break
                    content_lines.append(lines[i])
                    i += 1
                # Process content
                for content_line in content_lines:
                    content_line = content_line.strip()
                    if not content_line:
                        story.append(Spacer(1, 0.1*inch))
                    elif content_line.startswith('- ') or content_line.startswith('* '):
                        text = format_text(content_line[2:].strip())
                        story.append(Paragraph(f"• {text}", bullet_style))
                    else:
                        text = format_text(content_line)
                        if text:
                            story.append(Paragraph(text, normal_style))
                story.append(PageBreak())
                continue
            
            # Model sections (## Model X:)
            if line.startswith('## Model '):
                model_title = line[2:].strip()
                i += 1
                
                # Process model content (Input, Output, Technical Flow)
                model_content = [Paragraph(model_title, heading1_style), Spacer(1, 0.08*inch)]
                current_subsection = None
                subsection_content = []
                in_code_block = False
                code_block_lines = []
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Stop at next model or major section
                    if line.startswith('## '):
                        break
                    
                    # Handle code blocks
                    if line.startswith('```'):
                        if in_code_block:
                            # End of code block
                            if code_block_lines:
                                code_text = '\n'.join(code_block_lines)
                                # Add code block to current subsection or directly to model
                                if current_subsection:
                                    subsection_content.append(Paragraph(code_text, code_style))
                                else:
                                    model_content.append(Paragraph(code_text, code_style))
                                code_block_lines = []
                            in_code_block = False
                        else:
                            # Start of code block
                            in_code_block = True
                        i += 1
                        continue
                    
                    if in_code_block:
                        code_block_lines.append(line)
                        i += 1
                        continue
                    
                    # Subsection headings (###)
                    if line.startswith('### '):
                        # Save previous subsection
                        if current_subsection and subsection_content:
                            model_content.append(Paragraph(current_subsection, heading2_style))
                            for content in subsection_content:
                                model_content.append(content)
                            # Less spacing before step-by-step examples
                            if "Step-by-Step Example" not in current_subsection:
                                model_content.append(Spacer(1, 0.04*inch))
                            else:
                                model_content.append(Spacer(1, 0.02*inch))
                        
                        # Start new subsection
                        current_subsection = line[4:].strip()
                        subsection_content = []
                        i += 1
                        continue
                    
                    # Check if we're in a "Step-by-Step Example" section
                    is_example_section = current_subsection and "Step-by-Step Example" in current_subsection
                    
                    # Content lines
                    if not line:
                        if is_example_section:
                            subsection_content.append(Spacer(1, 0.02*inch))
                        else:
                            subsection_content.append(Spacer(1, 0.04*inch))
                    elif line.startswith('**Step ') or line.startswith('**Input ') or line.startswith('**Output'):
                        # Step headers in example sections - remove ** and format
                        text = format_text(line.replace('**', '').strip())
                        if is_example_section:
                            subsection_content.append(Paragraph(text, example_step_style))
                        else:
                            subsection_content.append(Paragraph(text, normal_style))
                    elif line.startswith('- ') or line.startswith('* '):
                        text = format_text(line[2:].strip())
                        # Check if it's a sub-bullet (starts with space or has nested structure)
                        if is_example_section:
                            # Check indentation level
                            if line.startswith('  - ') or line.startswith('  * '):
                                subsection_content.append(Paragraph(f"  • {text}", example_sub_bullet_style))
                            else:
                                subsection_content.append(Paragraph(f"• {text}", example_bullet_style))
                        else:
                            subsection_content.append(Paragraph(f"• {text}", bullet_style))
                    else:
                        text = format_text(line)
                        if text:
                            if is_example_section:
                                subsection_content.append(Paragraph(text, example_bullet_style))
                            else:
                                subsection_content.append(Paragraph(text, normal_style))
                    
                    i += 1
                
                # Save last subsection
                if current_subsection and subsection_content:
                    model_content.append(Paragraph(current_subsection, heading2_style))
                    for content in subsection_content:
                        model_content.append(content)
                
                # Handle any remaining code block (shouldn't happen, but safety check)
                if code_block_lines:
                    code_text = '\n'.join(code_block_lines)
                    if current_subsection:
                        model_content.append(Paragraph(current_subsection, heading2_style))
                    model_content.append(Paragraph(code_text, code_style))
                
                # Split model content: regular sections, example input/output, technical flow, step-by-step examples
                # Order: Example Input/Output (together) -> Technical Flow -> Step-by-Step Example
                regular_content = []
                example_input_output_sections = []
                technical_flow_sections = []
                step_by_step_sections = []
                current_example_io = None
                current_technical_flow = None
                current_step_example = None
                
                for item in model_content:
                    if isinstance(item, Paragraph):
                        # Check if this is Example Input heading
                        if item.style == heading2_style and "Example Input" in item.text:
                            # Save any previous sections
                            if current_step_example:
                                step_by_step_sections.append(current_step_example)
                                current_step_example = None
                            if current_technical_flow:
                                technical_flow_sections.append(current_technical_flow)
                                current_technical_flow = None
                            if current_example_io:
                                example_input_output_sections.append(current_example_io)
                            current_example_io = [item]
                        # Check if this is Example Output heading
                        elif item.style == heading2_style and "Example Output" in item.text:
                            if current_example_io:
                                current_example_io.append(item)
                            else:
                                current_example_io = [item]
                        # Check if this is Technical Flow heading
                        elif item.style == heading2_style and ("Technical Flow" in item.text or "Complete End-to-End Flow" in item.text):
                            # Save Example Input/Output if complete
                            if current_example_io:
                                example_input_output_sections.append(current_example_io)
                                current_example_io = None
                            if current_step_example:
                                step_by_step_sections.append(current_step_example)
                                current_step_example = None
                            if current_technical_flow:
                                technical_flow_sections.append(current_technical_flow)
                            current_technical_flow = [item]
                        # Check if this is a step-by-step example heading
                        elif item.style == heading2_style and "Step-by-Step Example" in item.text:
                            # Save Technical Flow first
                            if current_technical_flow:
                                technical_flow_sections.append(current_technical_flow)
                                current_technical_flow = None
                            if current_example_io:
                                example_input_output_sections.append(current_example_io)
                                current_example_io = None
                            if current_step_example:
                                step_by_step_sections.append(current_step_example)
                            current_step_example = [item]
                        elif item.style == heading2_style:
                            # Save any current sections
                            if current_step_example:
                                step_by_step_sections.append(current_step_example)
                                current_step_example = None
                            if current_technical_flow:
                                technical_flow_sections.append(current_technical_flow)
                                current_technical_flow = None
                            if current_example_io:
                                example_input_output_sections.append(current_example_io)
                                current_example_io = None
                            regular_content.append(item)
                        else:
                            # Add to current section
                            if current_step_example:
                                current_step_example.append(item)
                            elif current_technical_flow:
                                current_technical_flow.append(item)
                            elif current_example_io:
                                current_example_io.append(item)
                            else:
                                regular_content.append(item)
                    else:
                        # Add spacers and other items to current section
                        if current_step_example:
                            current_step_example.append(item)
                        elif current_technical_flow:
                            current_technical_flow.append(item)
                        elif current_example_io:
                            current_example_io.append(item)
                        else:
                            regular_content.append(item)
                
                # Save any remaining sections
                if current_example_io:
                    example_input_output_sections.append(current_example_io)
                if current_technical_flow:
                    technical_flow_sections.append(current_technical_flow)
                if current_step_example:
                    step_by_step_sections.append(current_step_example)
                
                # Add content in correct order:
                # 1. Regular content (Input, Output, etc.)
                if regular_content:
                    story.append(KeepTogether(regular_content))
                
                # 2. Example Input and Output together (on one page)
                for example_io_section in example_input_output_sections:
                    story.append(PageBreak())
                    story.append(KeepTogether(example_io_section))
                
                # 3. Technical Flow and Complete End-to-End Flow
                for tech_flow_section in technical_flow_sections:
                    story.append(PageBreak())
                    story.append(KeepTogether(tech_flow_section))
                
                # 4. Step-by-Step Examples
                for step_section in step_by_step_sections:
                    story.append(PageBreak())
                    story.append(KeepTogether(step_section))
                
                story.append(PageBreak())
                continue
            
            # Other sections (Model Integration Pipeline, Conclusion)
            if line.startswith('## '):
                section_title = line[2:].strip()
                story.append(Paragraph(section_title, heading1_style))
                story.append(Spacer(1, 0.2*inch))
                i += 1
                
                # Read content until next ## heading or end
                content_lines = []
                while i < len(lines):
                    if lines[i].strip().startswith('## '):
                        break
                    content_lines.append(lines[i])
                    i += 1
                
                # Process content
                for content_line in content_lines:
                    content_line = content_line.strip()
                    if not content_line:
                        story.append(Spacer(1, 0.1*inch))
                    elif content_line.startswith('- ') or content_line.startswith('* '):
                        text = format_text(content_line[2:].strip())
                        story.append(Paragraph(f"• {text}", bullet_style))
                    elif content_line.startswith('### '):
                        story.append(Paragraph(content_line[4:].strip(), heading3_style))
                    else:
                        text = format_text(content_line)
                        if text:
                            story.append(Paragraph(text, normal_style))
                
                story.append(PageBreak())
                continue
            
            i += 1
        
        pdf.build(story)
        
        pdf_path = Path('docs/Model_Technical_Documentation.pdf')
        if pdf_path.exists():
            print("✅ Success! Created docs/Model_Technical_Documentation.pdf")
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
    success = convert_to_pdf()
    if success:
        print("\n📄 PDF created: docs/Model_Technical_Documentation.pdf")
    sys.exit(0 if success else 1)

