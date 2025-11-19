# All Methods to Convert Project_Report.md to PDF

## Method 1: Install Pandoc (Best Quality - Recommended)

Pandoc produces high-quality PDFs with proper formatting.

### Install Pandoc:
```bash
brew install pandoc
brew install --cask basictex  # Or: brew install mactex (larger but complete)
```

### Convert:
```bash
pandoc Project_Report.md -o Project_Report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V linestretch=1.2
```

**Pros**: Professional quality, preserves formatting
**Cons**: Requires installation (~500MB for BasicTeX)

---

## Method 2: Using Python + markdown + pdfkit (If you have wkhtmltopdf)

### Install dependencies:
```bash
pip3 install markdown pdfkit
brew install wkhtmltopdf
```

### Convert script:
```python
import markdown
import pdfkit

with open('Project_Report.md', 'r') as f:
    md_content = f.read()

html = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
pdfkit.from_string(html, 'Project_Report.pdf')
```

---

## Method 3: Using Python + markdown + xhtml2pdf

### Install:
```bash
pip3 install markdown xhtml2pdf
```

### Convert script:
```python
import markdown
from xhtml2pdf import pisa

with open('Project_Report.md', 'r') as f:
    md_content = f.read()

html = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

with open('Project_Report.pdf', 'wb') as f:
    pisa.CreatePDF(html, dest=f)
```

---

## Method 4: macOS Native - Using TextEdit/Pages

1. Open `Project_Report.md` in TextEdit
2. Select All (Cmd+A) and Copy (Cmd+C)
3. Open **Pages** (or TextEdit)
4. Paste the content
5. Format as needed (ensure ≤ 5 pages)
6. File → Export to PDF → Save as `Project_Report.pdf`

**Pros**: No installation needed, native macOS
**Cons**: Manual formatting required

---

## Method 5: Using VS Code Extension

1. Install VS Code if you don't have it: https://code.visualstudio.com/
2. Install extension: "Markdown PDF" by yzane
3. Open `Project_Report.md` in VS Code
4. Right-click → "Markdown PDF: Export (pdf)"
5. PDF will be created in the same directory

**Pros**: Easy, good quality
**Cons**: Requires VS Code installation

---

## Method 6: Online Converters (No Installation)

### Option A: markdowntopdf.com
1. Go to: https://www.markdowntopdf.com/
2. Upload `Project_Report.md`
3. Click "Convert to PDF"
4. Download

### Option B: Dillinger.io
1. Go to: https://dillinger.io/
2. Click "Import from" → "File"
3. Select `Project_Report.md`
4. Click "Export as" → "PDF"

### Option C: StackEdit
1. Go to: https://stackedit.io/
2. Click "Start writing" → "Import from disk"
3. Select `Project_Report.md`
4. Click "Export to disk" → "PDF"

**Pros**: No installation, works immediately
**Cons**: Requires internet, upload your file

---

## Method 7: Using Python + markdown + reportlab (Programmatic)

### Install:
```bash
pip3 install markdown reportlab
```

### Convert script:
```python
import markdown
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Read markdown
with open('Project_Report.md', 'r') as f:
    md_content = f.read()

# Convert to HTML then extract text (simplified)
html = markdown.markdown(md_content)

# Create PDF
pdf = SimpleDocTemplate("Project_Report.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = []

# Parse and add content (simplified - you'd need HTML parsing)
for line in md_content.split('\n'):
    if line.strip():
        story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

pdf.build(story)
```

---

## Method 8: Using Chrome/Edge Browser Print

1. Open `Project_Report.md` in a markdown viewer (VS Code preview, GitHub, etc.)
2. Print (Cmd+P)
3. Select "Save as PDF" as destination
4. Save as `Project_Report.pdf`

**Pros**: No installation, works immediately
**Cons**: May need formatting adjustments

---

## Method 9: Quick Install Pandoc (Fastest Professional Method)

Since you have Homebrew, this is the fastest way to get professional results:

```bash
# Install pandoc (small, ~50MB)
brew install pandoc

# Install BasicTeX (LaTeX engine, ~100MB)
brew install --cask basictex

# After installation, refresh PATH or restart terminal, then:
pandoc Project_Report.md -o Project_Report.pdf \
  -V geometry:margin=1in \
  -V fontsize=11pt
```

This gives you the best quality PDF with minimal setup.

---

## Method 10: Using Python Script I Created

I've created a script that tries multiple methods. Run:

```bash
python3 convert_report_to_pdf.py
```

(Note: This requires weasyprint dependencies which may need system libraries)

---

## My Recommendation for You:

**Best Option**: Install pandoc via Homebrew (Method 9) - takes 5 minutes, gives professional results

**Quickest Option**: Use online converter (Method 6, Option A) - takes 2 minutes, no installation

**If you have VS Code**: Use Method 5 - easy and good quality

---

## After Conversion - Verify:

1. Open `Project_Report.pdf`
2. Check page count (should be ≤ 5 pages)
3. Verify formatting looks good
4. Check that your name (Winson Mak) appears in Section 5
5. Ensure all sections are present

---

## Need Help?

If you encounter issues with any method, let me know which one you're trying and I can help troubleshoot!

