# Converting Project Report to PDF

The `Project_Report.md` file needs to be converted to PDF format for submission. Here are several methods:

## Method 1: Using Pandoc (Recommended)

If you have pandoc installed:

```bash
pandoc Project_Report.md -o Project_Report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V linestretch=1.2
```

Or simpler (if using pdflatex):

```bash
pandoc Project_Report.md -o Project_Report.pdf
```

Install pandoc:
- macOS: `brew install pandoc`
- Linux: `sudo apt-get install pandoc texlive-latex-base`
- Windows: Download from https://pandoc.org/installing.html

## Method 2: Using Online Converters

1. **Markdown to PDF**: https://www.markdowntopdf.com/
   - Upload `Project_Report.md`
   - Download the generated PDF

2. **Dillinger**: https://dillinger.io/
   - Open the markdown file
   - Export as PDF

3. **StackEdit**: https://stackedit.io/
   - Open the markdown file
   - Export as PDF

## Method 3: Using VS Code Extension

1. Install "Markdown PDF" extension in VS Code
2. Open `Project_Report.md`
3. Right-click → "Markdown PDF: Export (pdf)"
4. Save the generated PDF

## Method 4: Using Python (weasyprint or markdown-pdf)

```bash
pip install weasyprint markdown
python -c "
import markdown
from weasyprint import HTML
with open('Project_Report.md', 'r') as f:
    md_content = f.read()
html = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
HTML(string=html).write_pdf('Project_Report.pdf')
"
```

## Method 5: Using LibreOffice or Word

1. Convert markdown to HTML first:
   ```bash
   pandoc Project_Report.md -o Project_Report.html
   ```
2. Open `Project_Report.html` in LibreOffice Writer or Microsoft Word
3. Save/Export as PDF

## Formatting Tips

To ensure the PDF fits within 5 pages:
- Use 11pt or 12pt font
- Set margins to 1 inch
- Line spacing: 1.15-1.2
- Consider reducing font size for code blocks if needed

## Verification

After conversion:
- [ ] PDF is readable
- [ ] All sections are included
- [ ] Page count ≤ 5 pages
- [ ] Tables/formatting are preserved
- [ ] Code blocks are formatted correctly

## Quick Check Script

To verify page count (macOS/Linux):

```bash
# If PDF is created
mdls -name kMDItemNumberOfPages Project_Report.pdf
```

Or use `pdfinfo`:
```bash
pdfinfo Project_Report.pdf | grep Pages
```

