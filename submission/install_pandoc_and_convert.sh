#!/bin/bash
# Install pandoc and convert PDF in one go

echo "Installing pandoc and BasicTeX..."
echo "This may take a few minutes..."
echo ""

# Install pandoc
if ! command -v pandoc &> /dev/null; then
    echo "Installing pandoc..."
    brew install pandoc
else
    echo "✓ pandoc already installed"
fi

# Install BasicTeX (LaTeX engine)
if ! command -v xelatex &> /dev/null; then
    echo "Installing BasicTeX (this may take 5-10 minutes)..."
    brew install --cask basictex
    
    echo ""
    echo "⚠️  After BasicTeX installation completes, you may need to:"
    echo "   1. Close and reopen your terminal, OR"
    echo "   2. Run: export PATH=\"/Library/TeX/texbin:\$PATH\""
    echo ""
    echo "Then run this script again, or run:"
    echo "   pandoc Project_Report.md -o Project_Report.pdf"
else
    echo "✓ LaTeX already installed"
fi

# Try to convert
if command -v pandoc &> /dev/null && command -v xelatex &> /dev/null; then
    echo ""
    echo "Converting Project_Report.md to PDF..."
    pandoc Project_Report.md -o Project_Report.pdf \
        --pdf-engine=xelatex \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        -V linestretch=1.2
    
    if [ -f "Project_Report.pdf" ]; then
        echo ""
        echo "✅ SUCCESS! Project_Report.pdf created!"
        echo ""
        # Check page count if possible
        if command -v mdls &> /dev/null; then
            pages=$(mdls -name kMDItemNumberOfPages Project_Report.pdf 2>/dev/null)
            if [ ! -z "$pages" ]; then
                echo "📄 Page count: $pages"
            fi
        fi
    else
        echo "❌ Conversion failed. Try running pandoc manually."
    fi
elif command -v pandoc &> /dev/null; then
    echo ""
    echo "⚠️  pandoc installed but LaTeX not ready yet."
    echo "   Please restart terminal and run:"
    echo "   pandoc Project_Report.md -o Project_Report.pdf"
else
    echo ""
    echo "❌ Installation incomplete. Please check errors above."
fi

