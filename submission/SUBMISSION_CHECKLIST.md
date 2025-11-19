# Final Submission Checklist

## Required Deliverables

- [ ] **Project Report (PDF)**: Maximum 5 A4 pages, single column
  - [ ] Motivation section
  - [ ] Methods section
  - [ ] Results section
  - [ ] Discussion section
  - [ ] **Mandatory**: Contribution of each group member section
  - [ ] Acknowledgments of major third-party libraries

- [ ] **Source Code (ZIP file)**: Single .zip file containing all source code
  - [ ] All Python source files (.py)
  - [ ] requirements.txt with all dependencies
  - [ ] README or setup instructions
  - [ ] Exclude: __pycache__ directories, .pyc files, cache files
  - [ ] Exclude: .git directory, virtual environment directories
  - [ ] Include: Trained model weights (.pth files) if applicable

## Pre-Submission Steps

### 1. Update Project Report

- [ ] Convert `Project_Report.md` to PDF format
  - Use pandoc: `pandoc Project_Report.md -o Project_Report.pdf`
  - Or use online converters like Markdown to PDF
  - Verify it's ≤ 5 pages when formatted
- [ ] Fill in group member contributions section with actual names and contributions
- [ ] Review and ensure all sections are complete

### 2. Prepare Source Code Package

#### Option A: Using prepare_submission.sh (Linux/macOS)
```bash
./prepare_submission.sh
```

#### Option B: Using prepare_submission.bat (Windows)
```batch
prepare_submission.bat
```

#### Option C: Manual Preparation
- Create a new directory for submission
- Copy all source directories: `api/`, `config/`, `domain/`, `frontend/`, `ml/`
- Copy data files from `data/` if needed
- Copy trained models from `models/` if included
- Copy: `requirements.txt`, `README.md`, `SUBMISSION_README.md`
- Remove: `__pycache__/`, `*.pyc`, `cache/`, `.git/`, `.venv/`
- Create zip file

### 3. Verify Submission Package

- [ ] Check zip file structure:
  ```
  AI_Stocks_Submission.zip
  ├── api/
  ├── config/
  ├── data/ (if applicable)
  ├── domain/
  ├── frontend/
  ├── ml/
  ├── models/ (if applicable)
  ├── requirements.txt
  ├── README.md
  └── SUBMISSION_README.md
  ```

- [ ] Test extraction:
  - Extract zip to a temporary directory
  - Verify all files are present
  - Check that no sensitive files (API keys) are included

- [ ] Verify requirements.txt:
  - [ ] numpy
  - [ ] pandas
  - [ ] yfinance
  - [ ] streamlit
  - [ ] scikit-learn
  - [ ] requests
  - [ ] vaderSentiment
  - [ ] torch
  - [ ] transformers
  - [ ] nltk
  - [ ] datasets

### 4. Test in Clean Environment (Recommended)

- [ ] Extract submission to a clean directory
- [ ] Create new virtual environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Download NLTK data: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`
- [ ] Test imports: `python -c "from ml.lab2_mlp_model import StockMLP; print('OK')"`
- [ ] Test application: `streamlit run frontend/app.py`
- [ ] Verify models load correctly

### 5. Code Quality Checks

- [ ] Remove debug print statements
- [ ] Remove commented-out code blocks (if excessive)
- [ ] Ensure consistent code formatting
- [ ] Verify all imports work
- [ ] Check for hardcoded paths (should use relative paths)

### 6. Documentation

- [ ] README.md includes setup instructions
- [ ] SUBMISSION_README.md includes FSC 8/F lab-specific instructions
- [ ] Code comments are clear and helpful
- [ ] Function docstrings are present for major functions

## Submission

- [ ] PDF report file named appropriately (e.g., `Project_Report.pdf`)
- [ ] ZIP file named appropriately (e.g., `AI_Stocks_Submission.zip` or `Source_Code.zip`)
- [ ] Both files ready for upload/submission
- [ ] Verify file sizes are reasonable (< 100MB total recommended)

## FSC 8/F Lab Environment Notes

The code should run smoothly in the FSC 8/F lab environment. Ensure:
- [ ] No hardcoded absolute paths
- [ ] All relative imports work correctly
- [ ] Dependencies are available via pip (no custom installations required)
- [ ] Models work on CPU (GPU is optional)
- [ ] Cache directory is created automatically if it doesn't exist
- [ ] API keys are optional (system works without them, though with reduced functionality)

## Final Reminders

- **Project Report**: Must include contribution section for each group member
- **Third-party Libraries**: Must acknowledge PyTorch, Hugging Face Transformers, Streamlit, yfinance, etc. in the report
- **Code Format**: Clean, well-organized, follows Python conventions
- **Testing**: Code should run without errors in the lab environment

---

**Good luck with your submission!**

