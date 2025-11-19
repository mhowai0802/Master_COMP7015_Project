#!/usr/bin/env python3
"""
Create detailed PowerPoint presentation for AI Stocks project.
Optimized: No blank lines, smaller fonts to prevent overflow.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import os

def create_presentation():
    """Create the detailed PowerPoint presentation."""
    
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # Define colors
    title_color = RGBColor(41, 128, 185)  # Blue
    accent_color = RGBColor(52, 73, 94)   # Dark gray
    text_color = RGBColor(44, 62, 80)      # Dark blue-gray
    
    # Slide 1: Title Slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "AI Stocks"
    title.text_frame.paragraphs[0].font.size = Pt(54)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    subtitle.text = "Multi-Model Stock Price Prediction System\nUsing MLP, Transformer, and Sentiment Analysis\n\nCOMP7015 Project"
    subtitle.text_frame.paragraphs[0].font.size = Pt(18)
    subtitle.text_frame.paragraphs[0].font.color.rgb = accent_color
    
    # Add name and student number at bottom right
    left = Inches(6.5)
    top = Inches(6.5)
    width = Inches(3)
    height = Inches(0.8)
    text_box = slide.shapes.add_textbox(left, top, width, height)
    text_frame = text_box.text_frame
    text_frame.text = "Mak Ho Wai Winson\nStudent No.: 24465828"
    text_frame.paragraphs[0].font.size = Pt(14)
    text_frame.paragraphs[0].font.color.rgb = accent_color
    text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    text_frame.paragraphs[1].font.size = Pt(14)
    text_frame.paragraphs[1].font.color.rgb = accent_color
    text_frame.paragraphs[1].alignment = PP_ALIGN.RIGHT
    
    # Slide 2: Problem
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Problem & Motivation"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Stock price prediction challenges:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📈 Market volatility: Non-stationary patterns",
        "🔀 Multiple data sources need integration",
        "⏱️ Real-time requirements for trading",
        "🎯 Multi-modal analysis needed"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(16)
        p.font.color.rgb = text_color
        p.space_after = Pt(6)
    
    # Slide 3: Solution
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Our Solution"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Multi-Model Approach:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "💡 Combine multiple ML models",
        "   • Baseline: Moving average crossover",
        "   • MLP: Feedforward network",
        "   • Transformer: Sequence encoder",
        "💡 Integrate heterogeneous data",
        "   • Price, news, fundamentals",
        "💡 Interactive web interface",
        "   • Real-time predictions",
        "   • Actionable recommendations"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 4: System Architecture
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "System Architecture"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "End-to-End Pipeline:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "1️⃣ Data Collection",
        "   • Price: Yahoo Finance",
        "   • News: NewsAPI",
        "   • Fundamentals: Financial metrics",
        "2️⃣ Feature Engineering",
        "   • Tabular: 8 features",
        "   • Sequential: 30-day windows",
        "3️⃣ ML Models",
        "   • Baseline, MLP, Transformer",
        "4️⃣ Frontend: Streamlit"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 5: Price Data
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Data: Price & News"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Price Data (Yahoo Finance):"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "💰 Daily: OHLCV for 365 days",
        "💰 10 stocks: AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AVGO, TSM, SMCI",
        "💰 Intraday: 30-minute intervals, 60 days",
        "💰 Caching: JSON format",
        "📰 News Data (NewsAPI):",
        "📰 Real-time headlines",
        "📰 Sentiment labels: 5 classes",
        "📰 Integration with models"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(16)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 6: Fundamentals & Features
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Data: Fundamentals & Features"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Fundamental Data:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📊 P/E ratio, P/S ratio, market cap",
        "📊 Revenue growth, profit margin",
        "📊 Historical financial statements",
        "🎯 Tabular Features (MLP):",
        "   • last_close, MA_10, MA_30",
        "   • std_10, std_30",
        "   • sentiment, PE_ratio, PS_ratio",
        "🎯 Sequential Features (Transformer):",
        "   • 30×8 matrix",
        "   • Daily OHLCV + sentiment/fundamentals"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 7: Baseline Algorithm
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Baseline Model: Algorithm"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Moving Average Crossover:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📐 Calculate 10-day MA (MA_short)",
        "📐 Calculate 30-day MA (MA_long)",
        "📐 Compare relative positions",
        "🔍 Signal Generation:",
        "   • UP: MA_short > MA_long by >2%",
        "   • DOWN: MA_short < MA_long by >2%",
        "   • FLAT: Otherwise"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 8: Baseline Recommendations
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Baseline Model: Recommendations"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Price Recommendations:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "💰 Buy price: Current close × 0.98",
        "💰 Sell price: Current close × 1.05",
        "✅ Advantages:",
        "   • Fast: No training required",
        "   • Interpretable: Clear rules",
        "   • Baseline: Comparison for ML",
        "📊 Example:",
        "   • Current: $185.20",
        "   • Buy: $181.50",
        "   • Sell: $194.46"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 9: MLP Architecture
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "MLP Model: Architecture"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Feedforward Neural Network:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🏗️ Input: 8-dimensional feature vector",
        "🏗️ Hidden: 2-3 layers, 64-128 units",
        "🏗️ Activation: ReLU",
        "🏗️ Output: 3-class logits",
        "🏗️ Parameters: ~5K-10K weights"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 9.5: MLP Design Rationale (Consolidated)
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "MLP: Key Design Decisions"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Architecture Rationale:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📊 8 Features: Technical + fundamental + sentiment",
        "🔢 2-3 Layers: Sufficient for tabular data, avoids overfitting",
        "🔢 64-128 Units: Balance capacity vs training data size",
        "⚡ ReLU: Standard activation, avoids vanishing gradients",
        "🎯 3 Classes: Actionable buy/hold/sell decisions",
        "📊 ~5K-10K Params: Small enough for limited data"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(16)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 10: MLP Training
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "MLP Model: Training"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Training Process:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🎯 Dataset: Rolling 30-day windows",
        "🎯 Labels: Future 5-day returns",
        "🎯 Threshold: ±1% for classes",
        "🔍 Hyperparameter Search:",
        "   • hidden_dim: [32, 64, 128]",
        "   • num_layers: [2, 3]",
        "⚙️ Config:",
        "   • Optimizer: Adam (lr=1e-3)",
        "   • Early stopping: Patience=5"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 11: Transformer Architecture
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Transformer Model: Architecture"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Sequence-Based Encoder:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🏗️ Input: 30-day sequence (30×8)",
        "🏗️ Input projection: Linear(8 → d_model)",
        "🏗️ Positional encoding: Sinusoidal",
        "🏗️ Encoder: 2-3 layers",
        "🏗️ Attention: 4-8 heads",
        "🏗️ Parameters: ~100K weights",
        "🔍 Attention:",
        "   • Captures day relationships",
        "   • Identifies important patterns"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 11.5: Transformer Design Rationale (Consolidated)
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Transformer: Key Design Decisions"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Architecture Rationale:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📅 30-Day Sequence: Standard for stock prediction, balances context vs memory",
        "🔢 d_model=32-64: Start small, scale up if needed (~30K-100K params)",
        "🔢 2-3 Layers: Sufficient depth without overfitting",
        "🔢 4-8 Attention Heads: Multiple perspectives on temporal patterns",
        "📍 Sinusoidal PE: Captures relative positions (attention is permutation-invariant)",
        "💧 Dropout 0.1: Standard regularization for transformers"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(16)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 12: Transformer Advantages
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Transformer Model: Advantages"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Why Transformer for Stocks:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "✨ Captures temporal dependencies",
        "✨ Attention provides interpretability",
        "✨ Better for complex patterns",
        "🔍 Attention Example:",
        "   • Day 30 → Day 29 (recent)",
        "   • Day 30 → Day 25 (support)",
        "   • Day 30 → Day 20 (peak)",
        "🎯 Training: Same as MLP",
        "🎯 Dropout: 0.1 regularization"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 13: Sentiment Dataset
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Sentiment Analysis: Dataset"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Financial Text Classification:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📚 Financial PhraseBank",
        "   • Pre-labeled sentences",
        "📚 News Headlines",
        "   • Labeled with VADER",
        "📚 Classes: 5-class sentiment",
        "   • Very Negative → Very Positive",
        "📚 Split: 70% train, 15% val, 15% test"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 14: Sentiment Models
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Sentiment Analysis: Models"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Deep Learning Models:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🤖 LSTM Model:",
        "   • Embedding → LSTM (2 layers, 128 units)",
        "   • Supports GloVe embeddings",
        "   • Performance: ~60-65% accuracy",
        "🤖 BERT Model:",
        "   • bert-base-uncased (110M params)",
        "   • Fine-tuning with Hugging Face",
        "   • Performance: ~70-75% accuracy",
        "🔗 Integration: Sentiment → Models"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 14.5: Sentiment Models Design Rationale (Consolidated)
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Sentiment Models: Key Design Decisions"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Architecture Rationale:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🤖 LSTM: 2 layers, 128 units, embedding 100-dim (GloVe optional)",
        "   • Dropout 0.5: High regularization for text data",
        "   • Performance: ~60-65% accuracy",
        "🤖 BERT: Pre-trained bert-base-uncased (110M params)",
        "   • Fine-tuning: LR 2e-5, dropout 0.1, 3-5 epochs",
        "   • Performance: ~70-75% accuracy (best)",
        "💡 Why BERT > LSTM: Transfer learning, better language understanding"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 15: Training
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Training & Evaluation"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Training Pipeline:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📊 Dataset: 10 stocks, rolling windows",
        "📊 Future returns: 5-day horizon",
        "🎯 Strategy:",
        "   • Split: 80% train, 20% validation",
        "   • Hyperparameter search",
        "   • Early stopping: Patience=5",
        "📈 Metrics:",
        "   • Cross-entropy loss",
        "   • Classification accuracy"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 16: Prediction System
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Prediction System"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "End-to-End Flow:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🔄 Steps:",
        "   1. Fetch data (price, news, fundamentals)",
        "   2. Extract features",
        "   3. Run inference (all models)",
        "   4. Generate recommendations",
        "💰 Buy/Sell Logic:",
        "   • UP: Buy@98%, Sell@105%",
        "   • DOWN: Sell signal, Buy@95%",
        "   • FLAT: Hold"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 17: Scenario Simulation
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Scenario Simulation"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Monte Carlo & Analysis:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🎲 Monte Carlo:",
        "   • 1000 paths, 20-day horizon",
        "   • Sentiment-adjusted returns",
        "   • Probability of gain",
        "⏰ Intraday Analysis:",
        "   • Volatility patterns by hour",
        "   • Optimal monitoring windows",
        "   • 30-minute intervals"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 18: Frontend
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Streamlit Frontend"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Interactive Web Interface:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🖥️ Features:",
        "   • Stock selection: 10-stock watchlist",
        "   • Real-time analysis",
        "   • Multi-tab interface",
        "📊 Tabs:",
        "   • Predictions: Model outputs",
        "   • Scenarios: Monte Carlo",
        "   • News: Headlines + sentiment",
        "   • Intraday: Volatility patterns"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 19: Results - Overview
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Results: Evaluation Overview"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Comprehensive evaluation performed:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📊 Dataset:",
        "   • 2,150 samples from 10 AI-related stocks",
        "   • Period: 365 days",
        "   • Split: 80/20 train/test",
        "📊 Evaluation Focus:",
        "   • Stock Direction Prediction Models",
        "   • Sentiment Analysis Models"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 19.1: Results 3.1 - Stock Direction Prediction Models
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Results: Stock Direction Prediction Models"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Evaluation Setup:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📊 Task: 3-class classification (future 5-day returns)",
        "   • Class 0 (DOWN): Return ≤ -1%",
        "   • Class 1 (FLAT): -1% < Return < 1%",
        "   • Class 2 (UP): Return ≥ 1%",
        "📊 Model Performance Comparison:",
        "   • Transformer: 50.0% accuracy (Best)",
        "   • MLP: 40.9% accuracy",
        "   • Baseline (MA Crossover): Evaluation in progress"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 19.1b: Results 3.1 - Performance Metrics
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Results: Model Performance Metrics"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Detailed Performance Metrics:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📊 Transformer Model (Best Performer):",
        "   • Accuracy: 50.0%",
        "   • Precision (Macro): 0.348",
        "   • Recall (Macro): 0.353",
        "   • F1-Score (Macro): 0.294",
        "📊 MLP Model:",
        "   • Accuracy: 40.9%",
        "   • Precision (Macro): 0.248",
        "   • Recall (Macro): 0.325",
        "   • F1-Score (Macro): 0.204",
        "📊 Baseline Model:",
        "   • Moving average crossover strategy",
        "   • Evaluation methodology being refined"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 19.1c: Results 3.1 - Key Findings
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Results: Key Findings & Model Selection"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Key Findings:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "✅ Transformer Model (Best):",
        "   • Highest accuracy: 50.0%",
        "   • Sequential pattern recognition through attention",
        "   • Captures temporal dependencies across 30-day windows",
        "   • Better performance on 'Up' class (F1=0.64)",
        "   • Issue: Class imbalance affects 'Flat' predictions",
        "✅ MLP Model:",
        "   • Accuracy: 40.9%",
        "   • Learns complex feature interactions",
        "   • Better performance on 'Down' class (F1=0.58)",
        "   • Issue: Class imbalance affects predictions",
        "✅ Model Selection:",
        "   • Transformer selected as best model",
        "   • Better temporal pattern recognition",
        "   • Class weighting recommended for improvement"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 19.2: Results 3.2 - Sentiment Analysis Models
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Results: Sentiment Analysis Models"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Evaluation Setup:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📊 Task: 5-class sentiment classification",
        "   • Classes: very_negative, negative, neutral, positive, very_positive",
        "   • Dataset: Financial PhraseBank",
        "📊 Model Performance:",
        "   • BERT: 50.0% test accuracy (Best)",
        "   • BERT Validation: 68.4% accuracy",
        "   • LSTM: 44.7% accuracy",
        "📊 BERT Details:",
        "   • Fine-tuned bert-base-uncased",
        "   • Pre-trained transformer knowledge",
        "   • Better performance than LSTM",
        "📊 LSTM Details:",
        "   • Bidirectional LSTM, 2 layers, 128 units",
        "   • Trained from scratch",
        "   • Issue: Bias toward 'Neutral' class"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 19.2b: Results 3.2 - Key Findings
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Results: Sentiment Analysis Key Findings"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Key Findings:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "✅ BERT Model (Best Performer):",
        "   • 50.0% test accuracy, 68.4% validation accuracy",
        "   • Macro Precision: 0.313, Macro Recall: 0.310",
        "   • Macro F1: 0.290",
        "   • Pre-trained transformer captures nuanced language",
        "   • Transfer learning benefits demonstrated",
        "   • Best performance on 'Neutral' class (F1=0.67)",
        "✅ LSTM Model:",
        "   • 44.7% accuracy on test set",
        "   • Macro Precision: 0.090, Macro Recall: 0.200",
        "   • Faster inference, lower computational requirements",
        "   • Trained from scratch on Financial PhraseBank",
        "✅ Model Comparison:",
        "   • BERT outperforms LSTM (50.0% vs 44.7%)",
        "   • Both models affected by small dataset (176 samples)",
        "   • BERT shows better validation performance (68.4%)"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 20: Architecture Choices
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Architecture Choices & Rationale"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Design Decisions:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "💡 Multi-Model Ensemble",
        "   • MLP: Tabular feature interactions",
        "   • Transformer: Sequential dependencies",
        "   • Baseline: Interpretability & sanity check",
        "💡 Feature Fusion",
        "   • Price: Market dynamics",
        "   • Sentiment: Public perception",
        "   • Fundamentals: Company health",
        "💡 Sentiment Analysis",
        "   • Deep learning (LSTM/BERT) > rule-based",
        "   • Financial domain training improves relevance"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 20.5: Challenges Part 1
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Technical Challenges & Solutions"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Challenges & Solutions:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🔧 Data Heterogeneity",
        "   • Problem: Price time series, text, tabular data",
        "   • Solution: Separate feature engineering pipelines",
        "   • Normalization & aggregation steps",
        "🔧 Limited Training Data",
        "   • Problem: Only 10 stocks, limited history",
        "   • Solution: Rolling window approach",
        "   • Generates many samples from limited data"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 21: Challenges Part 2
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Technical Challenges (Continued)"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "More Challenges & Solutions:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "🔧 Real-Time Inference",
        "   • Problem: Fast prediction requirements",
        "   • Solution: Efficient model loading",
        "   • CPU fallback for environments without GPU",
        "🔧 Cache Management",
        "   • Problem: API rate limits & network latency",
        "   • Solution: JSON-based local cache",
        "   • Date-based invalidation",
        "🔧 Model Integration",
        "   • Problem: Different architectures",
        "   • Solution: Unified prediction interface"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 23: Limitations
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Limitations & Considerations"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Important Limitations:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "⚠️ Model Evaluation",
        "   • Need quantitative accuracy metrics",
        "   • Current: Qualitative integration success",
        "⚠️ Market Efficiency",
        "   • Short-term predictability inherently limited",
        "   • Models are decision-support tools, not guarantees",
        "⚠️ Data Quality",
        "   • Yahoo Finance: Possible inconsistencies",
        "   • NewsAPI: Rate limits affect high-frequency use",
        "⚠️ Temporal Generalization",
        "   • May not generalize to future market regimes"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 23.5: Future Work (Consolidated)
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Future Improvements"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Potential Enhancements:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📈 Advanced Models: LSTM/GRU, Temporal CNNs",
        "📊 Enhanced Features: More indicators (RSI, MACD), alternative data",
        "🔍 Explainability: SHAP values, attention visualization",
        "📉 Risk Management: Position sizing, stop-loss, portfolio metrics",
        "📊 Backtesting: Historical evaluation with transaction costs",
        "🔄 Real-Time: Live data feeds, incremental updates, continuous learning"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(16)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 24: Conclusion
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Conclusion"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Project Summary:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "✅ Comprehensive Multi-Model System",
        "   • Baseline, MLP, Transformer models",
        "   • Sentiment analysis (LSTM/BERT)",
        "   • Scenario simulation & intraday analysis",
        "✅ Key Achievements:",
        "   • Integrated price, news, fundamentals",
        "   • Unified prediction pipeline",
        "   • Interactive Streamlit interface",
        "✅ Technical Contributions:",
        "   • Applied Lab 2 (MLP) & Lab 5 (Transformer) patterns",
        "   • Solved multi-modal data integration",
        "   • Robust caching & real-time inference",
        "💡 Impact: Actionable buy/sell recommendations"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 25: Acknowledgments
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Acknowledgments"
    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    tf = content.text_frame
    tf.text = "Open-Source Libraries & Resources:"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = text_color
    
    for bullet in [
        "📚 PyTorch: Deep learning framework",
        "📚 Hugging Face Transformers: BERT models",
        "📚 yfinance: Yahoo Finance data access",
        "📚 Streamlit: Web application framework",
        "📚 Financial PhraseBank: Sentiment dataset",
        "📚 NewsAPI: News headlines data",
        "📚 COMP7015 Course Materials:",
        "   • Lab 2: MLP architecture patterns",
        "   • Lab 5: Transformer architecture patterns"
    ]:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0 if not bullet.startswith("   ") else 1
        p.font.size = Pt(16) if not bullet.startswith("   ") else Pt(14)
        p.font.color.rgb = text_color
        p.space_after = Pt(5)
    
    # Slide 26: Q&A
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title = slide.shapes.title
    
    title.text = "Questions & Answers"
    title.text_frame.paragraphs[0].font.size = Pt(54)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = title_color
    
    subtitle = slide.placeholders[1]
    subtitle.text = "Thank you for your attention!"
    subtitle.text_frame.paragraphs[0].font.size = Pt(28)
    subtitle.text_frame.paragraphs[0].font.color.rgb = accent_color
    
    # Save presentation
    output_path = os.path.join(os.path.dirname(__file__), "AI_Stocks_Presentation.pptx")
    prs.save(output_path)
    print(f"Presentation created successfully: {output_path}")
    print(f"Total slides: {len(prs.slides)}")
    
    return output_path

if __name__ == "__main__":
    try:
        create_presentation()
    except ImportError:
        print("Error: python-pptx not installed. Please install it:")
        print("  pip install python-pptx")
    except Exception as e:
        print(f"Error creating presentation: {e}")
        import traceback
        traceback.print_exc()
