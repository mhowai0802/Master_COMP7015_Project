# AI Stocks Presentation

## Files

- **AI_Stocks_Presentation.pptx**: PowerPoint presentation file (30 slides)
- **create_presentation.py**: Python script to regenerate the presentation
- **README.md**: This file

## Presentation Overview

The presentation is designed for an **8-minute presentation** followed by Q&A. It contains **30 slides** with detailed model configurations and design rationale:

1. **Title Slide** - Project name and overview
2. **Problem & Motivation** - Why stock prediction is challenging
3. **Our Solution** - Multi-model approach overview
4. **System Architecture** - End-to-end pipeline overview
5. **Data: Price & News** - Price and news data collection
6. **Data: Fundamentals & Features** - Fundamental data and feature engineering
7. **Baseline Model: Algorithm** - Moving average crossover algorithm
8. **Baseline Model: Recommendations** - Price recommendations and advantages
9. **MLP Model: Architecture** - Feedforward network structure
10. **MLP Model: Design Rationale** - Why 2-3 layers, ReLU, 64-128 units
11. **MLP Model: Training** - Training process and hyperparameters
12. **Transformer Model: Architecture** - Sequence encoder structure
13. **Transformer Model: Design Rationale** - Why d_model, layers, attention heads
14. **Transformer: More Design Choices** - Positional encoding, dropout, FFN
15. **Transformer Model: Advantages** - Why Transformer for stocks
16. **Sentiment Analysis: Dataset** - Financial text dataset
17. **Sentiment Analysis: Models** - LSTM and BERT details
18. **Sentiment Models: Design Rationale** - Why LSTM architecture choices
19. **BERT Model: Design Rationale** - Why BERT and fine-tuning approach
20. **Training & Evaluation** - Training pipeline and metrics
21. **Prediction System** - End-to-end prediction flow
22. **Scenario Simulation** - Monte Carlo and intraday analysis
23. **Streamlit Frontend** - Interactive web interface
24. **Results & Achievements** - Key accomplishments
25. **Technical Challenges (Part 1)** - First set of challenges
26. **Technical Challenges (Part 2)** - More challenges and solutions
27. **Model Comparison** - Comparing all three models
28. **Future Improvements** - Potential enhancements
29. **Conclusion** - Project summary
30. **Q&A Slide** - Thank you slide

## Presentation Tips

### Timing (8 minutes total)
- **Slides 1-3**: Introduction, Problem, Solution (1 minute)
- **Slides 4-6**: System architecture & data sources (1 minute)
- **Slides 7-19**: Model details with design rationale (Baseline, MLP, Transformer, Sentiment) (4 minutes)
- **Slides 20-23**: Training, prediction, scenarios, frontend (1.5 minutes)
- **Slides 24-27**: Results, challenges, comparison (0.5 minutes)
- **Slides 28-30**: Future work, conclusion, Q&A (1 minute)

### Visualizations Recommended
- **Slide 10**: Show live demo of Streamlit interface
- **Slide 6-8**: Add architecture diagrams for each model
- **Slide 4**: Show sample data visualizations

### Key Features
- **Detailed Model Configurations**: Each model includes architecture details and design rationale
- **Design Rationale**: Explains why specific choices were made (layers, activations, dimensions)
- **Technical Depth**: Covers hyperparameters, training strategies, and architectural decisions
- **Comprehensive Coverage**: Baseline, MLP, Transformer, and Sentiment models all explained

### Model Design Rationale Included
- **MLP**: Why 2-3 layers, ReLU activation, 64-128 hidden units
- **Transformer**: Why d_model=32-64, 2-3 encoder layers, 4-8 attention heads, positional encoding
- **LSTM**: Why 2 layers, 128 units, dropout=0.5, embedding dimensions
- **BERT**: Why pre-trained model, fine-tuning approach, lower dropout
