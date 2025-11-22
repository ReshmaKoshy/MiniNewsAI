# MiniNewsAI: Kid-Safe News Content Reframing System

## Project Overview

MiniNewsAI is a deep learning system designed to classify and reframe news articles into kid-safe content for children aged 6-14. The system uses a two-stage pipeline: (1) multiclass classification to categorize content as SAFE, SENSITIVE, or UNSAFE, and (2) content reframing to rewrite articles into age-appropriate formats.

## System Architecture

### Pipeline Flow

```
Input News Articles
    ↓
[Multiclass Classifier Training] → RoBERTa-base (SAFE/SENSITIVE/UNSAFE)
    ↓
[Content Reframer Training] → Mistral-7B-Instruct with LoRA
    ↓
[Inference Pipeline] → Classify → Reframe based on label
    ↓
Kid-Safe Rewritten Article
```

## Dataset

### Source Datasets

- **Global News Dataset**: https://www.kaggle.com/datasets/everydaycodings/global-news-dataset
- **News Article Categories**: https://www.kaggle.com/datasets/timilsinabimal/newsarticlecategories

### Data Processing

1. **Categories Extracted**: ARTS & CULTURE, ENVIRONMENT, SCIENCE, SPORTS
2. **BART Summarization**: Articles summarized to ~500 words
3. **Labeling**:
   - Manual labeling: 1,000 article pairs
   - Automated labeling: Gemini 2.5 Pro API (with reasoning)
4. **DeepSeek Rewriting**: Generated kid-safe rewrites for SAFE/SENSITIVE articles

### Dataset Statistics

- **Total Samples**: 7,068 news articles
- **Train/Val/Test Split**: 70/15/15 (stratified by label and category)
- **Label Distribution**:
  - SENSITIVE: 3,043 (43.0%)
  - SAFE: 2,754 (39.0%)
  - UNSAFE: 1,271 (18.0%)

## Models

### 1. Multiclass Classifier

- **Model**: RoBERTa-base (125M parameters)
- **Task**: 3-way classification (SAFE, SENSITIVE, UNSAFE)
- **Loss Function**: Multi-class Focal Loss (gamma=3, alpha=[1.0, 3.0, 5.0])
- **Performance**:
  - Overall Accuracy: 79.9%
  - UNSAFE Recall: 84.4% (Target: ≥80%) ✓
  - SENSITIVE Recall: 75.0%
  - SAFE Recall: 83.3%

### 2. Content Reframer

- **Model**: Mistral-7B-Instruct-v0.2 with LoRA fine-tuning
- **Task**: Rewrite articles into kid-safe formats
- **Styles**:
  - **SAFE**: Faithful, factual summary with kid vocabulary
  - **SENSITIVE**: Educational reframing with moral/lesson framing
  - **UNSAFE**: Filtered out (no output)

## Installation

### Prerequisites

- Python 3.8+
- CUDA 12.8+ (for GPU support)
- NVIDIA GPU with 16GB+ VRAM (recommended)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd MiniNewsAI-main
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download pre-trained models (if available):
   - Place classifier model in `models/multiclass_classifier/best_model/`
   - Place rewriter model in `models/rewriter_seq2seq/`

## Usage

### Training Models

#### 1. Data Preparation

```bash
# Run data preparation notebook
jupyter notebook notebooks/01_data_preparation.ipynb
```

#### 2. Train Multiclass Classifier

```bash
# Run classifier training notebook
jupyter notebook notebooks/02_multiclass_classifier_training.ipynb
```

#### 3. Train Content Reframer

```bash
# Run reframer training notebook
jupyter notebook notebooks/03_kid_safe_rewriter_training.ipynb
```

### Running the Web UI

#### Using Gradio Interface

1. **Navigate to the project directory:**
   ```bash
   cd MiniNewsAI
   ```

2. **Run the application:**
   ```bash
   python ui/app.py
   ```

3. **Open your browser:**
   - The interface will be available at `http://localhost:7860`
   - Click "⚙️ Load Models" to initialize the classifier and rewriter models
   - Enter a news article and click "🚀 Process Article" to classify and rewrite

#### Using the Interface

1. **Load Models**: Click "⚙️ Load Models" to initialize the classifier and rewriter models (required before processing)

2. **Enter Article**: 
   - Paste your news article in the text box
   - Optionally add a title

3. **Process**: Click "🚀 Process Article" to:
   - Classify the article (SAFE/SENSITIVE/UNSAFE)
   - See confidence scores for each category
   - Get a kid-safe rewrite (if applicable)

#### Classification Labels

- 🟢 **SAFE**: Content is already appropriate for children
- 🟡 **SENSITIVE**: Content needs rewriting to be child-friendly  
- 🔴 **UNSAFE**: Content cannot be made safe for children

#### Output

The interface displays:
- **Classification Results**: Label and confidence scores for all three categories
- **Original Article**: The input article
- **Kid-Safe Rewrite**: The rewritten version (if SAFE or SENSITIVE)

#### Troubleshooting

**Models not loading?**
- Check that model directories exist and contain the required files:
  - `models/multiclass_classifier/best_model/` (classifier)
  - `models/kid_safe_rewriter/best_model/` (rewriter)
- Ensure you have enough GPU/CPU memory
- Check that transformers and peft libraries are installed

**Slow processing?**
- First run will be slower (model loading)
- GPU acceleration significantly speeds up inference
- Consider reducing max_new_tokens if generation is too slow

**Port already in use?**
- Change the port in `ui/app.py`: `demo.launch(server_port=7861)`

## Project Structure

```
MiniNewsAI-main/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Processed datasets (train/val/test)
│   └── keywords/               # Keyword filters
├── models/                     # All trained models (uniform structure)
│   ├── multiclass_classifier/  # Notebook 2 trained classifier model
│   │   ├── best_model/         # Best model checkpoint
│   │   └── best_metrics.json   # Model metrics
│   └── kid_safe_rewriter/      # Notebook 3 trained reframer model (LoRA)
│       ├── best_model/         # Best model adapter (LoRA weights)
│       ├── checkpoint-*/       # Training checkpoints
│       └── *.json, *.model     # Tokenizer files
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_multiclass_classifier_training.ipynb
│   └── 03_kid_safe_rewriter_training.ipynb
├── scripts/
│   ├── bart_summarize.ipynb
│   ├── gemini_label_data.py
│   ├── deepseek_safe_summarize.ipynb
│   └── deepseek_sensitive_reframe.ipynb
├── ui/
│   └── app.py                  # Gradio web interface
├── results/                    # All outputs from notebooks (nothing at root level)
│   ├── data_preparation/       # Notebook 1 outputs
│   │   ├── label_distribution.png
│   │   └── category_label_distribution.png
│   ├── multiclass_classifier/  # Notebook 2 outputs
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   └── validation_predictions.csv
│   └── kid_safe_rewriter/      # Notebook 3 outputs
│       ├── training_history.png
│       ├── training_history_per_epoch.png
│       ├── evaluation_metrics.png
│       ├── evaluation_metrics.csv
│       ├── evaluation_results.csv
│       ├── label_distribution.png
│       ├── text_length_distribution.png
│       └── comparison_*.png     # Comparison visualizations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Folder Organization

**Unified Output Structure**: All outputs from notebooks are saved to subfolders within `results/` (nothing at root level):

- **Notebook 1 (Data Preparation)**: Saves visualizations to `results/data_preparation/`
- **Notebook 2 (Multiclass Classifier)**: Saves all outputs to `results/multiclass_classifier/`
- **Notebook 3 (Kid-Safe Rewriter)**: Saves all outputs to `results/kid_safe_rewriter/`

**Unified Model Structure**: All trained model checkpoints follow a uniform structure in `models/`:

- `models/multiclass_classifier/best_model/`: Best model checkpoint and tokenizer
- `models/multiclass_classifier/best_metrics.json`: Model performance metrics
- `models/kid_safe_rewriter/best_model/`: Best LoRA adapter weights and tokenizer
- `models/kid_safe_rewriter/checkpoint-*/`: Training checkpoint snapshots

## Results

### Classification Performance

- **Overall Accuracy**: 79.9%
- **UNSAFE Recall**: 84.4% (Target: ≥80%) ✓
- **SENSITIVE Recall**: 75.0%
- **SAFE Recall**: 83.3%

### Reframing Quality

- **Average Jaccard Similarity**: 0.51
- **Length Preservation**: ~104% of expected length
- **Style Adherence**: Good for both SAFE and SENSITIVE styles

### Visualizations

**Notebook 1 (Data Preparation)**:

- Label distribution: `results/data_preparation/label_distribution.png`
- Category-label distribution: `results/data_preparation/category_label_distribution.png`

**Notebook 2 (Multiclass Classifier)**:

- Training history: `results/multiclass_classifier/training_history.png`
- Confusion matrix: `results/multiclass_classifier/confusion_matrix.png`
- Validation predictions: `results/multiclass_classifier/validation_predictions.csv`

**Notebook 3 (Kid-Safe Rewriter)**:

- Training history: `results/kid_safe_rewriter/training_history.png`
- Training history per epoch: `results/kid_safe_rewriter/training_history_per_epoch.png`
- Evaluation metrics: `results/kid_safe_rewriter/evaluation_metrics.png`
- Evaluation metrics CSV: `results/kid_safe_rewriter/evaluation_metrics.csv`
- Evaluation results: `results/kid_safe_rewriter/evaluation_results.csv`
- Label distribution: `results/kid_safe_rewriter/label_distribution.png`
- Text length distribution: `results/kid_safe_rewriter/text_length_distribution.png`
- Comparison visualizations: `results/kid_safe_rewriter/comparison_*.png`

## Known Issues and Limitations

1. **False Negatives**: 30 UNSAFE articles misclassified (15.6% miss rate)
   - 4.2% misclassified as SAFE (critical)
   - 11.5% misclassified as SENSITIVE
2. **Class Imbalance**: UNSAFE is minority class (18% of data)
3. **Reframing Quality**: Room for improvement in faithfulness and style consistency
4. **Evaluation Metrics**: Need human evaluation for quality assessment

## Future Improvements

### Short-term

- [ ] Improve UNSAFE recall to 90%+
- [ ] Enhance reframing quality with better prompts
- [ ] Add batch processing support
- [ ] Implement confidence score thresholds

### Medium-term

- [ ] Human evaluation for reframing quality
- [ ] Style metrics (faithfulness, simplicity, moral clarity)
- [ ] Longer context handling
- [ ] Multi-turn refinement support

### Long-term

- [ ] Customization for different age ranges
- [ ] Production-ready API
- [ ] Monitoring and logging
- [ ] Deployment infrastructure

## Ethical Considerations

### Responsible AI

1. **Content Safety**: Primary goal is to protect children from harmful content
2. **Bias Mitigation**: Ensuring fair representation across categories
3. **Transparency**: Clear labeling and explanation of classification decisions
4. **Privacy**: No storage of user input articles (unless explicitly requested)

### Challenges

1. **False Positives**: SAFE content incorrectly flagged as UNSAFE
2. **False Negatives**: UNSAFE content slipping through (critical issue)
3. **Cultural Sensitivity**: Ensuring content is appropriate across cultures
4. **Age Appropriateness**: Balancing safety with educational value

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{mininewsai2024,
  title={MiniNewsAI: Kid-Safe News Content Reframing System},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/ReshmaKoshy/MiniNewsAI}
}
```

## Contact

- **Author**: Reshma Koshy
- **Email**: reshmakoshy01@gmail.com
- **GitHub**: https://github.com/ReshmaKoshy/MiniNewsAI

## Acknowledgments

### Datasets

- Global News Dataset: https://www.kaggle.com/datasets/everydaycodings/global-news-dataset
- News Article Categories: https://www.kaggle.com/datasets/timilsinabimal/newsarticlecategories

### Models

- RoBERTa: Facebook AI Research
- BART: Facebook AI Research
- Mistral-7B-Instruct: Mistral AI
- Gemini 2.5 Pro: Google DeepMind
- DeepSeek LLM: DeepSeek AI
