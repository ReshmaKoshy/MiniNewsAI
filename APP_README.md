# MiniNewsAI - Kid-Safe News Rewriter UI

A clean, user-friendly web interface for classifying and rewriting news articles to be safe for children.

## Features

- **Article Classification**: Automatically classifies articles as SAFE, SENSITIVE, or UNSAFE
- **Confidence Scores**: Shows confidence percentages for each classification category
- **Automatic Rewriting**: Rewrites SAFE and SENSITIVE articles to be kid-friendly
- **Clean UI**: Modern, intuitive interface built with Gradio

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your models are trained and saved in:
   - `models/multiclass_classifier/best_model/` (classifier)
   - `models/kid_safe_rewriter/best_model/` (rewriter)

## Usage

### Running the App

```bash
python app.py
```

The app will start on `http://localhost:7860` by default.

### Using the Interface

1. **Load Models**: Click "⚙️ Load Models" to initialize the classifier and rewriter models (required before processing)

2. **Enter Article**: 
   - Paste your news article in the text box
   - Optionally add a title

3. **Process**: Click "🚀 Process Article" to:
   - Classify the article (SAFE/SENSITIVE/UNSAFE)
   - See confidence scores for each category
   - Get a kid-safe rewrite (if applicable)

### Classification Labels

- 🟢 **SAFE**: Content is already appropriate for children
- 🟡 **SENSITIVE**: Content needs rewriting to be child-friendly  
- 🔴 **UNSAFE**: Content cannot be made safe for children

### Output

The interface displays:
- **Classification Results**: Label and confidence scores
- **Original Article**: The input article
- **Kid-Safe Rewrite**: The rewritten version (if SAFE or SENSITIVE)

## Model Requirements

- **Classifier**: RoBERTa-base fine-tuned for 3-way classification
- **Rewriter**: Mistral-7B-Instruct with LoRA adapter for kid-safe rewriting

## Technical Details

- Uses GPU if available (CUDA), falls back to CPU
- Classifier uses RoBERTa tokenizer with max length 512
- Rewriter uses Mistral tokenizer with dynamic max_new_tokens (512-768)
- Generation parameters: temperature=0.7, top_p=0.9, repetition_penalty=1.2

## Troubleshooting

**Models not loading?**
- Check that model directories exist and contain the required files
- Ensure you have enough GPU/CPU memory
- Check that transformers and peft libraries are installed

**Slow processing?**
- First run will be slower (model loading)
- GPU acceleration significantly speeds up inference
- Consider reducing max_new_tokens if generation is too slow

**Port already in use?**
- Change the port in `app.py`: `demo.launch(server_port=7861)`

## Example

**Input:**
```
Title: New Playground Opens
Article: A local community center opened a new playground for children. 
The playground includes swings, slides, and climbing equipment. 
Parents are happy that their children have a safe place to play.
```

**Output:**
- Classification: SAFE (95.2% confidence)
- Rewritten version: Simplified, kid-friendly version of the article

