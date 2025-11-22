"""
MiniNewsAI - Kid-Safe News Rewriter UI
A clean Gradio interface for classifying and rewriting news articles.
"""

import os
import torch
import gradio as gr
import numpy as np
import re
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel

# Project paths (app.py is in ui/ folder, so go up one level for project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIER_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'multiclass_classifier', 'best_model')
REWRITER_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'kid_safe_rewriter', 'best_model')
REWRITER_BASE_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global model variables
classifier_model = None
classifier_tokenizer = None
rewriter_model = None
rewriter_tokenizer = None

# Label mapping
LABEL_NAMES = ['SAFE', 'SENSITIVE', 'UNSAFE']
LABEL_COLORS = {
    'SAFE': '#28a745',      # Green
    'SENSITIVE': '#ffc107',  # Yellow/Orange
    'UNSAFE': '#dc3545'     # Red
}


def load_models():
    """Load both classifier and rewriter models. Models are cached in memory."""
    global classifier_model, classifier_tokenizer, rewriter_model, rewriter_tokenizer
    
    # Check if models are already loaded (cached in memory)
    if classifier_model is not None and rewriter_model is not None:
        return "✅ Models already loaded! (Using cached models in memory)"
    
    try:
        # Check if model paths exist
        if not os.path.exists(CLASSIFIER_MODEL_PATH):
            return f"❌ Error: Classifier model not found at {CLASSIFIER_MODEL_PATH}\nPlease ensure the model is trained and saved."
        
        if not os.path.exists(REWRITER_MODEL_PATH):
            return f"❌ Error: Rewriter model not found at {REWRITER_MODEL_PATH}\nPlease ensure the model is trained and saved."
        
        print("Loading models...")
        
        # Load classifier
        print(f"Loading classifier from {CLASSIFIER_MODEL_PATH}...")
        classifier_tokenizer = RobertaTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
        
        # Check if model file exists and is valid
        model_file = os.path.join(CLASSIFIER_MODEL_PATH, "model.safetensors")
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file)
            # RoBERTa-base should be ~500MB, if much smaller it's likely corrupted
            if file_size < 1000000:  # Less than 1MB
                raise Exception(
                    f"Model file appears corrupted (size: {file_size/1024:.1f}KB, expected ~500MB). "
                    f"Please re-train the model using notebook 02_multiclass_classifier_training.ipynb"
                )
        
        # Try loading with different methods to handle version mismatches
        try:
            # First try with safetensors (default)
            classifier_model = RobertaForSequenceClassification.from_pretrained(
                CLASSIFIER_MODEL_PATH,
                use_safetensors=True
            )
        except Exception as e1:
            error_msg1 = str(e1)
            print(f"Warning: Failed to load with safetensors: {error_msg1[:200]}")
            try:
                # Try without safetensors (use pickle format if available)
                classifier_model = RobertaForSequenceClassification.from_pretrained(
                    CLASSIFIER_MODEL_PATH,
                    use_safetensors=False
                )
            except Exception as e2:
                error_msg2 = str(e2)
                print(f"Warning: Failed to load without safetensors: {error_msg2[:200]}")
                # Provide helpful error message
                raise Exception(
                    f"Failed to load classifier model. The model file may be corrupted or incompatible.\n\n"
                    f"Error details:\n"
                    f"- Safetensors: {error_msg1[:150]}\n"
                    f"- Pickle: {error_msg2[:150]}\n\n"
                    f"Solution: Please re-train the model using:\n"
                    f"  jupyter notebook notebooks/02_multiclass_classifier_training.ipynb"
                )
        
        classifier_model = classifier_model.to(device)
        classifier_model.eval()
        print("✓ Classifier loaded")
        
        # Load rewriter
        print(f"Loading rewriter base model: {REWRITER_BASE_MODEL}...")
        rewriter_tokenizer = AutoTokenizer.from_pretrained(REWRITER_BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            REWRITER_BASE_MODEL,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print(f"Loading rewriter adapter from {REWRITER_MODEL_PATH}...")
        rewriter_model = PeftModel.from_pretrained(base_model, REWRITER_MODEL_PATH)
        rewriter_model = rewriter_model.to(device) if not torch.cuda.is_available() else rewriter_model
        rewriter_model.eval()
        print("✓ Rewriter loaded")
        
        return "✅ Models loaded successfully! You can now process articles."
    
    except Exception as e:
        error_msg = f"❌ Error loading models: {str(e)}\n\nPlease check:\n1. Model files exist in the correct paths\n2. All dependencies are installed\n3. You have sufficient memory (GPU/CPU)"
        print(error_msg)
        return error_msg


def smart_truncate_article(article_text, tokenizer, max_tokens, min_tokens_to_keep=None):
    """
    Smart truncation: Keep full article if short enough, otherwise use sentence-based scoring.
    
    Args:
        article_text: Original article text
        tokenizer: Tokenizer to count tokens
        max_tokens: Maximum tokens allowed
        min_tokens_to_keep: Minimum tokens to keep (default: 80% of max_tokens)
    
    Returns:
        Truncated article text
    """
    if min_tokens_to_keep is None:
        min_tokens_to_keep = int(max_tokens * 0.8)  # Keep at least 80% of max
    
    # Count tokens in original article
    article_tokens = tokenizer.encode(article_text, add_special_tokens=False)
    article_token_count = len(article_tokens)
    
    # If article is already short enough, return as-is
    if article_token_count <= max_tokens:
        return article_text
    
    # If article is too short after truncation, return as-is (better to truncate later)
    if article_token_count < min_tokens_to_keep:
        return article_text
    
    # Smart truncation: sentence-based scoring
    # Split into sentences (simple regex-based, lightweight)
    sentences = re.split(r'(?<=[.!?])\s+', article_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        # Fallback: truncate by tokens
        truncated_tokens = article_tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # Score each sentence
    sentence_scores = []
    total_sentences = len(sentences)
    
    for idx, sentence in enumerate(sentences):
        # Position-based score (earlier sentences more important)
        # Linear decay: first sentence = 1.0, last = 0.3
        pos_score = 1.0 - (idx / total_sentences) * 0.7
        
        # Length-based score (longer sentences often more informative)
        # Normalize by average sentence length
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        avg_length = article_token_count / total_sentences
        length_score = min(sentence_tokens / avg_length, 2.0) / 2.0  # Cap at 2x average
        
        # Keyword importance (simple: check for important words)
        # News articles often have important info in first few sentences
        keyword_bonus = 0.0
        important_keywords = ['announced', 'reported', 'according', 'study', 'research', 
                             'discovered', 'found', 'revealed', 'said', 'according to']
        if any(kw in sentence.lower() for kw in important_keywords):
            keyword_bonus = 0.1
        
        # Combined score (weighted)
        # 0.50*position + 0.30*length + 0.20*keyword (similar to user's formula)
        score = 0.50 * pos_score + 0.30 * length_score + 0.20 * keyword_bonus
        
        sentence_scores.append((score, idx, sentence, sentence_tokens))
    
    # Sort by score (highest first)
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select sentences until we hit token limit
    selected_indices = set()
    total_tokens = 0
    
    for score, idx, sentence, sent_tokens in sentence_scores:
        if total_tokens + sent_tokens <= max_tokens:
            selected_indices.add(idx)
            total_tokens += sent_tokens
        else:
            # Try to fit at least one more sentence if we're close
            if total_tokens < min_tokens_to_keep and sent_tokens <= (max_tokens - total_tokens):
                selected_indices.add(idx)
                total_tokens += sent_tokens
            break
    
    # Reconstruct article in original order
    truncated_sentences = [sentences[i] for i in sorted(selected_indices)]
    truncated_article = ' '.join(truncated_sentences)
    
    # Final safety check: if still too long, truncate by tokens
    final_tokens = tokenizer.encode(truncated_article, add_special_tokens=False)
    if len(final_tokens) > max_tokens:
        final_tokens = final_tokens[:max_tokens]
        truncated_article = tokenizer.decode(final_tokens, skip_special_tokens=True)
    
    return truncated_article


def create_instruction_prompt(label, article, title):
    """Create instruction prompt for the rewriter model."""
    instruction = f"""Instruction: Rewrite the article below into a kid-safe format according to its label ({label}).

Label: {label}

Title: {title}

Input Article:
{article}

Output:"""
    return instruction


def classify_article(article_text):
    """Classify article into SAFE, SENSITIVE, or UNSAFE."""
    if classifier_model is None or classifier_tokenizer is None:
        raise ValueError("Classifier model not loaded. Please click 'Load Models' first.")
    
    if not article_text or not article_text.strip():
        return None, None, None
    
    # Smart truncate article to max 512 tokens (classifier context window)
    # Leave some buffer for special tokens
    truncated_article = smart_truncate_article(
        article_text, 
        classifier_tokenizer, 
        max_tokens=510  # Leave 2 tokens for special tokens
    )
    
    # Tokenize
    inputs = classifier_tokenizer(
        truncated_article,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Get probabilities
    probabilities = probs[0].cpu().numpy()
    
    # Get label name
    predicted_label = LABEL_NAMES[predicted_class]
    confidence = float(probabilities[predicted_class])
    
    # Create confidence scores dict
    confidence_scores = {
        LABEL_NAMES[i]: float(probabilities[i]) 
        for i in range(len(LABEL_NAMES))
    }
    
    return predicted_label, confidence, confidence_scores


def rewrite_article(article_text, title, label):
    """Rewrite article to be kid-safe."""
    if rewriter_model is None or rewriter_tokenizer is None:
        raise ValueError("Rewriter model not loaded. Please click 'Load Models' first.")
    
    if not article_text or not article_text.strip():
        return ""
    
    if label == 'UNSAFE':
        return "⚠️ This article is classified as UNSAFE and cannot be rewritten. Please use a different article."
    
    # Smart truncate article before creating prompt
    # Instruction takes ~80 tokens, so article gets ~430 tokens
    INSTRUCTION_TOKEN_BUDGET = 80
    ARTICLE_MAX_TOKENS = 512 - INSTRUCTION_TOKEN_BUDGET  # ~432 tokens
    
    truncated_article = smart_truncate_article(
        article_text,
        rewriter_tokenizer,
        max_tokens=ARTICLE_MAX_TOKENS,
        min_tokens_to_keep=int(ARTICLE_MAX_TOKENS * 0.7)  # Keep at least 70% for short articles
    )
    
    # Create prompt with truncated article
    instruction = create_instruction_prompt(label, truncated_article, title)
    prompt = f"[INST] {instruction} [/INST]\n\n"
    
    # Tokenize (final safety truncation)
    inputs = rewriter_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate (using same parameters as training/validation)
    # Note: During training, we used: max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True
    # We use slightly higher max_new_tokens for SENSITIVE (longer rewrites needed)
    rewriter_model.eval()
    with torch.no_grad():
        # Use same max_new_tokens as training (256) or slightly higher for SENSITIVE
        max_new_tokens = 350 if label == 'SENSITIVE' else 256
        
        outputs = rewriter_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,  # Same as training
            top_p=0.9,  # Same as training
            do_sample=True,  # Same as training
            pad_token_id=rewriter_tokenizer.pad_token_id,
            eos_token_id=rewriter_tokenizer.eos_token_id
        )
    
    # Decode
    generated = rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after [/INST])
    if "[/INST]" in generated:
        generated = generated.split("[/INST]")[-1].strip()
    
    return generated


def process_article(article_text, title=""):
    """Main processing function: classify and rewrite."""
    try:
        if not article_text or not article_text.strip():
            return (
                "<div style='color: #666; padding: 10px;'>Please enter an article to process.</div>",
                "",
                "",
                "",
                ""
            )
        
        # Check if models are loaded
        if classifier_model is None or classifier_tokenizer is None:
            error_msg = "<div style='color: red; padding: 10px; background: #ffe6e6; border-left: 4px solid red;'><strong>⚠️ Models not loaded!</strong><br>Please click '⚙️ Load Models' first.</div>"
            return (error_msg, "", "", "✗ Models not loaded", "")
        
        # Default title if not provided
        if not title or not title.strip():
            title = "Untitled Article"
        
        # Classify
        predicted_label, confidence, confidence_scores = classify_article(article_text)
        
        if predicted_label is None:
            return (
                "<div style='color: red; padding: 10px;'>Error: Could not classify article.</div>",
                "",
                "",
                "",
                ""
            )
    except ValueError as e:
        error_msg = f"<div style='color: red; padding: 10px; background: #ffe6e6; border-left: 4px solid red;'><strong>Error:</strong> {str(e)}</div>"
        return (error_msg, "", "", "✗ Error", "")
    except Exception as e:
        error_msg = f"<div style='color: red; padding: 10px; background: #ffe6e6; border-left: 4px solid red;'><strong>Error:</strong> {str(e)}</div>"
        return (error_msg, "", "", "✗ Error", "")
    
    # Create confidence display
    confidence_html = f"""
    <div style="margin: 10px 0;">
        <h3 style="color: {LABEL_COLORS[predicted_label]}; margin-bottom: 10px;">
            Classification: {predicted_label} (Confidence: {confidence:.1%})
        </h3>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; color: #000;">
            <p style="margin: 5px 0; color: #000;"><strong>SAFE:</strong> {confidence_scores['SAFE']:.1%}</p>
            <p style="margin: 5px 0; color: #000;"><strong>SENSITIVE:</strong> {confidence_scores['SENSITIVE']:.1%}</p>
            <p style="margin: 5px 0; color: #000;"><strong>UNSAFE:</strong> {confidence_scores['UNSAFE']:.1%}</p>
        </div>
    </div>
    """
    
    # Rewrite if SAFE or SENSITIVE
    try:
        if predicted_label in ['SAFE', 'SENSITIVE']:
            rewritten_text = rewrite_article(article_text, title, predicted_label)
            rewrite_status = f"✓ Rewritten as {predicted_label}"
        else:
            rewritten_text = "⚠️ This article is classified as UNSAFE and cannot be rewritten for children."
            rewrite_status = "✗ Cannot rewrite UNSAFE content"
    except ValueError as e:
        rewritten_text = f"Error: {str(e)}"
        rewrite_status = "✗ Error during rewriting"
    except Exception as e:
        rewritten_text = f"Error during rewriting: {str(e)}"
        rewrite_status = "✗ Error during rewriting"
    
    # Format original article display
    original_display = f"""
    <div style="background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; color: #000;">
        <h4 style="margin-top: 0; color: #000;">Original Article</h4>
        <p style="white-space: pre-wrap; margin: 0; color: #000;">{article_text}</p>
    </div>
    """
    
    # Format rewritten article display
    if predicted_label in ['SAFE', 'SENSITIVE']:
        rewrite_display = f"""
        <div style="background: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; color: #000;">
            <h4 style="margin-top: 0; color: #000;">Kid-Safe Rewrite ({predicted_label})</h4>
            <p style="white-space: pre-wrap; margin: 0; color: #000;">{rewritten_text}</p>
        </div>
        """
    else:
        rewrite_display = f"""
        <div style="background: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; color: #000;">
            <h4 style="margin-top: 0; color: #000;">Cannot Rewrite</h4>
            <p style="white-space: pre-wrap; margin: 0; color: #000;">{rewritten_text}</p>
        </div>
        """
    
    return (
        confidence_html,
        original_display,
        rewrite_display,
        rewrite_status,
        rewritten_text
    )


# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="MiniNewsAI - Kid-Safe News Rewriter") as demo:
        gr.Markdown(
            """
            # 🗞️ MiniNewsAI - Kid-Safe News Rewriter
            
            This tool classifies news articles and rewrites them to be safe for children.
            
            **How it works:**
            1. Enter a news article (and optional title)
            2. The AI classifies it as **SAFE**, **SENSITIVE**, or **UNSAFE**
            3. If SAFE or SENSITIVE, the article is automatically rewritten for children
            4. View confidence scores and the rewritten version
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                article_input = gr.Textbox(
                    label="📝 News Article",
                    placeholder="Paste your news article here...",
                    lines=10,
                    max_lines=20
                )
                title_input = gr.Textbox(
                    label="📰 Article Title (Optional)",
                    placeholder="Enter article title (optional)...",
                    lines=1
                )
                process_btn = gr.Button("🚀 Process Article", variant="primary", size="lg")
                
                gr.Markdown("### Model Status")
                model_status = gr.Textbox(
                    label="",
                    value="⚠️ Models not loaded. Click '⚙️ Load Models' to initialize.\n\n💡 Note: Models stay in memory after loading. Restart the app to reload.",
                    interactive=False,
                    lines=4
                )
                load_models_btn = gr.Button("⚙️ Load Models", variant="secondary", size="lg")
        
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Classification Results")
                classification_output = gr.HTML(label="Classification")
                
                gr.Markdown("### 📄 Original Article")
                original_output = gr.HTML(label="Original")
                
                gr.Markdown("### ✨ Kid-Safe Rewrite")
                rewrite_output = gr.HTML(label="Rewritten")
                rewrite_status = gr.Textbox(label="Status", interactive=False)
        
        # Hidden output for raw rewritten text (for copying)
        raw_rewrite = gr.Textbox(visible=False)
        
        # Examples
        gr.Markdown("---")
        gr.Markdown("### 📚 Example Articles")
        examples = gr.Examples(
            examples=[
                [
                    "Scientists have discovered a new species of butterfly in the Amazon rainforest. The butterfly has bright blue wings and is about the size of a quarter. Researchers are excited about this discovery because it helps us understand more about biodiversity.",
                    "New Butterfly Species Discovered"
                ],
                [
                    "A local community center opened a new playground for children. The playground includes swings, slides, and climbing equipment. Parents are happy that their children have a safe place to play.",
                    "New Playground Opens"
                ]
            ],
            inputs=[article_input, title_input],
            label="Click an example to try it out!"
        )
        
        # Event handlers
        load_models_btn.click(
            fn=load_models,
            outputs=model_status
        )
        
        process_btn.click(
            fn=process_article,
            inputs=[article_input, title_input],
            outputs=[classification_output, original_output, rewrite_output, rewrite_status, raw_rewrite]
        )
        
        gr.Markdown(
            """
            ---
            **Note:** Models need to be loaded before processing articles. Click "Load Models" first.
            
            **Classification Labels:**
            - 🟢 **SAFE**: Content is already appropriate for children
            - 🟡 **SENSITIVE**: Content needs rewriting to be child-friendly
            - 🔴 **UNSAFE**: Content cannot be made safe for children
            """
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

