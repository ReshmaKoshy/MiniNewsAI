"""
MiniNewsAI - Kid-Safe News Rewriter UI
A clean Gradio interface for classifying and rewriting news articles.
"""

import os
import sys
import torch
import gradio as gr
import numpy as np
import re
import gc
import html
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIER_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'multiclass_classifier', 'best_model')
REWRITER_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'kid_safe_rewriter', 'best_model')
REWRITER_BASE_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

# Global model variables
classifier_model = None
classifier_tokenizer = None
rewriter_model = None
rewriter_tokenizer = None

# Label mapping
LABEL_NAMES = ['SAFE', 'SENSITIVE', 'UNSAFE']
LABEL_COLORS = {
    'SAFE': '#28a745',
    'SENSITIVE': '#ffc107',
    'UNSAFE': '#dc3545'
}


def log(message):
    """Log with immediate flush."""
    print(message, flush=True)


def load_models():
    """Load both classifier and rewriter models."""
    global classifier_model, classifier_tokenizer, rewriter_model, rewriter_tokenizer
    
    log("=" * 80)
    log("LOADING MODELS")
    log("=" * 80)
    
    # Check if already loaded
    if classifier_model is not None and rewriter_model is not None:
        log("✓ Models already loaded in memory")
        return "✅ Models already loaded! (Using cached models)"
    
    try:
        # Load classifier
        log(f"Loading classifier from {CLASSIFIER_MODEL_PATH}...")
        classifier_tokenizer = RobertaTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
        
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        classifier_model = RobertaForSequenceClassification.from_pretrained(
            CLASSIFIER_MODEL_PATH,
            torch_dtype=model_dtype
        )
        classifier_model = classifier_model.to(device)
        classifier_model.eval()
        log("✓ Classifier loaded")
        
        # Load rewriter
        log(f"Loading rewriter base model: {REWRITER_BASE_MODEL}...")
        rewriter_tokenizer = AutoTokenizer.from_pretrained(REWRITER_BASE_MODEL)
        
        if torch.cuda.is_available():
            base_model = AutoModelForCausalLM.from_pretrained(
                REWRITER_BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                REWRITER_BASE_MODEL,
                torch_dtype=torch.float32
            )
            base_model = base_model.to(device)
        
        log(f"Loading rewriter adapter from {REWRITER_MODEL_PATH}...")
        rewriter_model = PeftModel.from_pretrained(base_model, REWRITER_MODEL_PATH)
        
        if torch.cuda.is_available():
            model_device = next(rewriter_model.parameters()).device
            if model_device.type != 'cuda':
                rewriter_model = rewriter_model.to(device)
        else:
            rewriter_model = rewriter_model.to(device)
        
        rewriter_model.eval()
        log("✓ Rewriter loaded")
        log("=" * 80)
        
        return "✅ Models loaded successfully! You can now process articles."
    
    except Exception as e:
        error_msg = f"❌ Error loading models: {str(e)}"
        log(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def smart_truncate_article(article_text, max_tokens=430, tokenizer=None):
    """Smart truncation using sentence-based scoring."""
    if not article_text or not article_text.strip():
        return article_text
    
    if tokenizer is None:
        # Fallback: simple character-based truncation
        estimated_tokens = len(article_text) * 0.75
        if estimated_tokens <= max_tokens:
            return article_text
        max_chars = int(max_tokens / 0.75)
        return article_text[:max_chars] + "..."
    
    try:
        tokens = tokenizer.encode(article_text, add_special_tokens=False, max_length=10000, truncation=True)
    except Exception as e:
        log(f"  Warning: Tokenization error, using simple truncation: {e}")
        max_chars = int(max_tokens / 0.75)
        return article_text[:max_chars] + "..."
    
    if len(tokens) <= max_tokens:
        return article_text
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', article_text)
    if len(sentences) <= 1:
        # No sentence breaks, use simple truncation
        final_tokens = tokens[:max_tokens]
        return tokenizer.decode(final_tokens, skip_special_tokens=True)
    
    # Score sentences (position bias: earlier sentences more important)
    scored_sentences = []
    for i, sent in enumerate(sentences):
        sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
        position_score = 1.0 / (1.0 + i * 0.1)  # Earlier sentences get higher score
        length_score = min(len(sent_tokens) / 50.0, 1.0)  # Prefer medium-length sentences
        score = 0.7 * position_score + 0.3 * length_score
        scored_sentences.append((score, len(sent_tokens), sent))
    
    # Select sentences greedily
    selected_sentences = []
    total_tokens = 0
    for score, sent_tokens, sent in sorted(scored_sentences, reverse=True):
        if total_tokens + sent_tokens <= max_tokens:
            selected_sentences.append((score, sent))
            total_tokens += sent_tokens
    
    # Sort selected sentences by original position
    selected_sentences.sort(key=lambda x: sentences.index(x[1]))
    truncated_article = ' '.join([sent for _, sent in selected_sentences])
    
    # Final check: if still too long, do hard truncation
    final_tokens = tokenizer.encode(truncated_article, add_special_tokens=False)
    if len(final_tokens) > max_tokens:
        final_tokens = final_tokens[:max_tokens]
        truncated_article = tokenizer.decode(final_tokens, skip_special_tokens=True)
    
    return truncated_article


def classify_article(article_text):
    """Classify article into SAFE, SENSITIVE, or UNSAFE."""
    log("  [Classification] Starting...")
    
    if classifier_model is None or classifier_tokenizer is None:
        raise ValueError("Classifier model not loaded")
    
    if not article_text or not article_text.strip():
        return None, None, None
    
    classifier_model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    inputs = None
    outputs = None
    logits = None
    probs = None
    
    try:
        inputs = classifier_tokenizer(
            article_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = classifier_model(**inputs)
            logits = outputs.logits.float()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        probabilities = probs[0].cpu().float().numpy()
        predicted_label = LABEL_NAMES[predicted_class]
        confidence = float(probabilities[predicted_class])
        
        confidence_scores = {
            LABEL_NAMES[i]: float(probabilities[i]) 
            for i in range(len(LABEL_NAMES))
        }
        
        log(f"  [Classification] Result: {predicted_label} (confidence: {confidence:.2%})")
        return predicted_label, confidence, confidence_scores
    
    finally:
        if inputs is not None:
            del inputs
        if outputs is not None:
            del outputs
        if logits is not None:
            del logits
        if probs is not None:
            del probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def rewrite_article(article_text, title, label):
    """Rewrite article to be kid-safe."""
    log(f"  [Rewriting] Starting for label: {label}")
    
    if rewriter_model is None or rewriter_tokenizer is None:
        raise ValueError("Rewriter model not loaded")
    
    if not article_text or not article_text.strip():
        return ""
    
    if label == 'UNSAFE':
        return "⚠️ This article is classified as UNSAFE and cannot be rewritten."
    
    rewriter_model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    inputs = None
    outputs = None
    
    try:
        # Create prompt
        if label == 'SENSITIVE':
            instruction = f"Rewrite the following news article to make it appropriate and safe for children. Maintain the key facts and information, but simplify language, remove any sensitive or inappropriate content, and make it engaging for young readers.\n\nArticle: {article_text}"
        else:  # SAFE
            instruction = f"Rewrite the following news article to make it more engaging and age-appropriate for children while keeping all the important information.\n\nArticle: {article_text}"
        
        prompt = f"[INST] {instruction} [/INST]\n\n"
        
        log("  [Rewriting] Tokenizing prompt...")
        inputs = rewriter_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        model_device = next(rewriter_model.parameters()).device
        if inputs['input_ids'].device != model_device:
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        max_new_tokens = 350 if label == 'SENSITIVE' else 256
        log(f"  [Rewriting] Generating (max_new_tokens: {max_new_tokens})...")
        
        with torch.no_grad():
            outputs = rewriter_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=rewriter_tokenizer.pad_token_id,
                eos_token_id=rewriter_tokenizer.eos_token_id,
                early_stopping=True
            )
        
        log("  [Rewriting] Decoding output...")
        generated = rewriter_tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        
        if "[/INST]" in generated:
            generated = generated.split("[/INST]")[-1].strip()
        
        log(f"  [Rewriting] Complete (output length: {len(generated)} chars)")
        return generated
    
    except Exception as e:
        log(f"  [Rewriting] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if inputs is not None:
            del inputs
        if outputs is not None:
            del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def process_article(article_text, title=""):
    """Main processing function: classify and rewrite."""
    # CRITICAL: Force immediate output - multiple flushes to ensure visibility
    sys.stdout.write("\n" + "=" * 80 + "\n")
    sys.stdout.write("PROCESS_ARTICLE CALLED\n")
    sys.stdout.write("=" * 80 + "\n")
    sys.stdout.write(f"Article length: {len(article_text) if article_text else 0} chars\n")
    sys.stdout.write(f"Title: {title}\n")
    sys.stdout.flush()
    sys.stderr.write("PROCESS_ARTICLE CALLED (stderr)\n")
    sys.stderr.flush()
    
    # Also use print with flush as backup
    print("\n" + "=" * 80, flush=True)
    print("PROCESS_ARTICLE CALLED", flush=True)
    print("=" * 80, flush=True)
    print(f"Article length: {len(article_text) if article_text else 0} chars", flush=True)
    print(f"Title: {title}", flush=True)
    
    if not article_text or not article_text.strip():
        return (
            "<div style='color: #666; padding: 10px;'>Please enter an article to process.</div>",
            "",
            "",
            "",
            ""
        )
    
    if classifier_model is None or classifier_tokenizer is None:
        error_msg = "<div style='color: red; padding: 10px;'><strong>⚠️ Models not loaded!</strong><br>Please click '⚙️ Load Models' first.</div>"
        return (error_msg, "", "", "✗ Models not loaded", "")
    
    if not title or not title.strip():
        title = "Untitled Article"
    
    try:
        # Step 1: Smart truncation
        log("Step 1: Smart truncation...")
        truncated_article = smart_truncate_article(
            article_text, 
            max_tokens=430,
            tokenizer=classifier_tokenizer
        )
        was_truncated = truncated_article != article_text
        log(f"✓ Truncation complete (truncated: {was_truncated})")
        
        truncation_note = ""
        if was_truncated:
            truncation_note = f"<small style='color: #666;'>(Article truncated from {len(article_text.split())} to ~{len(truncated_article.split())} words)</small>"
        
        # Step 2: Classification
        log("Step 2: Classification...")
        predicted_label, confidence, confidence_scores = classify_article(truncated_article)
        log(f"✓ Classification: {predicted_label} (confidence: {confidence:.2%})")
        
        if predicted_label is None:
            return (
                "<div style='color: red; padding: 10px;'>Error: Could not classify article.</div>",
                "",
                "",
                "",
                ""
            )
        
        # Create confidence display
        confidence_html = f"""
        <div style="margin: 10px 0;">
            <h3 style="color: {LABEL_COLORS[predicted_label]}; margin-bottom: 10px;">
                Classification: {predicted_label} (Confidence: {confidence:.1%})
            </h3>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                <p style="margin: 5px 0;">
                    <strong style="color: {LABEL_COLORS['SAFE']}; font-size: 14px; font-weight: bold;">🟢 SAFE:</strong> 
                    <span style="color: #000;">{confidence_scores['SAFE']:.1%}</span>
                </p>
                <p style="margin: 5px 0;">
                    <strong style="color: {LABEL_COLORS['SENSITIVE']}; font-size: 14px; font-weight: bold;">🟡 SENSITIVE:</strong> 
                    <span style="color: #000;">{confidence_scores['SENSITIVE']:.1%}</span>
                </p>
                <p style="margin: 5px 0;">
                    <strong style="color: {LABEL_COLORS['UNSAFE']}; font-size: 14px; font-weight: bold;">🔴 UNSAFE:</strong> 
                    <span style="color: #000;">{confidence_scores['UNSAFE']:.1%}</span>
                </p>
            </div>
        </div>
        """
        
        # Step 3: Rewriting
        if predicted_label in ['SAFE', 'SENSITIVE']:
            log(f"Step 3: Rewriting ({predicted_label})...")
            try:
                rewritten_text = rewrite_article(truncated_article, title, predicted_label)
                log(f"✓ Rewriting complete (output length: {len(rewritten_text)} chars)")
                rewrite_status = f"✓ Rewritten as {predicted_label}"
            except Exception as e:
                log(f"✗ Rewriting error: {str(e)}")
                rewritten_text = f"Error during rewriting: {str(e)}"
                rewrite_status = "✗ Error during rewriting"
        else:
            rewritten_text = "⚠️ This article is classified as UNSAFE and cannot be rewritten for children."
            rewrite_status = "✗ Cannot rewrite UNSAFE content"
        
        # Format displays
        truncated_article_escaped = html.escape(truncated_article)
        original_display = f"""
        <div style="background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; color: #000;">
            <h4 style="margin-top: 0; color: #000;">Original Article {truncation_note}</h4>
            <p style="white-space: pre-wrap; margin: 0; color: #000;">{truncated_article_escaped}</p>
        </div>
        """
        
        rewritten_text_escaped = html.escape(rewritten_text)
        if predicted_label in ['SAFE', 'SENSITIVE']:
            rewrite_display = f"""
            <div style="background: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; color: #000;">
                <h4 style="margin-top: 0; color: #000;">Kid-Safe Rewrite ({predicted_label})</h4>
                <p style="white-space: pre-wrap; margin: 0; color: #000;">{rewritten_text_escaped}</p>
            </div>
            """
        else:
            rewrite_display = f"""
            <div style="background: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; color: #000;">
                <h4 style="margin-top: 0; color: #000;">Cannot Rewrite</h4>
                <p style="white-space: pre-wrap; margin: 0; color: #000;">{rewritten_text_escaped}</p>
            </div>
            """
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        log("=" * 80)
        log("PROCESS_ARTICLE COMPLETED")
        log("=" * 80)
        log("")
        sys.stdout.flush()
        
        return (
            confidence_html,
            original_display,
            rewrite_display,
            rewrite_status,
            rewritten_text
        )
        
    except Exception as e:
        log(f"ERROR in process_article: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        error_msg = f"<div style='color: red; padding: 10px;'><strong>Error:</strong> {str(e)}</div>"
        return (error_msg, "", "", "✗ Error", "")


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="MiniNewsAI - Kid-Safe News Rewriter") as demo:
        gr.Markdown(
            """
            # 🗞️ MiniNewsAI - Kid-Safe News Rewriter
            
            This tool classifies news articles and rewrites them to be safe for children.
            
            **How it works:**
            1. Click "⚙️ Load Models" to initialize the AI models
            2. Enter a news article (and optional title)
            3. Click "🚀 Process Article"
            4. View the classification and kid-safe rewrite
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model Management")
                model_status = gr.Textbox(
                    label="",
                    value="⚠️ Models not loaded. Click '⚙️ Load Models' to initialize.",
                    interactive=False,
                    lines=3
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
        
        # Hidden output for raw rewritten text
        raw_rewrite = gr.Textbox(visible=False)
        
        with gr.Row():
            with gr.Column():
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
            **Classification Labels:**
            - 🟢 **SAFE**: Content is already appropriate for children
            - 🟡 **SENSITIVE**: Content needs rewriting to be child-friendly
            - 🔴 **UNSAFE**: Content cannot be made safe for children
            """
        )
    
    return demo


if __name__ == "__main__":
    # Force unbuffered output for immediate log visibility
    import sys
    sys.stdout = sys.__stdout__  # Ensure we're using real stdout
    sys.stderr = sys.__stderr__  # Ensure we're using real stderr
    
    # Set Python to unbuffered mode
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    print("Starting MiniNewsAI app...", flush=True)
    print("Python stdout is unbuffered", flush=True)
    
    demo = create_interface()
    
    print("Interface created, launching...", flush=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
