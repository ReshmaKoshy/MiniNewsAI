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
        # Use bfloat16 on GPU (B200 optimized), float32 on CPU
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            # First try with safetensors (default)
            classifier_model = RobertaForSequenceClassification.from_pretrained(
                CLASSIFIER_MODEL_PATH,
                use_safetensors=True,
                torch_dtype=model_dtype
            )
        except Exception as e1:
            error_msg1 = str(e1)
            print(f"Warning: Failed to load with safetensors: {error_msg1[:200]}")
            try:
                # Try without safetensors (use pickle format if available)
                classifier_model = RobertaForSequenceClassification.from_pretrained(
                    CLASSIFIER_MODEL_PATH,
                    use_safetensors=False,
                    torch_dtype=model_dtype
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
        
        # Use device_map="auto" only on GPU, manual device placement on CPU
        if torch.cuda.is_available():
            base_model = AutoModelForCausalLM.from_pretrained(
                REWRITER_BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print(f"✓ Base model loaded with device_map='auto'")
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                REWRITER_BASE_MODEL,
                torch_dtype=torch.float32
            )
            base_model = base_model.to(device)
            print(f"✓ Base model loaded to {device}")
        
        print(f"Loading rewriter adapter from {REWRITER_MODEL_PATH}...")
        rewriter_model = PeftModel.from_pretrained(base_model, REWRITER_MODEL_PATH)
        
        # Ensure model is on correct device (device_map="auto" might place it elsewhere)
        if torch.cuda.is_available():
            # Check where model is actually placed
            model_device = next(rewriter_model.parameters()).device
            print(f"  Model device after loading: {model_device}")
            if model_device.type != 'cuda':
                print(f"  Warning: Model not on CUDA, moving to {device}")
                rewriter_model = rewriter_model.to(device)
        else:
            rewriter_model = rewriter_model.to(device)
        
        rewriter_model.eval()
        print("✓ Rewriter loaded")
        
        return "✅ Models loaded successfully! You can now process articles."
    
    except Exception as e:
        error_msg = f"❌ Error loading models: {str(e)}\n\nPlease check:\n1. Model files exist in the correct paths\n2. All dependencies are installed\n3. You have sufficient memory (GPU/CPU)"
        print(error_msg)
        return error_msg


def smart_truncate_article(article_text, max_tokens=430, tokenizer=None):
    """
    Smart truncation of article using sentence-based scoring.
    Uses position bias (earlier sentences more important) and length-based scoring.
    
    Args:
        article_text: Full article text
        max_tokens: Maximum tokens to keep (default 430 for rewriter: 512 - 80 instruction)
        tokenizer: Tokenizer to use for counting tokens (uses classifier tokenizer if available)
    
    Returns:
        Truncated article text
    """
    if not article_text or not article_text.strip():
        return article_text
    
    # Use classifier tokenizer if available, otherwise estimate
    if tokenizer is None:
        # Fallback: estimate tokens as ~0.75 * characters (rough estimate)
        estimated_tokens = len(article_text) * 0.75
        if estimated_tokens <= max_tokens:
            return article_text
        # Simple character-based truncation as fallback
        max_chars = int(max_tokens / 0.75)
        return article_text[:max_chars] + "..."
    
    # Count tokens in full article (with timeout protection)
    try:
        tokens = tokenizer.encode(article_text, add_special_tokens=False, max_length=10000, truncation=True)
    except Exception as e:
        print(f"  Warning: Tokenization error, using simple truncation: {e}")
        # Fallback to simple truncation
        max_chars = int(max_tokens / 0.75)
        return article_text[:max_chars] + "..."
    
    if len(tokens) <= max_tokens:
        return article_text  # Already short enough, keep fully
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', article_text)
    if len(sentences) <= 1:
        # No sentence boundaries, use simple token truncation
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # Score each sentence
    sentence_scores = []
    total_sentences = len(sentences)
    
    for idx, sentence in enumerate(sentences):
        if not sentence.strip():
            sentence_scores.append((0, sentence, idx))
            continue
        
        # Tokenize sentence
        sent_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sent_length = len(sent_tokens)
        
        if sent_length == 0:
            sentence_scores.append((0, sentence, idx))
            continue
        
        # Position bias: earlier sentences are more important (0.30 weight)
        # First 30% of sentences get higher priority
        position_score = 0.30 * (1.0 - (idx / total_sentences)) if idx < total_sentences * 0.3 else 0.10 * (1.0 - (idx / total_sentences))
        
        # Length score: prefer sentences of moderate length (0.20 weight)
        # Ideal sentence length: 15-30 tokens
        if 15 <= sent_length <= 30:
            length_score = 0.20
        elif sent_length < 15:
            length_score = 0.10  # Too short
        else:
            length_score = 0.15  # Too long
        
        # Centrality score: sentences with important words (0.20 weight)
        # Simple heuristic: sentences with more capitalized words (proper nouns) or numbers
        important_words = len(re.findall(r'\b[A-Z][a-z]+\b', sentence)) + len(re.findall(r'\d+', sentence))
        centrality_score = 0.20 * min(important_words / 5.0, 1.0)  # Normalize to max 5 important words
        
        # Eventness: sentences with action verbs or question words (0.15 weight)
        action_indicators = len(re.findall(r'\b(announced|reported|discovered|created|launched|opened|started|began)\b', sentence, re.IGNORECASE))
        eventness_score = 0.15 * min(action_indicators / 2.0, 1.0)
        
        # Late update bonus: last few sentences might have updates (0.05 weight)
        late_update_bonus = 0.05 if idx >= total_sentences * 0.8 else 0.0
        
        # Total score
        total_score = position_score + length_score + centrality_score + eventness_score + late_update_bonus
        
        sentence_scores.append((total_score, sentence, idx, sent_length))
    
    # Sort by score (descending)
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select sentences until we reach max_tokens
    selected_sentences = []
    selected_indices = set()
    current_tokens = 0
    
    # First, always include the first sentence (lead)
    if sentences:
        first_sent_tokens = tokenizer.encode(sentences[0], add_special_tokens=False)
        if len(first_sent_tokens) <= max_tokens:
            selected_sentences.append((0, sentences[0]))
            selected_indices.add(0)
            current_tokens += len(first_sent_tokens)
    
    # Then add high-scoring sentences
    for score, sentence, idx, sent_length in sentence_scores:
        if idx in selected_indices:
            continue
        
        if current_tokens + sent_length <= max_tokens:
            selected_sentences.append((idx, sentence))
            selected_indices.add(idx)
            current_tokens += sent_length
        else:
            # Can't fit full sentence, break
            break
    
    # Sort selected sentences by original position
    selected_sentences.sort(key=lambda x: x[0])
    
    # Reconstruct article
    truncated_article = ' '.join([sent for _, sent in selected_sentences])
    
    # Final check: if still too long, do hard truncation
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
    """Classify article into SAFE, SENSITIVE, or UNSAFE.
    
    Note: article_text should already be truncated by process_article().
    """
    if classifier_model is None or classifier_tokenizer is None:
        raise ValueError("Classifier model not loaded. Please click 'Load Models' first.")
    
    if not article_text or not article_text.strip():
        return None, None, None
    
    # Tokenize (article is already truncated, but add safety truncation)
    inputs = classifier_tokenizer(
        article_text,
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


def rewrite_article(article_text, title, label, progress=None):
    """Rewrite article to be kid-safe."""
    if rewriter_model is None or rewriter_tokenizer is None:
        raise ValueError("Rewriter model not loaded. Please click 'Load Models' first.")
    
    if not article_text or not article_text.strip():
        return ""
    
    if label == 'UNSAFE':
        return "⚠️ This article is classified as UNSAFE and cannot be rewritten. Please use a different article."
    
    if progress:
        progress(0.6, desc="Preparing generation...")
    print(f"  Creating prompt (label: {label}, article length: {len(article_text)} chars)")
    # Create prompt
    instruction = create_instruction_prompt(label, article_text, title)
    prompt = f"[INST] {instruction} [/INST]\n\n"
    
    if progress:
        progress(0.65, desc="Tokenizing input...")
    print(f"  Tokenizing prompt")
    # Tokenize
    inputs = rewriter_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    print(f"  Input tokens: {inputs['input_ids'].shape[1]}")
    
    # Generate (using same parameters as training/validation)
    rewriter_model.eval()
    
    max_new_tokens = 350 if label == 'SENSITIVE' else 256
    print(f"  Starting generation (max_new_tokens: {max_new_tokens})...")
    print(f"  Device: {device}, Model device: {next(rewriter_model.parameters()).device}")
    
    try:
        with torch.no_grad():
            # Ensure inputs are on correct device
            if inputs['input_ids'].device != next(rewriter_model.parameters()).device:
                inputs = {k: v.to(next(rewriter_model.parameters()).device) for k, v in inputs.items()}
            
            if progress:
                progress(0.7, desc="Generating rewrite (this may take 30-60 seconds)...")
            
            outputs = rewriter_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,  # Same as training
                top_p=0.9,  # Same as training
                do_sample=True,  # Same as training
                pad_token_id=rewriter_tokenizer.pad_token_id,
                eos_token_id=rewriter_tokenizer.eos_token_id,
                early_stopping=True
            )
        print(f"  ✓ Generation complete (output shape: {outputs.shape})")
        if progress:
            progress(0.9, desc="Decoding output...")
    except RuntimeError as e:
        error_msg = str(e)
        print(f"  ✗ Generation RuntimeError: {error_msg}")
        if "out of memory" in error_msg.lower():
            raise Exception("GPU out of memory. Try reducing max_new_tokens or using a shorter article.")
        raise
    except Exception as e:
        print(f"  ✗ Generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Decode
    print(f"  Decoding output...")
    generated = rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after [/INST])
    if "[/INST]" in generated:
        generated = generated.split("[/INST]")[-1].strip()
    
    print(f"  ✓ Final output length: {len(generated)} chars")
    return generated


def process_article(article_text, title="", progress=None):
    """Main processing function: classify and rewrite.
    
    Smart truncation is applied once at the beginning, then the same
    truncated article is used for both classification and rewriting.
    
    Args:
        progress: Gradio Progress tracker for UI updates
    """
    try:
        print("=" * 80)
        print("PROCESSING ARTICLE")
        print("=" * 80)
        
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
        
        # Step 1: Smart truncation
        if progress:
            progress(0.1, desc="Truncating article...")
        print(f"Step 1: Smart truncation (article length: {len(article_text)} chars)")
        truncated_article = smart_truncate_article(
            article_text, 
            max_tokens=430,
            tokenizer=classifier_tokenizer
        )
        print(f"✓ Truncation complete (truncated: {truncated_article != article_text})")
        
        # Store original for display
        original_article = article_text
        if truncated_article != article_text:
            truncation_note = f"<small style='color: #666;'>(Article truncated from {len(article_text.split())} to ~{len(truncated_article.split())} words for processing)</small>"
        else:
            truncation_note = ""
        
        # Step 2: Classification
        if progress:
            progress(0.3, desc="Classifying article...")
        print(f"Step 2: Classification")
        predicted_label, confidence, confidence_scores = classify_article(truncated_article)
        print(f"✓ Classification: {predicted_label} (confidence: {confidence:.2%})")
        
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
    
    # Rewrite if SAFE or SENSITIVE (using same truncated article)
    try:
        if predicted_label in ['SAFE', 'SENSITIVE']:
            if progress:
                progress(0.5, desc=f"Rewriting article ({predicted_label})...")
            print(f"Step 3: Rewriting ({predicted_label})")
            rewritten_text = rewrite_article(truncated_article, title, predicted_label, progress)
            if progress:
                progress(0.95, desc="Finalizing...")
            print(f"✓ Rewriting complete (output length: {len(rewritten_text)} chars)")
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
    
    # Format original article display (show full original, note if truncated)
    original_display = f"""
    <div style="background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; color: #000;">
        <h4 style="margin-top: 0; color: #000;">Original Article {truncation_note}</h4>
        <p style="white-space: pre-wrap; margin: 0; color: #000;">{original_article}</p>
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
                with gr.Row():
                    process_btn = gr.Button("🚀 Process Article", variant="primary", size="lg")
                    cancel_btn = gr.Button("⏹️ Cancel", variant="stop", size="lg", visible=False)
                
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
        
        # Process button with cancellation support
        process_event = process_btn.click(
            fn=process_article,
            inputs=[article_input, title_input],
            outputs=[classification_output, original_output, rewrite_output, rewrite_status, raw_rewrite],
            show_progress="full"
        )
        
        # Cancel button functionality
        def toggle_cancel_btn():
            return gr.update(visible=True)
        
        def hide_cancel_btn():
            return gr.update(visible=False)
        
        # Show cancel button when processing starts
        process_btn.click(
            fn=toggle_cancel_btn,
            outputs=cancel_btn
        )
        
        # Hide cancel button and cancel processing when cancel is clicked
        cancel_btn.click(
            fn=lambda: (gr.update(visible=False),),
            outputs=[cancel_btn],
            cancels=[process_event]
        )
        
        # Hide cancel button when processing completes
        process_event.then(
            fn=hide_cancel_btn,
            outputs=[cancel_btn]
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

