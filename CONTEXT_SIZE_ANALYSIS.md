# Context Size Analysis for MiniNewsAI Models

## Model Context Sizes

### 1. Classifier (RoBERTa-base)

- **Model max_position_embeddings**: 514 tokens
- **Training max_length**: 512 tokens (with truncation=True)
- **Current UI**: `max_length=512` with `truncation=True` ✓ **CORRECT**

**Status**: Already handling correctly. Articles are truncated to 512 tokens during tokenization.

### 2. Rewriter (Mistral-7B-Instruct-v0.2)

- **Model max_position_embeddings**: 32,768 tokens (very large!)
- **Training max_length**: 512 tokens for tokenization (with truncation=True)
- **Training data**: Articles were preprocessed to ~500 words (`article_500` column)
- **Current UI**: `max_length=512` with `truncation=True` ⚠️ **POTENTIAL ISSUE**

## Current Issue with Rewriter

The rewriter currently truncates the **full prompt** (instruction + article) to 512 tokens:

```python
prompt = f"[INST] {instruction} [/INST]\n\n"  # instruction includes article
inputs = rewriter_tokenizer(prompt, max_length=512, truncation=True)
```

**Problem**: The instruction prompt itself takes ~50-100 tokens, so the article only gets ~400-450 tokens, which may truncate important content.

## Recommendations

### Option 1: Truncate Article Before Creating Prompt (RECOMMENDED)

Truncate the article text to fit within the token budget, leaving room for the instruction:

```python
# Estimate instruction tokens (~80 tokens)
INSTRUCTION_TOKEN_BUDGET = 80
ARTICLE_MAX_TOKENS = 512 - INSTRUCTION_TOKEN_BUDGET  # ~432 tokens

# Truncate article before creating prompt
article_tokens = rewriter_tokenizer.encode(article_text, add_special_tokens=False)
if len(article_tokens) > ARTICLE_MAX_TOKENS:
    article_tokens = article_tokens[:ARTICLE_MAX_TOKENS]
    article_text = rewriter_tokenizer.decode(article_tokens, skip_special_tokens=True)

# Then create prompt with truncated article
instruction = create_instruction_prompt(label, article_text, title)
prompt = f"[INST] {instruction} [/INST]\n\n"
```

### Option 2: Increase max_length for Rewriter

Since Mistral supports 32K tokens, we could increase to 1024 or 2048:

- **Pros**: More article content preserved
- **Cons**: Slower inference, may not match training distribution

### Option 3: Keep Current Approach

- **Pros**: Matches training (512 tokens total)
- **Cons**: Articles get truncated more than necessary

## Recommendation

**Use Option 1**: Truncate the article to ~430 tokens before creating the prompt. This:

1. Maximizes article content within the 512-token limit
2. Matches training approach (articles were pre-truncated to ~500 words)
3. Ensures consistent behavior

## Implementation Notes

During training:

- Articles were already truncated to ~500 words (`article_500` column)
- The full prompt (instruction + article) was tokenized with `max_length=512`
- This means articles were effectively ~400-450 tokens

For inference:

- We should pre-truncate articles to ~430 tokens
- Then create the prompt
- This ensures we use the full token budget efficiently
