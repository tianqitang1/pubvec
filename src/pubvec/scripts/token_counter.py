#!/usr/bin/env python3
"""
Token Counter Script

This script prompts for user input and estimates the number of tokens in the input text.
It uses the tiktoken library which provides tokenizers used by OpenAI models.

Usage:
    python token_counter.py

Requirements:
    pip install tiktoken
"""

import sys
import tiktoken

def count_tokens(text, model="cl100k_base"):
    """
    Count the number of tokens in the given text using the specified encoding.
    
    Args:
        text (str): The text to tokenize
        model (str): The encoding model to use (default: cl100k_base which is used by GPT-4)
    
    Returns:
        int: The number of tokens in the text
    """
    try:
        encoding = tiktoken.get_encoding(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        print("Make sure you have tiktoken installed: pip install tiktoken")
        sys.exit(1)

def main():
    # Print welcome message
    print("=" * 60)
    print("Token Counter - Estimate the number of tokens in your text")
    print("=" * 60)
    print("Enter or paste your text below. Press Ctrl+D (Unix) or Ctrl+Z (Windows) followed by Enter to finish.")
    print()
    
    # Collect input from user
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    except KeyboardInterrupt:
        print("\nInput cancelled.")
        return
    
    # Join the lines with newlines to preserve formatting
    text = "\n".join(lines)
    
    # Skip processing if no text was entered
    if not text.strip():
        print("\nNo text entered. Exiting.")
        return
    
    # Count tokens
    token_count = count_tokens(text)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Text length: {len(text)} characters")
    print(f"Token count: {token_count} tokens")
    print("=" * 60)
    
    # Add some context about token pricing
    print("\nFor reference:")
    print("- GPT-3.5 Turbo: ~4K tokens per $0.01")
    print("- GPT-4 Turbo: ~1K tokens per $0.01")
    print("- Claude 3 Opus: ~1K tokens per $0.015")

if __name__ == "__main__":
    main() 