"""
Proposition extraction utility for LLM response feedback and Memory Console.
Splits text into semantically complete, self-contained propositions with filtering.
"""

import re
from typing import List

# Conversational markers to filter out (case-insensitive, prefix match)
CONVERSATIONAL_MARKERS = [
    "based on",
    "here is",
    "here are",
    "according to",
    "in summary",
    "to summarize",
    "this response",
    "the answer",
    "as an ai",
    "i cannot",
    "i don't",
    "i am",
    "i'm",
    "as a language",
    "let me",
    "sure,",
    "of course",
]

# Abbreviations that contain periods and should not split on
ABBREVIATIONS = [
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "mr.",
    "mrs.",
    "dr.",
    "prof.",
    "inc.",
    "ltd.",
    "jr.",
    "sr.",
    "st.",
]

def _protect_abbreviations(text: str) -> tuple[str, dict]:
    """Replace abbreviations with placeholders to avoid false sentence splits."""
    placeholders = {}
    protected = text
    for idx, abbr in enumerate(ABBREVIATIONS):
        placeholder = f"__ABBR_{idx}__"
        # Case-insensitive replacement
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        # Find all occurrences and replace
        def repl(match):
            placeholders[placeholder] = match.group(0)
            return placeholder
        protected = pattern.sub(repl, protected)
    return protected, placeholders

def _restore_abbreviations(text: str, placeholders: dict) -> str:
    """Restore abbreviations from placeholders."""
    restored = text
    for placeholder, original in placeholders.items():
        restored = restored.replace(placeholder, original)
    return restored

def extract_propositions(text: str) -> List[str]:
    """
    Split text into semantically complete propositions with filtering.
    
    - Split on sentence boundaries (?<=[.!?])\\s+
    - Protect abbreviations
    - Filter: length >= 30 chars
    - Filter: exclude sentences starting with conversational markers
    
    Args:
        text: Raw LLM response or user text
        
    Returns:
        List of filtered proposition strings
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Protect abbreviations
    protected, placeholders = _protect_abbreviations(text)
    
    # Split on sentence boundaries
    # Use regex that splits on . ! ? followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    
    propositions = []
    for sent in sentences:
        if not sent:
            continue
        
        # Restore abbreviations
        sent = _restore_abbreviations(sent, placeholders)
        sent = sent.strip()
        
        if not sent:
            continue
        
        # Filter by length
        if len(sent) < 30:
            continue
        
        # Filter by conversational markers (case-insensitive prefix)
        lower = sent.lower()
        # Also strip leading quotes or bullet markers
        stripped_lower = lower.lstrip(' "\'-*•1234567890.').strip()
        is_marker = False
        for marker in CONVERSATIONAL_MARKERS:
            if stripped_lower.startswith(marker):
                is_marker = True
                break
        if is_marker:
            continue
        
        # Ensure sentence ends with punctuation, add period if not
        if sent[-1] not in '.!?':
            sent = sent + '.'
        
        propositions.append(sent)
    
    return propositions

def extract_propositions_with_fallback(text: str, min_length: int = 30) -> List[str]:
    """
    Extract propositions with configurable min length, used for Memory Console fallback.
    """
    if not text:
        return []
    
    # Try to use same logic but with custom min_length
    protected, placeholders = _protect_abbreviations(text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    
    props = []
    for sent in sentences:
        sent = _restore_abbreviations(sent, placeholders).strip()
        if len(sent) >= min_length:
            # Check markers
            lower = sent.lower().lstrip(' "\'-*•1234567890.').strip()
            if not any(lower.startswith(m) for m in CONVERSATIONAL_MARKERS):
                if sent[-1] not in '.!?':
                    sent += '.'
                props.append(sent)
    
    return props
