"""
Prompt loading utilities for MTG Search Agent.

This module provides functions to load system prompts from external files,
making them easier to maintain and modify without changing code.
"""

import os
from pathlib import Path
from typing import Optional


def _get_prompt_path(filename: str) -> Path:
    """
    Get the absolute path to a prompt file.
    
    Args:
        filename: Name of the prompt file
        
    Returns:
        Path to the prompt file
    """
    current_dir = Path(__file__).parent
    return current_dir / filename


def _load_prompt_file(filename: str) -> str:
    """
    Load a prompt from a text file with error handling.
    
    Args:
        filename: Name of the prompt file to load
        
    Returns:
        The prompt content as a string
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        IOError: If there's an error reading the file
    """
    prompt_path = _get_prompt_path(filename)
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            raise IOError(f"Prompt file is empty: {prompt_path}")
            
        return content
        
    except Exception as e:
        raise IOError(f"Error reading prompt file {prompt_path}: {e}")


def load_query_agent_prompt() -> str:
    """
    Load the system prompt for the Query Agent.
    
    Returns:
        The query agent system prompt as a string
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        IOError: If there's an error reading the file
    """
    return _load_prompt_file('query_agent_prompt.txt')


def load_feedback_synthesis_prompt() -> str:
    """
    Load the system prompt for feedback synthesis.
    
    Returns:
        The feedback synthesis prompt as a string
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        IOError: If there's an error reading the file
    """
    return _load_prompt_file('feedback_synthesis_prompt.txt')


# Export the public functions
__all__ = [
    'load_query_agent_prompt',
    'load_feedback_synthesis_prompt'
]