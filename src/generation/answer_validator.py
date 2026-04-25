"""
Answer Format Validation Layer

Validates that generated answers match expected output format:
- Count questions: ensure output is an integer
- Proportion questions: ensure output is percentage with units
- List questions: ensure output is properly formatted list
- Text questions: basic non-empty validation

Note: The motivation behind the following validation logic is that during development, 
we observed that the LLM sometimes outputs answers in unexpected formats (e.g., "2 students" instead of "2", or 
"27 percent" instead of "27%"). This validation layer is designed to catch common formatting issues 
and correct them when possible, while also providing feedback on any validation failures.

From a metrics evaluation standpoint, particularly for generation metrics like Exact Match or F1, having a consistent answer format is crucial.
For example, if the ground truth answer is "27%" but the model outputs "27 percent", a strict Exact Match would fail even though the answer is semantically correct. 
By validating and correcting the format to a standard representation (e.g., always using "27%"), 
we can ensure that our evaluation metrics more accurately reflect the model's performance rather 
than being skewed by formatting inconsistencies. 

In other words, doing this validation and correction step allows us to better assess the true quality 
of the generated answers, rather than penalizing the model for minor formatting variations that do not 
affect the overall correctness of the answer.

When questioned whether this is considered fair, it's important to note that the goal of this validation layer is 
not to give the model an unfair advantage, but rather to ensure that we are evaluating the model's understanding 
and reasoning capabilities rather than its ability to guess the exact formatting of the answer.
"""

import logging
import re
import json
from typing import Any, Tuple
import ast

logger = logging.getLogger(__name__)


class AnswerValidator:
    """Validates generated answers against expected output formats."""
    
    @staticmethod
    def validate(answer: str, question: str = "", ground_truth: Any = None) -> Tuple[bool, str, str]:
        """
        Validate answer format and structure.
        
        Args:
            answer: The generated answer string
            question: The original question (optional, for context)
            ground_truth: Expected answer type/format (optional, for validation)
            
        Returns:
            Tuple of (is_valid, corrected_answer, validation_message)
        """
        if not answer or not isinstance(answer, str):
            return False, "", "Answer is empty or not a string"
        
        answer = answer.strip()
        
        # Detect question type from the ground truth or question text
        question_lower = question.lower() if question else ""
        
        # Scenario: EXPLICIT LIST QUESTIONS
        # Keywords: "list", "enumerate", "which", "names", "items", "examples"
        is_explicit_list = (
            "list" in question_lower or 
            "enumerate" in question_lower or
            "which" in question_lower or
            "names of" in question_lower or
            "examples" in question_lower or
            "items" in question_lower or
            "identify" in question_lower or
            "tell me" in question_lower
        )
        
        # Also check if answer contains multiple items (commas, semicolons, bracketed)
        has_multiple_items = (
            ',' in answer or ';' in answer or answer.startswith('[')
        )
        
        # If question explicitly asks for a list OR answer contains multiple items, validate as list
        if is_explicit_list or (has_multiple_items and "[" in answer):
            is_valid, corrected = AnswerValidator._validate_list(answer)
            if is_valid:
                return True, corrected, "Valid list format"
            else:
                return False, answer, f"Invalid list format. Expected ['item1', 'item2'], got: {answer}"
        
        # Scenario: COUNT QUESTIONS
        # Keywords: "how many", "how much", "count", "number of"
        if ("how many" in question_lower or 
            "how much" in question_lower or 
            "count" in question_lower or
            "number of" in question_lower):
            
            is_valid, corrected = AnswerValidator._validate_count(answer)
            if is_valid:
                return True, corrected, "Valid count format"
            else:
                return False, answer, f"Invalid count format. Expected integer, got: {answer}"
        
        # Scenario: PROPORTION/PERCENTAGE QUESTIONS
        # Keywords: "percentage", "percent", "%", "how many %", "what percentage"
        # Also, check if the question asks for MULTIPLE percentages (e.g., "how many % ... and how many %")
        num_how_many_percent = question_lower.count("how many %")
        num_what_percent = question_lower.count("what %")
        asks_multiple_percentages = (
            num_how_many_percent > 1 or 
            num_what_percent > 1 or
            (num_how_many_percent > 0 and " and " in question_lower and ("how many %" in question_lower or "what %" in question_lower))
        )
        
        # Also check if the answer contains multiple values (commas, "and", bracketed)
        has_multiple_percentages_in_answer = (
            (',' in answer and '%' in answer) or 
            (' and ' in answer and '%' in answer) or
            answer.startswith('[')
        )
        
        
        if ("percentage" in question_lower or 
            "percent" in question_lower or 
            "what %" in question_lower or
            "how many %" in question_lower):
            
            # If question asks for multiple percentages OR answer contains multiple values --> treat as list
            if asks_multiple_percentages or has_multiple_percentages_in_answer:
                # For multi-percentage answers with prose text, try to extract all percentages
                if not answer.startswith('[') and '%' in answer:
                    # Try to extract multiple percentages from prose: look for "number%" or "number percent"
                    percentages = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', answer, re.IGNORECASE)
                    if percentages and len(percentages) > 1:  # Found multiple percentages in prose, format as list
                        # Format as JSON list with % symbols
                        formatted = json.dumps([f"{p}%" for p in percentages])
                        return True, formatted, "Valid list format (extracted percentages from prose)"
                    elif percentages: # Only one percentage found, despite multiple being asked for
                        # Still process it as list in case LLM provided alternate format
                        formatted = json.dumps([f"{p}%" for p in percentages])
                        return True, formatted, "Valid list format (single percentage found, expected multiple)"
                
                # Try standard list validation for already-structured answers
                is_valid, corrected = AnswerValidator._validate_list(answer)
                if is_valid:
                    return True, corrected, "Valid list format (percentage list)"
                else:
                    return False, answer, f"Invalid list format. Expected ['item1', 'item2'], got: {answer}"
            
            # If single percentage question
            is_valid, corrected = AnswerValidator._validate_percentage(answer)
            if is_valid:
                return True, corrected, "Valid percentage format"
            else:
                return False, answer, f"Invalid percentage format. Expected X% or X percent, got: {answer}"
        
        # Scenario: YES/NO QUESTIONS (ONLY IF EXPLICIT)
        # Must have BOTH yes/no, or be clear inverted question
        # But NOT questions with "Figure", "Table", "what", or other list-like patterns
        is_likely_yes_no = (
            ("yes" in question_lower and "no" in question_lower) or
            ("is it" in question_lower) or
            ("does it" in question_lower) or
            ("are there" in question_lower and "yes" not in question_lower or "no" not in question_lower)
        )
        
        # Exclude if question contains list/count/percentage keywords or asks about visual content
        is_visual_question = (
            "figure" in question_lower or
            "table" in question_lower or
            "image" in question_lower or
            "diagram" in question_lower or
            "graph" in question_lower
        )
        
        # If it looks like a yes/no question and doesn't have list/count/percentage keywords or visual content, validate as yes/no
        if is_likely_yes_no and not is_visual_question and not has_multiple_items:
            is_valid, corrected = AnswerValidator._validate_yes_no(answer)
            if is_valid:
                return True, corrected, "Valid yes/no format"
            else:
                return False, answer, f"Invalid yes/no format. Expected 'Yes' or 'No', got: {answer}"
        
        # Scenario: DEFAULT --> TEXT ANSWER 
        # For other questions, just ensure non-empty
        if len(answer) > 0:
            return True, answer, "Valid text answer"
        else:
            return False, answer, "Empty answer"
    
    # ---------------------------------------------------------------------------
    # Helper validation functions for specific formats
    
    # Helper function for validating "counting-like" answers (should be an integer, but may be in various formats)
    @staticmethod
    def _validate_count(answer: str) -> Tuple[bool, str]:
        """
        Validate that the answer is a valid integer.
        
        Returns:
            (is_valid, corrected_answer)
        """
        answer = answer.strip()
        
        # Try to extract integer from various formats
        # "2", "two", "approximately 2", "around 2", "2 items", etc.
        
        # Direct integer parse
        try:
            num = int(answer)
            return True, str(num)
        except ValueError:
            pass
        
        # Extract number from text like "approximately 2" or "2 items"
        match = re.search(r'\b(\d+)\b', answer)
        if match:
            num = int(match.group(1))
            return True, str(num)
        
        # Handle word numbers (one, two, three, etc.)
        word_nums = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
            'hundred': 100, 'thousand': 1000
        }
        
        answer_lower = answer.lower().strip()
        for word, num in word_nums.items(): # Check if any "number word" is present in the answer
            if word in answer_lower: # If we find a number word, we can attempt to parse it (this is a simple approach and may not handle complex cases like "twenty one")
                return True, str(num)
        
        return False, answer # If we can't extract a valid integer, return False with original answer for logging
    
    # Helper function for validating "percentage-like" answers (should be a percentage, but may be in various formats)
    @staticmethod
    def _validate_percentage(answer: str) -> Tuple[bool, str]:
        """
        Validate that the answer is a valid percentage format.
        
        Expected formats: "27%", "27 percent", "27.5%", etc.
        
        Returns:
            (is_valid, corrected_answer)
        """
        answer = answer.strip()
        
        # Try direct percentage parse (with %)
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', answer)
        if match:
            num = float(match.group(1)) # Allow for decimal percentages like "27.5%"
            # Return with % symbol
            if num == int(num):
                return True, f"{int(num)}%"
            else: # Keep decimal if present
                return True, f"{num}%"
        
        # Try "percent" format (e.g., "27 percent")
        match = re.search(r'(\d+(?:\.\d+)?)\s*percent', answer, re.IGNORECASE)
        if match:
            num = float(match.group(1)) # Allow for decimal percentages like "27.5 percent"
            if num == int(num):
                return True, f"{int(num)}%"
            else:
                return True, f"{num}%"
        
        # Try just number (assume percentage)
        match = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', answer)
        if match:
            num = float(match.group(1))
            if num == int(num):
                return True, f"{int(num)}%"
            else:
                return True, f"{num}%"
        
        return False, answer
    
   # Helper function for validating "list-like" answers (should be a list, but may be in various formats)
    @staticmethod
    def _validate_list(answer: str) -> Tuple[bool, str]:
        """
        Validate that the answer is a properly formatted list.
        
        Expected formats:
        - ["item1", "item2"]
        - ['item1', 'item2']
        - [item1, item2]
        - item1, item2
        - item1; item2
        - item1 and item2
        
        Returns:
            (is_valid, corrected_answer)
        """
        answer = answer.strip()
        
        # Already a JSON list (with single or double quotes)
        if answer.startswith('[') and answer.endswith(']'):
            try:
                # Try parsing as-is first (double quotes (""))
                parsed = json.loads(answer)
            except json.JSONDecodeError:
                try:
                    # If double quotes fail, try converting single quotes ('') to double quotes (e.g. handles LLM outputs like ['item1', 'item2'])
                    converted = answer.replace("'", '"')
                    parsed = json.loads(converted)
                except json.JSONDecodeError:
                    # If both fail, treat as literal Python list
                    try:                        
                        parsed = ast.literal_eval(answer)
                    except (ValueError, SyntaxError):
                        parsed = None
            
            # If we successfully parsed a list, ensure all items are strings and preserve all items (even if some are empty after stripping)
            if parsed is not None and isinstance(parsed, list):
                # Ensure all items are strings and preserve all items
                items = [str(item).strip() for item in parsed]
                items = [item for item in items if item]  # Remove empty items
                if items:
                    return True, json.dumps(items)
        
        # If comma-separated list: "item1, item2, item3"
        if ',' in answer and not answer.startswith('['):
            items = [item.strip().strip('\'"') for item in answer.split(',')]
            items = [item for item in items if item]  
            if items:
                return True, json.dumps(items)
        
        # If semicolon-separated list: "item1; item2; item3"
        if ';' in answer and not answer.startswith('['):
            items = [item.strip().strip('\'"') for item in answer.split(';')]
            items = [item for item in items if item]  
            if items:
                return True, json.dumps(items)
        
        # If "and"-separated list: "item1 and item2"
        if ' and ' in answer and not answer.startswith('['):
            items = [item.strip().strip('\'"') for item in answer.split(' and ')]
            items = [item for item in items if item]  
            if items:
                return True, json.dumps(items)
        
        # Single item that looks like a list (only if no delimiters detected)
        if ',' not in answer and ';' not in answer and ' and ' not in answer:
            items = [answer.strip().strip('\'"')]
            if items and items[0]:
                return True, json.dumps(items)
        
        return False, answer
    
    # Helper function for validating "yes/no-like" answers (should be "Yes" or "No", but may be in various formats)
    @staticmethod
    def _validate_yes_no(answer: str) -> Tuple[bool, str]:
        """
        Validate that the answer is "Yes" or "No".
        
        Handles variations: yes, no, Yes, No, YES, NO, Yes., No.
        
        Returns:
            (is_valid, corrected_answer)
        """
        answer = answer.strip()
        
        # Remove trailing punctuation
        answer_clean = answer.rstrip('.!?').strip().lower()
        
        # Check for yes/no variations
        if answer_clean in ['yes', 'y', 'true']:
            return True, "Yes"
        elif answer_clean in ['no', 'n', 'false']:
            return True, "No"
        
        return False, answer



# Main validation function to be called externally
def validate_answer_format(
    answer: str,
    question: str = "",
    ground_truth: Any = None,
    log_issues: bool = True
) -> Tuple[bool, str]:
    """
    Validate and correct answer format.
    
    Args:
        answer: The generated/predicted answer string
        question: The original question (for context)
        ground_truth: Expected answer format (optional)
        log_issues: Log validation issues to logger
        
    Returns:
        Tuple of (is_valid, corrected_answer)
    """
    validator = AnswerValidator() # Instantiate the validator 
    is_valid, corrected, message = validator.validate(answer, question, ground_truth) # Validate the answer format, and get the corrected answer and validation message
    
    if not is_valid and log_issues:
        logger.warning(f"Answer format validation failed: {message}")
        logger.debug(f"Original answer: {answer}")
        logger.debug(f"Suggestion: {corrected}")
    
    return is_valid, corrected if is_valid else answer
