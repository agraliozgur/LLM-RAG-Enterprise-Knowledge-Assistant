#!/usr/bin/env python3
# FILE: alignment.py
# --------------------------------------------------
# Placeholder for advanced LLM alignment (policy filtering, RLHF, etc.)
# --------------------------------------------------

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example: Basic policy checks
RESTRICTED_KEYWORDS = ["secret_code", "internal_server", "classified"]

def violates_policy(answer_text: str) -> bool:
    """
    Check if the provided text violates any simple policy rules.
    
    Returns:
        True if policy is violated, False otherwise.
    """
    lower_text = answer_text.lower()
    for keyword in RESTRICTED_KEYWORDS:
        if keyword in lower_text:
            return True
    return False

def filter_answer(answer_text: str) -> str:
    """
    Either modifies or blocks the answer if it violates policy.
    Otherwise, returns the original answer.
    """
    if violates_policy(answer_text):
        logger.warning("Answer violates policy. Providing a restricted response.")
        return "This answer is withheld due to policy restrictions."
    else:
        return answer_text

def role_based_adjustment(answer_text: str, user_role: str) -> str:
    """
    Optionally adapt the answer based on the user_role.
    For example, managers see more details than junior staff.
    
    Args:
        answer_text (str): The model-generated answer.
        user_role (str): A user role (e.g., "manager", "engineer", "intern").
    
    Returns:
        str: Possibly modified answer text.
    """
    # Simple example logic
    if user_role.lower() in ["manager", "admin"]:
        return answer_text  # no change
    else:
        # You might remove or summarize certain sections
        return answer_text  # This is a placeholder

def align_answer(answer_text: str, user_role: str = "employee") -> str:
    """
    Master function to apply policy filtering and role-based adjustments.
    """
    filtered = filter_answer(answer_text)
    adjusted = role_based_adjustment(filtered, user_role)
    return adjusted
