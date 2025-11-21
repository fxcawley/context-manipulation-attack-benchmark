"""
Attack implementation modules for context manipulation experiments.
"""

from .conversation_poisoning import (
    FalseConversationInjection,
    IterativeContextPoisoning,
    GaslightingAttack
)

__all__ = [
    "FalseConversationInjection",
    "IterativeContextPoisoning", 
    "GaslightingAttack"
]

