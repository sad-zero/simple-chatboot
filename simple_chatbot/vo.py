"""
Chatbot VO
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DocumentPair:
    document: str
    embedding: List[float]
