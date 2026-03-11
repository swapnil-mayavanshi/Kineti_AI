"""
Kineti-AI Pydantic Models
Request/response models for API endpoints.
"""

from pydantic import BaseModel


class TextRequest(BaseModel):
    """Request model for the /speak endpoint."""
    text: str
