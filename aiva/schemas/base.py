"""
Base schema definitions for the AIVA application.
"""
from datetime import datetime
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class AIVABaseModel(BaseModel):
    """Base Pydantic model with common configuration."""
    
    class Config:
        """Pydantic model configuration."""
        populate_by_name = True
        validate_assignment = True
        json_schema_extra = {"example": {}}