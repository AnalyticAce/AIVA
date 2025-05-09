"""
Base tool interface for AIVA finance agent tools.

Note: This is a placeholder. In the simplified implementation, tools are integrated
directly into the LangGraph workflow rather than as separate classes.
"""
from typing import Any, Dict

class BaseTool:
    """Placeholder base class for backward compatibility."""
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Placeholder execute method.
        
        Args:
            params: Tool parameters
            
        Returns:
            Dict: Empty result
        """
        return {}