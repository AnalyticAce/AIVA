from fastapi import APIRouter, Body, HTTPException, status
from decimal import Decimal
from typing import Optional

from aiva.core.logging import logger
from aiva.schemas.agent import (
    PromptRequest, 
    PromptResponse, 
    FinanceData, 
    FinanceActionType,
    AnalyticsResponse
)
from aiva.services.agents import finance_agent

router = APIRouter(prefix="/agent", tags=["Agent"])

@router.post(
    "/process",
    response_model=PromptResponse,
    summary="Process a financial prompt",
    status_code=status.HTTP_200_OK,
)
async def process_financial_prompt(
    request: PromptRequest = Body(..., 
        example={
            "prompt": "I spent $42.50 on groceries yesterday and $100 on Uber the day before",
            "thread_id": None
        })
) -> PromptResponse:
    try:
        logger.info(f"Processing financial prompt: {request.prompt[:50]}...")
        
        # Use the new LangGraph-based process_finance_query function
        result = finance_agent.process_finance_query(
            query=request.prompt,
            thread_id=request.thread_id
        )
        
        # Initialize response
        response = PromptResponse(
            transactions=[],
            error=None,
            analytics=None,
            query_type=str(result.get("query_type", "unknown"))
        )
        
        # Extract messages from the result
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        
        # Process based on query type
        query_type = result.get("query_type")
        
        if query_type == "data_entry":
            # Extract transactions from the result
            if result.get("transactions"):
                finance_data_list = []
                
                for data in result["transactions"]:
                    try:
                        finance_data = FinanceData(
                            action=data.get("action", FinanceActionType.UNKNOWN),
                            amount=Decimal(str(data.get("amount", 0))),
                            category=data.get("category", ""),
                            date=data.get("date", ""),
                            description=data.get("description")
                        )
                        finance_data_list.append(finance_data)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Validation error for transaction data: {str(e)}")

                response.transactions = finance_data_list
                
            # If no transactions were found, look in the last message content
            elif last_message and hasattr(last_message, "content"):
                # Try to extract transactions from the message content
                try:
                    import json
                    # Try to find JSON in the content
                    content = last_message.content
                    start_idx = content.find("{")
                    end_idx = content.rfind("}") + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        trans_data = json.loads(json_str)
                        
                        if "transactions" in trans_data:
                            finance_data_list = []
                            
                            for data in trans_data["transactions"]:
                                try:
                                    finance_data = FinanceData(
                                        action=data.get("action", FinanceActionType.UNKNOWN),
                                        amount=Decimal(str(data.get("amount", 0))),
                                        category=data.get("category", ""),
                                        date=data.get("date", ""),
                                        description=data.get("description")
                                    )
                                    finance_data_list.append(finance_data)
                                except (ValueError, TypeError) as e:
                                    logger.error(f"Validation error for transaction data: {str(e)}")

                            response.transactions = finance_data_list
                except Exception as e:
                    logger.error(f"Error extracting transactions from message: {str(e)}")
                    
        elif query_type in ["analysis", "listing"]:
            # Set the analytics data if it exists
            if result.get("summary"):
                response.analytics = result["summary"]
            
            # If no analytics data but we have a message, use that
            elif last_message and hasattr(last_message, "content"):
                response.analytics = {
                    "content": last_message.content,
                    "format": "markdown"
                }
        
        # If no valid data was found in either case
        if not response.transactions and not response.analytics and not response.error:
            if last_message and hasattr(last_message, "content"):
                # Just return the message content directly
                response.analytics = {
                    "content": last_message.content,
                    "format": "markdown"
                }
            else:
                response.error = "No valid data was extracted from the prompt"
        
        logger.info(f"Successfully processed prompt with query type: {query_type}")
        return response
    
    except Exception as e:
        logger.error(f"Error processing financial prompt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing financial prompt: {str(e)}"
        )