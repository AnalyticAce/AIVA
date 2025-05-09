from fastapi import APIRouter, Body, HTTPException, status
from decimal import Decimal

from aiva.core.logging import logger
from aiva.schemas.agent import (
    PromptRequest, 
    PromptResponse, 
    FinanceData, 
    FinanceActionType
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
            "prompt": "I spent $42.50 on groceries yesterday and $100 on Uber the day before"
        })
) -> PromptResponse:
    try:
        logger.info(f"Processing financial prompt: {request.prompt[:50]}...")
        
        result = await finance_agent.process_prompt(prompt=request.prompt)
        
        response = PromptResponse(
            transactions=[],
            error=result.get("error")
        )
        
        if result.get("transactions"):
            transactions = result["transactions"]
            finance_data_list = []
            
            for data in transactions:
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
        
        if not response.transactions and not response.error:
            response.error = "No valid transactions were extracted from the prompt"
        
        logger.info(f"Successfully processed prompt with {len(response.transactions)} transactions")
        return response
    
    except Exception as e:
        logger.error(f"Error processing financial prompt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing financial prompt: {str(e)}"
        )