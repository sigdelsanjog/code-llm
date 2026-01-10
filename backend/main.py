# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import concurrent.futures

# Import services (Dependency Inversion Principle - depend on abstractions)
from services import ServiceFactory, BaseModelService
from config import get_available_models


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; for production, specify only necessary origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

class PromptRequest(BaseModel):
    prompt: str
    model_name: Optional[str] = None  # Optional: specific model to use

class ModelResponse(BaseModel):
    model: str
    response: str
    reference_answer: Optional[str] = None  # Ground truth from training data

class GeneratedCodeResponse(BaseModel):
    prompt: str
    model_responses: List[ModelResponse]
    formatted_response: str

@app.get("/models")
async def get_models():
    """
    Get list of available models.
    Returns model metadata including id, display name, and description.
    """
    return {"models": get_available_models()}

@app.post("/generate-code", response_model=GeneratedCodeResponse)
async def generate_code(request: PromptRequest):
    """
    Generate code using model services with dependency injection.
    
    If model_name is specified, uses that specific service.
    Otherwise, uses all available services.
    """
    try:
        # Dependency Injection: Get appropriate service(s) based on request
        if request.model_name:
            # Inject single service for specific model
            service = ServiceFactory.get_service(request.model_name)
            result = service.generate(request.prompt)
            model_responses = [ModelResponse(
                model=result["model"], 
                response=result["response"],
                reference_answer=result.get("reference_answer")
            )]
            formatted_response = f"**{result['model']} Response:**\n{result['response']}\n"
        else:
            # Inject all services for "All Models" option
            services = ServiceFactory.get_all_services()
            
            # Execute all services in parallel for better performance
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(services)) as executor:
                # Submit all service generation tasks
                future_to_service = {
                    executor.submit(service.generate, request.prompt): service
                    for service in services
                }
                
                # Collect results as they complete
                model_responses = []
                formatted_paragraphs = []
                
                for future in concurrent.futures.as_completed(future_to_service):
                    try:
                        result = future.result()
                        model_responses.append(
                            ModelResponse(
                                model=result["model"], 
                                response=result["response"],
                                reference_answer=result.get("reference_answer")
                            )
                        )
                        formatted_paragraphs.append(
                            f"**{result['model']} Response:**\n{result['response']}\n"
                        )
                    except Exception as e:
                        service = future_to_service[future]
                        error_result = {
                            "model": service.get_model_name(),
                            "response": f"Error: {str(e)}"
                        }
                        model_responses.append(
                            ModelResponse(
                                model=error_result["model"], 
                                response=error_result["response"],
                                reference_answer=None
                            )
                        )
                        formatted_paragraphs.append(
                            f"**{error_result['model']} Response:**\n{error_result['response']}\n"
                        )
            
            formatted_response = "\n".join(formatted_paragraphs)
        
        return GeneratedCodeResponse(
            prompt=request.prompt,
            model_responses=model_responses,
            formatted_response=formatted_response
        )
    except ValueError as e:
        # Handle invalid model name
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating code: {str(e)}")
