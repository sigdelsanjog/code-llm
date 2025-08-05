# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from utils import generate_pandas_code


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

class ModelResponse(BaseModel):
    model: str
    response: str

class GeneratedCodeResponse(BaseModel):
    prompt: str
    model_responses: List[ModelResponse]
    formatted_response: str

@app.post("/generate-code", response_model=GeneratedCodeResponse)
async def generate_code(request: PromptRequest):
    try:
        # Use the util function to process the prompt and generate code from all models
        results = await generate_pandas_code(request.prompt)
        
        # Format the responses into paragraphs
        formatted_paragraphs = []
        model_responses = []
        
        for result in results["responses"]:
            model_name = result["model"]
            response = result["response"]
            
            # Create paragraph for each model
            paragraph = f"**{model_name} Response:**\n{response}\n"
            formatted_paragraphs.append(paragraph)
            
            # Add to model responses
            model_responses.append(ModelResponse(model=model_name, response=response))
        
        # Join all paragraphs
        formatted_response = "\n".join(formatted_paragraphs)
        
        return GeneratedCodeResponse(
            prompt=request.prompt,
            model_responses=model_responses,
            formatted_response=formatted_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating code: {str(e)}")
