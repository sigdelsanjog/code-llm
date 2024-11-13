# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from config import OPENAI_API_KEY
from utils import generate_pandas_code

openai.api_key = OPENAI_API_KEY

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate-code")
async def generate_code(request: PromptRequest):
    try:
        # Use the util function to process the prompt and generate code
        code = generate_pandas_code(request.prompt)
        return {"generated_code": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating code: {str(e)}")
