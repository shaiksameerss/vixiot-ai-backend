import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI


# ==========================================
# CONFIGURATION
# ==========================================

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="VIXIOT AI Assistant",
    description="AI chatbot for VIXIOT smart home products"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "https://vixiot.com",
        "https://www.vixiot.com"
    ],

    allow_methods=[
        "GET",
        "POST"
    ],

    allow_headers=[
        "Content-Type"
    ]
)


# ==========================================
# REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=2000
    )


# ==========================================
# AI INSTRUCTIONS
# ==========================================

SYSTEM_PROMPT = """

You are the official AI assistant for VIXIOT.

VIXIOT is a smart home automation brand.

VIXIOT website products include:

1. One-channel Wi-Fi smart switch
2. Two-channel Wi-Fi smart switch
3. Four-channel Wi-Fi smart switch
4. Smart Wi-Fi fan controller

VIXIOT highlights:

- Remote control
- Smart home automation
- Alexa and Google Assistant support
- Products designed for Indian homes

YOUR RESPONSIBILITIES:

1. Help customers understand smart home automation.
2. Explain the general purpose of VIXIOT products.
3. Help customers choose between product categories.
4. Answer general product questions.
5. Explain general Wi-Fi troubleshooting steps.
6. Guide customers to contact VIXIOT for enquiries.
7. Keep answers concise and friendly.

IMPORTANT RULES:

- Never invent product prices.
- Never invent product specifications.
- Never invent electrical ratings.
- Never invent compatibility information.
- Never invent warranty information.
- If information is not provided,
  tell the customer to contact VIXIOT.
- Do not give dangerous electrical wiring instructions.
- Recommend a qualified electrician for electrical installation.
- Do not claim a product supports a feature
  unless that feature is confirmed.
- Answer in simple language.

You are a customer support and product information assistant.

"""


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
def home():

    return {
        "status": "VIXIOT AI Assistant is running"
    }


# ==========================================
# CHAT ENDPOINT
# ==========================================

@app.post("/chat")
def chat(request: ChatRequest):

    question = request.message.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )


    try:

        response = client.responses.create(

            model=OPENAI_MODEL,

            instructions=SYSTEM_PROMPT,

            input=question

        )


        answer = response.output_text


        return {

            "answer": answer

        }


    except Exception as error:

        print("AI request error:", error)

        raise HTTPException(

            status_code=500,

            detail="AI service temporarily unavailable"

        )
