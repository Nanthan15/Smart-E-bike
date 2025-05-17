from fastapi import FastAPI, HTTPException
from firebase_admin import credentials, db, initialize_app
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import JSONResponse
import asyncio

from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Initialize Firebase
cred = credentials.Certificate("firebase_config.json")
initialize_app(cred, {
    'databaseURL': 'https://smart-ebike-f4ba1-default-rtdb.firebaseio.com/'
})

app = FastAPI()

# CORS
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama model setup
LLM_MODEL = "phi"
llm = OllamaLLM(model=LLM_MODEL, base_url="http://localhost:11434")

prompt_template = PromptTemplate(
    input_variables=["speed", "battery", "terrain", "temp"],
    template=(
        "You are analyzing data from a smart e-bike.\n"
        "Sensor readings are:\n"
        "Speed: {speed} km/h\n"
        "Battery: {battery}%\n"
        "Terrain: {terrain}\n"
        "Temperature: {temp}°C\n"
        "Give helpful real-time feedback to the rider."
    )
)
chain = prompt_template | llm
executor = ThreadPoolExecutor()

@app.get("/sensor-data")
async def get_sensor_data():
    loop = asyncio.get_event_loop()
    ref = db.reference("/sensors")
    data = await loop.run_in_executor(executor, ref.get)

    if not data:
        raise HTTPException(status_code=404, detail="No sensor data found.")
    
    response_data = {
        "speed": data.get("speed"),
        "battery": data.get("battery"),
        "terrain": data.get("terrain"),
        "temp": data.get("temp")
    }

    return JSONResponse(content=response_data, headers={"Cache-Control": "no-cache"})

@app.get("/feedback")
async def get_feedback():
    ref = db.reference("/sensors")
    data = ref.get()

    if not data:
        raise HTTPException(status_code=404, detail="No sensor data found.")

    inputs = {
        "speed": data.get("speed", "unknown"),
        "battery": data.get("battery", "unknown"),
        "terrain": data.get("terrain", "unknown"),
        "temp": data.get("temp", "unknown"),
    }

    try:
        result = chain.invoke(inputs)
        feedback = result
    except Exception as e:
        print(f"❌ LangChain/Ollama error: {e}")
        feedback = "Unable to generate feedback at the moment. Please try again later."

    return {"feedback": feedback.strip() if isinstance(feedback, str) else str(feedback)}
