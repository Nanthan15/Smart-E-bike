from fastapi import FastAPI
from firebase_admin import credentials, db, initialize_app
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import httpx
import json

# Initialize Firebase
cred = credentials.Certificate("firebase_config.json")  # path to your Firebase service account
initialize_app(cred, {
    'databaseURL': 'https://smart-ebike-f4ba1-default-rtdb.firebaseio.com/'  # Replace with your DB URL
})

app = FastAPI()

# CORS setup
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

# 🔧 Ollama model name
LLM_MODEL = "phi"

@app.get("/realtime-data")
async def get_data():
    ref = db.reference("/sensors")
    data = ref.get()

    if not data:
        return {"error": "No sensor data found."}

    prompt = (
        f"You are analyzing data from a smart e-bike. "
        f"Sensor readings are:\n"
        f"Speed: {data.get('speed')} km/h\n"
        f"Battery: {data.get('battery')}%\n"
        f"Terrain: {data.get('terrain')}\n"
        f"Temperature: {data.get('temp')}°C\n"
        f"Give helpful real-time feedback to the rider."
    )

    feedback = "Generating feedback..."
    try:
        timeout = httpx.Timeout(90.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(3):  # Retry 3 times
                try:
                    response = await client.post(
                        "http://localhost:11434/api/generate",
                        json={"model": LLM_MODEL, "prompt": prompt, "stream": True}
                    )

                    # Stream handling
                    feedback_chunks = []
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk_data = json.loads(line)
                            chunk = chunk_data.get("response")
                            if chunk:
                                feedback_chunks.append(chunk)
                        except Exception as parse_err:
                            print(f"⚠️ Failed to parse chunk: {parse_err} - Raw line: {line}")

                    feedback = ''.join(feedback_chunks) if feedback_chunks else feedback
                    break  # Exit retry loop if successful
                except httpx.ReadTimeout:
                    print(f"⚠️ Attempt {attempt+1}: Ollama is still loading...")
                    await asyncio.sleep(5)
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        feedback = "Ollama is not responding. Please try again later."

    return {
        "speed": data.get("speed"),
        "battery": data.get("battery"),
        "terrain": data.get("terrain"),
        "temp": data.get("temp"),
        "feedback": feedback.strip()
    }
