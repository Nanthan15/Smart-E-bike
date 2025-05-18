from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, initialize_app, db
import asyncio
from collections import deque, Counter
import copy

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains import LLMChain

# Firebase setup
cred = credentials.Certificate("firebase_config.json")
firebase_app = initialize_app(cred, {
    'databaseURL': 'https://smart-ebike-f4ba1-default-rtdb.firebaseio.com/'
})
ref = db.reference('sensors')

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

sensor_deque = deque(maxlen=5)
last_sent_data = None
last_feedback = None

# LangChain LLM setup
LLM_MODEL = "phi"
llm = Ollama(model=LLM_MODEL, base_url="http://localhost:11434")

prompt_template = PromptTemplate(
    input_variables=["speed", "battery", "terrain", "temp"],
    template=(
        "You are analyzing data from a smart e-bike.\n"
        "Latest 5 sensor readings aggregated:\n"
        "Average Speed: {speed} km/h\n"
        "Average Battery: {battery}%\n"
        "Most Common Terrain: {terrain}\n"
        "Average Temperature: {temp}°C\n"
        "Give helpful real-time feedback to the rider based on these aggregated values and don't give long result keep results short and informative."
    )
)

chain = LLMChain(llm=llm, prompt=prompt_template)

def aggregate_sensor_data(sensor_deque):
    n = len(sensor_deque)
    if n == 0:
        return None

    avg_speed = sum(entry.get("speed", 0) for entry in sensor_deque) / n
    avg_battery = sum(entry.get("battery", 0) for entry in sensor_deque) / n
    avg_temp = sum(entry.get("temp", 0) for entry in sensor_deque) / n

    terrains = [entry.get("terrain", "unknown") for entry in sensor_deque]
    most_common_terrain = Counter(terrains).most_common(1)[0][0] if terrains else "unknown"

    return {
        "speed": f"{avg_speed:.2f}",
        "battery": f"{avg_battery:.2f}",
        "terrain": most_common_terrain,
        "temp": f"{avg_temp:.2f}"
    }

async def send_to_llm(sensor_deque):
    global last_feedback

    aggregated_data = aggregate_sensor_data(sensor_deque)
    if aggregated_data is None:
        print("⚠️ No data to send to LLM")
        return None

    # LangChain LLMChain is sync, run in thread to not block event loop
    import asyncio

    def run_chain():
        return chain.invoke(aggregated_data)

    try:
        feedback_obj = await asyncio.to_thread(run_chain)
        print("✅ Raw feedback object from LLM chain:", feedback_obj)

        # Extract text safely
        if isinstance(feedback_obj, dict) and "text" in feedback_obj:
            feedback_text = feedback_obj["text"]
        elif hasattr(feedback_obj, "text"):
            feedback_text = feedback_obj.text
        else:
            feedback_text = str(feedback_obj)

        print("✅ Extracted feedback text:", feedback_text)
        last_feedback = feedback_text
        return feedback_text
    except Exception as e:
        print("❌ Error running LLM chain:", e)
        last_feedback = "Error communicating with LLM."
        return None



async def watch_firebase():
    global last_sent_data
    while True:
        data = ref.get()
        if data:
            current_data = {
                "battery": data.get("battery"),
                "speed": data.get("speed"),
                "temp": data.get("temp"),
                "terrain": data.get("terrain")
            }

            if current_data != last_sent_data:
                last_sent_data = copy.deepcopy(current_data)
                sensor_deque.append(current_data)
                print("📥 New sensor data detected.")
                print("📊 Current deque ({} entries):".format(len(sensor_deque)))
                for item in sensor_deque:
                    print(item)

                # Send aggregated data to LLM for feedback
                await send_to_llm(sensor_deque)
            else:
                print("⏩ Skipping duplicate sensor values.")
        else:
            print("⚠️ No sensor data found.")
        await asyncio.sleep(2)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(watch_firebase())

@app.get("/")
def root():
    return {"status": "Smart E-bike LLM backend is running"}

@app.get("/feedback")
def get_feedback():
    if last_feedback is None:
        return {"feedback": "No feedback available yet"}
    return {"feedback": last_feedback}
