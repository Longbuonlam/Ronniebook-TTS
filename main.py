from gradio_client import Client, handle_file
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import requests
import sseclient
import uuid # For generating unique session hashes
import io  # For handling in-memory file operations
from pydantic import BaseModel
from typing import Optional


class UserRecord(BaseModel):
    path: str
    recordUrl: str
    originalName: str
    size: Optional[int] = None

class AudioRequest(BaseModel):
    prompt: str
    language: str
    normalize_vi_text: bool
    user_record: UserRecord


app = FastAPI()

# Allow requests from specific ports (e.g., frontend on localhost:3000)
origins = [
    "http://localhost:9000", 
    "http://127.0.0.1:9000", 
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # origins allowed to make requests
    allow_credentials=True,
    allow_methods=["*"],    # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # allow all headers
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/synthesize")
def synthesize_text(prompt: str = Body(...), language: str = Body(...), audio_file_pth: str = Body(...)):
    client = Client("thinhlpg/vixtts-demo")
    result = client.predict(
        prompt=prompt,
        language=language,
        audio_file_pth=handle_file(audio_file_pth),
        normalize_text=True,
        api_name="/predict"
    )
    return result

@app.post("/process_audio")
def process_audio(request: AudioRequest):
    # Step 1: Make the POST request
    post_url = "https://thinhlpg-vixtts-demo.hf.space/queue/join?__theme=system"
    session_hash = str(uuid.uuid4())  # Generate a unique session hash
    post_data = {
        "data": [
            request.prompt,
            request.language,
            {
                "path": request.user_record.path,
                "url": request.user_record.recordUrl,
                "orig_name": request.user_record.originalName,
                "size": request.user_record.size,
                "is_stream": False,
                "mime_type": None,
                "meta": {"_type": "gradio.FileData"}
            },
            request.normalize_vi_text
        ],
        "event_data": None,
        "fn_index": 0,
        "trigger_id": 11,
        "session_hash": session_hash
    }

    headers = {"Content-Type": "application/json"}
    post_response = requests.post(post_url, headers=headers, json=post_data)

    if post_response.status_code != 200:
        return {"error": "Failed to initiate processing", "details": post_response.text}

    # Step 2: Listen to the SSE stream
    get_url = f"https://thinhlpg-vixtts-demo.hf.space/queue/data?session_hash={session_hash}"
    client = sseclient.SSEClient(get_url)

    # Listen for messages
    for msg in client:
        print("Received message:", msg.data)

        # Check if the message is "process_completed"
        if '"msg":"process_completed"' in msg.data:
            # Extract the URL directly from the message string
            start_index = msg.data.find('"url":"') + len('"url":"')
            end_index = msg.data.find('"', start_index)
            output_url = msg.data[start_index:end_index]

            print("Process completed. File URL:", output_url)
            # Stop the SSE stream
            break  # Exit the loop after receiving the "process_completed" message

    # response = requests.get(output_url)
    # if response.status_code == 200:
    #     wav_file = io.BytesIO(response.content)  # Load the file into memory
    #     wav_file.seek(0)  # Reset the file pointer to the beginning
    # else:
    #     return {"error": "Failed to fetch the audio file", "details": response.text}

    
    return output_url