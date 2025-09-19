import os
import io
import tempfile
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import transformers
from transformers import pipeline
from google.cloud import texttospeech
import asyncio
import uvicorn

# Import our agent
from services.agent import get_agent

# Initialize FastAPI app
app = FastAPI(title="Awaaz Seva API", version="1.0.0")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize STT pipeline (Whisper)
stt_pipeline = None

# In-memory storage for chat history (for hackathon)
chat_sessions = {}
chat_history = []

# Load existing chat history if available
HISTORY_FILE = "chat_history.json"

def load_chat_history():
    """Load chat history from file"""
    global chat_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
                print(f"Loaded {len(chat_history)} chat messages from history")
    except Exception as e:
        print(f"Failed to load chat history: {e}")
        chat_history = []

def save_chat_history():
    """Save chat history to file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save chat history: {e}")

class TextQuestionRequest(BaseModel):
    question: str
    language_preference: str = "auto"
    include_audio: bool = True
    search_mode: str = "auto"

class NewSessionResponse(BaseModel):
    session_id: str
    message: str

class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]]

def init_stt():
    """Initialize Speech-to-Text pipeline"""
    global stt_pipeline
    if stt_pipeline is None:
        stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=0 if torch.cuda.is_available() else -1
        )

def text_to_speech(text: str, lang_code: str = "en-US") -> bytes:
    """Convert text to speech using Google TTS"""
    try:
        client = texttospeech.TextToSpeechClient()
        
        if lang_code.startswith('hi'):
            language_code = "hi-IN"
            voice_name = "hi-IN-Neural2-A"
        else:
            language_code = "en-US"
            voice_name = "en-US-Neural2-F"
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        return response.audio_content
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def save_audio_file(audio_content: bytes) -> str:
    """Save audio content and return URL path"""
    if not audio_content:
        return None
    
    # Create audio directory if not exists
    os.makedirs("audio_files", exist_ok=True)
    
    # Generate unique filename
    filename = f"response_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("audio_files", filename)
    
    with open(filepath, "wb") as f:
        f.write(audio_content)
    
    return f"/audio/{filename}"

# Serve audio files
app.mount("/audio", StaticFiles(directory="audio_files"), name="audio")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    init_stt()
    load_chat_history()  # Load existing chat history

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Awaaz Seva API is running!"}

@app.post("/api/new-session", response_model=NewSessionResponse)
async def start_new_session():
    """Start new chat session"""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    return NewSessionResponse(
        session_id=session_id,
        message="New session started successfully"
    )

@app.get("/api/history", response_model=HistoryResponse)
async def get_chat_history():
    """Get chat history"""
    return HistoryResponse(history=chat_history)

@app.post("/api/ask")
async def handle_text_question(request: TextQuestionRequest):
    """Handle text question - matches your frontend exactly"""
    try:
        # Get agent and process query
        agent = get_agent()
        result = agent.process_query(request.question)
        
        # Generate audio if requested
        audio_url = None
        if request.include_audio:
            audio_content = text_to_speech(
                result["answer"], 
                result["detected_language"]
            )
            if audio_content:
                audio_url = save_audio_file(audio_content)
        
        # Save to history
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": request.question,
            "answer": result["answer"],
            "language": result["detected_language"],
            "type": "text"
        }
        chat_history.append(chat_entry)
        save_chat_history()  # Save to file
        
        # Response format matching your frontend expectations
        return {
            "answer": result["answer"],
            "audio_url": audio_url,
            "detected_language": result["detected_language"],
            "original_query": result["original_query"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/api/voice-ask")
async def handle_voice_question(
    audio: UploadFile = File(...),
    language_preference: str = Form("auto"),
    include_audio: str = Form("true"),
    search_mode: str = Form("auto")
):
    """Handle voice question - matches your frontend exactly"""
    try:
        # Initialize STT if needed
        if stt_pipeline is None:
            init_stt()
        
        # Read and process audio file
        audio_bytes = await audio.read()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            # Transcribe audio to text
            transcription = stt_pipeline(tmp_file_path)
            transcribed_text = transcription["text"]
            
            # Process the transcribed text
            agent = get_agent()
            result = agent.process_query(transcribed_text)
            
            # Generate audio response if requested
            audio_url = None
            if include_audio.lower() == "true":
                audio_content = text_to_speech(
                    result["answer"],
                    result["detected_language"]
                )
                if audio_content:
                    audio_url = save_audio_file(audio_content)
            
            # Save to history
            chat_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": transcribed_text,
                "answer": result["answer"],
                "language": result["detected_language"],
                "type": "voice"
            }
            chat_history.append(chat_entry)
            save_chat_history()  # Save to file
            
            # Response format matching your frontend expectations
            return {
                "transcription": {
                    "text": transcribed_text,
                    "language": result["detected_language"]
                },
                "answer": {
                    "text": result["answer"],
                    "audio_url": audio_url
                },
                "detected_language": result["detected_language"]
            }
            
        finally:
            # Clean up uploaded file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice: {str(e)}")

@app.get("/audio/{filename}")
async def serve_audio_file(filename: str):
    """Serve audio files"""
    file_path = os.path.join("audio_files", filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

# Keep your existing test endpoints
@app.post("/test/tts")
async def test_tts(query: TextQuestionRequest):
    """Test TTS functionality"""
    try:
        audio_content = text_to_speech(query.question, "en-US")
        
        if audio_content:
            audio_url = save_audio_file(audio_content)
            return {"message": "TTS successful", "audio_url": audio_url}
        else:
            return {"message": "TTS failed"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS test failed: {str(e)}")

@app.post("/test/stt")
async def test_stt(audio_file: UploadFile = File(...)):
    """Test STT functionality"""
    try:
        if stt_pipeline is None:
            init_stt()
        
        audio_bytes = await audio_file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            transcription = stt_pipeline(tmp_file_path)
            return {"transcribed_text": transcription["text"]}
        finally:
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT test failed: {str(e)}")

if __name__ == "__main__":
    # Create audio directory
    os.makedirs("audio_files", exist_ok=True)
    
    # Required environment variables check
    required_env_vars = [
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Warning: Missing environment variables: {missing_vars}")
        print("The application may not work correctly without these.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)