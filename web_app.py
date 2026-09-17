"""
web_app.py — Modern AR Sign Language Assistive Web Platform
============================================================
FastAPI backend powering the interactive web dashboard, two-way
communication bridge, Gemini AI expansion, and live recognition.
"""

import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai_expander import AIExpander
from two_way_bridge import TwoWayBridge
from dynamic_trainer import load_vocabulary, save_vocabulary, add_word, remove_word
from prediction_memory import PredictionMemory
from grammar_corrector import GrammarCorrector

app = FastAPI(title="AR Sign Language Communication System")

# Initialize modules
ai_expander = AIExpander()
two_way_bridge = TwoWayBridge()
grammar_corrector = GrammarCorrector()
prediction_memory = PredictionMemory()

# Create static directory if missing
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic schemas
class ExpandRequest(BaseModel):
    words: List[str]
    tone: Optional[str] = "polite"
    api_key: Optional[str] = None


class PartnerSpeechRequest(BaseModel):
    text: str


class SignerSpeechRequest(BaseModel):
    text: str
    raw_signs: Optional[List[str]] = None
    tone: Optional[str] = "polite"


class WordRequest(BaseModel):
    word: str


class PredictRequest(BaseModel):
    landmarks: List[List[float]]  # list of 84-dim frames


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>AR Sign Platform Backend Running</h1>")


# ── System Status ─────────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    word_model_exists = os.path.exists("model/model_mlp.pkl")
    letter_model_exists = os.path.exists("model/model_letter.pkl")
    vocab = load_vocabulary()
    return {
        "status": "online",
        "models": {
            "word_model": word_model_exists,
            "letter_model": letter_model_exists,
        },
        "vocabulary_count": len(vocab.get("words", [])),
        "ai_expansion_ready": bool(ai_expander.api_key),
    }


# ── AI Sentence Expansion ─────────────────────────────────────────────────────
@app.post("/api/expand")
async def expand_sentence(req: ExpandRequest):
    if req.api_key:
        ai_expander.set_api_key(req.api_key)

    expanded = ai_expander.expand(req.words, tone=req.tone)
    return {
        "raw_words": req.words,
        "expanded_sentence": expanded,
        "tone": req.tone,
        "used_ai": bool(ai_expander.api_key),
    }


# ── Two-Way Dialogue Bridge ───────────────────────────────────────────────────
@app.post("/api/dialogue/partner")
async def partner_speaks(req: PartnerSpeechRequest):
    turn = two_way_bridge.add_partner_message(req.text)
    return {"status": "ok", "turn": turn}


@app.post("/api/dialogue/signer")
async def signer_signs(req: SignerSpeechRequest):
    turn = two_way_bridge.add_signer_message(
        text=req.text, raw_signs=req.raw_signs, tone=req.tone
    )
    return {"status": "ok", "turn": turn}


@app.get("/api/dialogue/history")
async def get_dialogue_history():
    return {
        "history": two_way_bridge.get_history(),
        "latest_partner_subtitle": two_way_bridge.get_latest_partner_subtitle(),
    }


@app.post("/api/dialogue/clear")
async def clear_dialogue():
    two_way_bridge.clear()
    return {"status": "cleared"}


@app.get("/api/dialogue/export")
async def export_transcript(format: str = "text"):
    if format == "json":
        return Response(content=two_way_bridge.export_json(), media_type="application/json")
    return PlainTextResponse(two_way_bridge.export_text())


# ── Vocabulary Management ─────────────────────────────────────────────────────
@app.get("/api/vocabulary")
async def get_vocabulary():
    return load_vocabulary()


@app.post("/api/vocabulary")
async def add_vocabulary_word(req: WordRequest):
    success = add_word(req.word.strip().upper())
    if not success:
        raise HTTPException(status_code=400, detail="Word already exists or invalid")
    return {"status": "added", "vocabulary": load_vocabulary()}


@app.delete("/api/vocabulary/{word}")
async def delete_vocabulary_word(word: str):
    success = remove_word(word.strip().upper())
    if not success:
        raise HTTPException(status_code=404, detail="Word not found")
    return {"status": "deleted", "vocabulary": load_vocabulary()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
