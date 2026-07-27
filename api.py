from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

app = FastAPI()

state = {
    "people_now": 0,
    "total_people": 0,
    "fps": 0,
    "camera": False,
    "active_ids": []
}

class StatusResponse(BaseModel):
    people_now: int
    total_people: int
    fps: float
    camera: bool

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        people_now=state["people_now"],
        total_people=state["total_people"],
        fps=state["fps"],
        camera=state["camera"]
    )

@app.get("/people")
async def get_people():
    return {"people_now": state["people_now"]}

@app.get("/fps")
async def get_fps():
    return {"fps": state["fps"]}

@app.get("/total")
async def get_total():
    return {"total_people": state["total_people"]}

@app.post("/reset")
async def reset_counter():
    return {"status": "reset requested"}

@app.post("/start")
async def start_camera():
    return {"status": "start requested"}

@app.post("/stop")
async def stop_camera():
    return {"status": "stop requested"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = {
                "people_now": state["people_now"],
                "total_people": state["total_people"],
                "fps": state["fps"],
                "timestamp": datetime.now().isoformat(),
                "active_ids": state["active_ids"]
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
