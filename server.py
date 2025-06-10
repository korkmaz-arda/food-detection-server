from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import time
import logging
import argon2
from pathlib import Path
import os
import shutil
from typing import Dict, Optional
import tempfile

from detection import DetectionManager
from model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()


def load_auth_token():
    try:
        with open(".auth_token", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning("No .auth_token file found. Admin functions will be disabled.")
        return None


MASTER_PASSWORD_HASH = load_auth_token()


def verify_password(password: str) -> bool:
    if not MASTER_PASSWORD_HASH:
        return False
    ph = argon2.PasswordHasher()
    try:
        ph.verify(MASTER_PASSWORD_HASH, password)
        return True
    except Exception:
        return False


detection_manager: Optional[DetectionManager] = None
active_connections: Dict[str, WebSocket] = {}

stats = {
    "total_frames": 0,
    "total_detections": 0,
    "start_time": time.time(),
    "models_loaded": False
}


@app.on_event("startup")
async def startup_event():
    global detection_manager
    try:
        active_models = model_manager.get_active_models()
        if not active_models.get("tray") or not active_models.get("food"):
            logger.warning("Active models not found. Please set active models.")
            stats["models_loaded"] = False
            return

        registry = model_manager._load_registry()
        active_tray = registry.get("active", {}).get("tray")
        active_food = registry.get("active", {}).get("food")
        tray_config = model_manager.get_model_config("tray", active_tray) if active_tray else {}
        food_config = model_manager.get_model_config("food", active_food) if active_food else {}

        detection_manager = DetectionManager(
            tray_model_path=active_models["tray"],
            food_model_path=active_models["food"],
            skip_frames=8,
            tray_conf=tray_config.get("confidence", 0.65),
            food_conf=food_config.get("confidence", 0.2),
            food_classes=food_config.get("food_classes")
        )
        stats["models_loaded"] = True
        logger.info(f"Detection manager initialized with models: tray={active_tray}, food={active_food}")

    except Exception as e:
        logger.error(f"Failed to initialize detection manager: {e}")
        stats["models_loaded"] = False


@app.get("/")
async def root():
    return HTMLResponse(f"""
    <html>
        <head>
            <title>Real-time Food Detection Server</title>
        </head>
        <body>
            <h1>Real-time Food Detection Server</h1>
            <p>Status: {"Ready" if stats["models_loaded"] else "Models not loaded"}</p>
            <p>Model management: <a href="/models">/models</a></p>
            <p>User interface: <a href="/user">/user</a></p>
            <p>Developer interface: <a href="/dev">/dev</a></p>
            <p>Server uptime: {int(time.time() - stats["start_time"])} seconds</p>
            <p>Total frames processed: {stats["total_frames"]}</p>
            <p>Total detections: {stats["total_detections"]}</p>
            <p>Active connections: {len(active_connections)}</p>
        </body>
    </html>
    """)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    active_connections[client_id] = websocket
    logger.info(f"Client {client_id} connected. Total connections: {len(active_connections)}")

    try:
        registry = model_manager._load_registry()
        active_food = registry.get("active", {}).get("food")
        food_config = model_manager.get_model_config("food", active_food) if active_food else {}

        await websocket.send_json({
            "type": "status",
            "models_loaded": stats["models_loaded"],
            "message": f"Connected to detection server. Client ID: {client_id}",
            "model_config": {
                "food_classes": food_config.get("food_classes", []),
                "food_prices": food_config.get("food_prices", {})
            }
        })

        if not stats["models_loaded"] or detection_manager is None:
            await websocket.send_json({
                "type": "error",
                "message": "Models not loaded. Please upload models first."
            })
            return

        detection_manager.reset()

        while True:
            data = await websocket.receive_bytes()
            stats["total_frames"] += 1

            try:
                result = detection_manager.process_frame(data)
                total_detections = len(result["trays"]) + len(result["foods"])
                stats["total_detections"] += total_detections

                await websocket.send_json({
                    "type": "detection",
                    "data": result
                })

                if stats["total_frames"] % 100 == 0:
                    logger.info(f"Processed {stats['total_frames']} frames, {stats['total_detections']} total detections")

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Frame processing error: {str(e)}"
                })

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error with client {client_id}: {e}")
    finally:
        if client_id in active_connections:
            del active_connections[client_id]
        logger.info(f"Client {client_id} cleanup complete. Remaining connections: {len(active_connections)}")


@app.get("/api/models/{model_type}")
async def list_models(model_type: str):
    if model_type not in ["tray", "food"]:
        raise HTTPException(status_code=400, detail="Model type must be 'tray' or 'food'")
    return model_manager.list_models(model_type)


@app.post("/api/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    name: Optional[str] = Form(None),
    config: Optional[str] = Form(None),
    master_password: Optional[str] = Form(None)
):
    if MASTER_PASSWORD_HASH and not verify_password(master_password or ""):
        raise HTTPException(status_code=403, detail="Invalid password")

    if model_type not in ["food", "tray"]:
        raise HTTPException(status_code=400, detail="Model type must be 'food' or 'tray'")

    if not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="File must be a .pt file")

    model_config = None
    if config:
        try:
            model_config = json.loads(config)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid config JSON")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        from ultralytics import YOLO
        test_model = YOLO(tmp_path)
        result = model_manager.upload_model(model_type, tmp_path, name, model_config)
        logger.info(f"Model uploaded: {result}")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid model file: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/api/models/activate")
async def activate_model(model_type: str = Form(...), model_name: str = Form(...)):
    if model_type not in ["tray", "food"]:
        raise HTTPException(status_code=400, detail="Model type must be 'tray' or 'food'")
    if not model_manager.activate_model(model_type, model_name):
        raise HTTPException(status_code=404, detail="Model not found")
    await startup_event()
    
    registry = model_manager._load_registry()
    active_food = registry.get("active", {}).get("food")
    food_config = model_manager.get_model_config("food", active_food) if active_food else {}
    
    for ws in active_connections.values():
        try:
            await ws.send_json({
                "type": "model_changed",
                "model_config": {
                    "food_classes": food_config.get("food_classes", []),
                    "food_prices": food_config.get("food_prices", {})
                }
            })
        except Exception:
            pass
    return {"status": "success", "message": f"Model {model_name} activated"}


@app.delete("/api/models/{model_type}/{model_name}")
async def delete_model(
    model_type: str,
    model_name: str,
    master_password: Optional[str] = Form(None)
):
    if MASTER_PASSWORD_HASH and not verify_password(master_password or ""):
        raise HTTPException(status_code=403, detail="Invalid password")
    try:
        if model_manager.delete_model(model_type, model_name):
            return {"status": "success", "message": f"Model {model_name} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models")
async def models_admin_page():
    return FileResponse("models_admin.html")


@app.get("/user")
async def user_client_page():
    """Serve the user client page"""
    return FileResponse("user_client.html")


@app.get("/dev")
async def dev_client_page():
    """Serve the developer client page"""
    return FileResponse("dev_client.html")


@app.get("/stats")
async def get_stats():
    uptime = time.time() - stats["start_time"]
    fps = stats["total_frames"] / uptime if uptime > 0 else 0
    return {
        "uptime_seconds": uptime,
        "total_frames": stats["total_frames"],
        "total_detections": stats["total_detections"],
        "average_fps": fps,
        "active_connections": len(active_connections),
        "models_loaded": stats["models_loaded"]
    }


@app.post("/reset-stats")
async def reset_stats():
    stats["total_frames"] = 0
    stats["total_detections"] = 0
    stats["start_time"] = time.time()
    return {"status": "success", "message": "Statistics reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890)
