"""Small asynchronous ComfyUI client used by the private video worker."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
import uuid
from urllib.parse import urlencode

import httpx
import websockets

from .workflow import VIDEO_EXPORT_NODE

MODEL_LOADER_TYPES = {
    "UNETLoader": "diffusion_model_loader",
    "CLIPLoader": "text_encoder_loader",
    "VAELoader": "vae_loader",
    "LoraLoaderModelOnly": "lora_loader",
}
MODEL_LOADER_NODES = {
    "1": "diffusion_model_loader",
    "2": "text_encoder_loader",
    "3": "video_vae_loader",
    "4": "audio_vae_loader",
}


class ModelLoaderError(RuntimeError):
    """Sanitized classification for known model-file/mmap loader failures."""

    def __init__(self, loader: str, kind: str):
        self.loader = loader
        self.kind = kind
        super().__init__("Model loading failed; check model storage, permissions, and mmap compatibility")


def _model_loader_failure(data: dict) -> ModelLoaderError | None:
    """Classify only known loader nodes and explicit mmap/file access failures."""
    node_type = data.get("node_type")
    loader = MODEL_LOADER_TYPES.get(node_type)
    if not loader:
        loader = MODEL_LOADER_NODES.get(str(data.get("node_id") or data.get("node") or ""))
    if not loader:
        return None
    details = " ".join(
        value for value in (data.get("exception_type"), data.get("exception_message"))
        if isinstance(value, str)
    ).lower()
    if any(marker in details for marker in (
        "mmap", "memory map", "failed to map", "cannot map", "cannot allocate memory",
    )):
        return ModelLoaderError(loader, "mmap")
    if any(marker in details for marker in (
        "safetensors", "permission denied", "input/output error", "file too short",
        "unexpected eof", "no such file", "file not found", "invalid header", "not a regular file",
    )):
        return ModelLoaderError(loader, "file_access")
    return None


def _raise_history_loader_failure(history: dict, prompt_id: str):
    record = history.get(prompt_id, {})
    status = record.get("status", {}) if isinstance(record, dict) else {}
    messages = status.get("messages", []) if isinstance(status, dict) else []
    for item in messages:
        if not isinstance(item, (list, tuple)) or len(item) < 2 or item[0] != "execution_error":
            continue
        data = item[1]
        if isinstance(data, dict):
            loader_error = _model_loader_failure(data)
            if loader_error:
                raise loader_error


class ComfyClient:
    def __init__(self, base_url: str, ws_url: str):
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url
        self.http = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(60, connect=10))

    async def close(self):
        await self.http.aclose()

    async def health(self, *, audio: bool = True):
        response = await self.http.get("/system_stats")
        response.raise_for_status()
        if not any(device.get("type") == "cuda" for device in response.json().get("devices", [])):
            raise RuntimeError("CUDA device is unavailable")
        schemas = await self.http.get("/object_info")
        schemas.raise_for_status()
        available = schemas.json()
        required = {"UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "MiniMaxH3ImageToVideo", "BasicScheduler", "KSamplerSelect", "RandomNoise", "BasicGuider", "SamplerCustomAdvanced", "VAEDecode", VIDEO_EXPORT_NODE}
        if audio:
            required.add("VAEDecodeAudio")
        missing = required.difference(available)
        if missing:
            raise RuntimeError("Required ComfyUI nodes are unavailable")
        return set(available)

    async def upload(self, path, name: str, content_type: str = "image/jpeg") -> str:
        with path.open("rb") as image:
            response = await self.http.post("/upload/image", files={"image": (name, image, content_type)}, data={"overwrite": "true"})
        response.raise_for_status()
        data = response.json()
        return f"{data.get('subfolder', '') + '/' if data.get('subfolder') else ''}{data['name']}"

    async def queue(self, workflow: dict, client_id: str) -> str:
        response = await self.http.post("/prompt", json={"prompt": workflow, "client_id": client_id})
        response.raise_for_status()
        return response.json()["prompt_id"]

    async def history(self, prompt_id: str) -> dict:
        response = await self.http.get(f"/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    async def wait(self, prompt_id: str, client_id: str, report, cancelled, timeout_seconds: int = 90 * 60) -> dict | None:
        query = urlencode({"clientId": client_id})
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        try:
            async with websockets.connect(f"{self.ws_url}?{query}", open_timeout=10, ping_interval=20) as socket:
                while asyncio.get_running_loop().time() < deadline:
                    if cancelled():
                        return None
                    try:
                        raw = await asyncio.wait_for(socket.recv(), timeout=2)
                    except asyncio.TimeoutError:
                        history = await self.history(prompt_id)
                        if prompt_id in history:
                            report({"node_started": None})
                            _raise_history_loader_failure(history, prompt_id)
                            return history[prompt_id]
                        continue
                    if not isinstance(raw, str):
                        continue
                    message = json.loads(raw)
                    data = message.get("data", {})
                    if data.get("prompt_id") not in (None, prompt_id):
                        continue
                    if message.get("type") == "progress":
                        report(data)
                    elif message.get("type") == "executing" and data.get("node") is not None:
                        node = str(data.get("node"))
                        report({"node_started": node})
                        stage = "decoding_video" if node in {"12", "13"} else "exporting_video" if node == "15" else None
                        if stage:
                            report({"node": node, "stage": stage})
                    elif message.get("type") == "execution_error":
                        report({"node_started": None})
                        loader_error = _model_loader_failure(data)
                        if loader_error:
                            raise loader_error
                        raise RuntimeError("ComfyUI execution failed")
                    elif message.get("type") == "executing" and data.get("node") is None:
                        report({"node_started": None})
                        # Comfy can emit the terminal event just before history is committed.
                        for _ in range(30):
                            if cancelled():
                                return None
                            history = await self.history(prompt_id)
                            if prompt_id in history:
                                _raise_history_loader_failure(history, prompt_id)
                                return history[prompt_id]
                            await asyncio.sleep(1)
                        raise RuntimeError("ComfyUI history was not committed after execution")
        except (websockets.WebSocketException, httpx.HTTPError):
            while asyncio.get_running_loop().time() < deadline:
                if cancelled():
                    return None
                history = await self.history(prompt_id)
                if prompt_id in history:
                    report({"node_started": None})
                    _raise_history_loader_failure(history, prompt_id)
                    return history[prompt_id]
                await asyncio.sleep(2)
        raise TimeoutError("ComfyUI did not complete the job in time")

    async def download(self, file_info: dict, path: Path, cancelled: Callable[[], bool]) -> None:
        params = {"filename": file_info["filename"], "subfolder": file_info.get("subfolder", ""), "type": file_info.get("type", "output")}
        async with self.http.stream("GET", "/view", params=params) as response:
            response.raise_for_status()
            with path.open("xb") as output:
                async for chunk in response.aiter_bytes(chunk_size=256 * 1024):
                    if cancelled():
                        return
                    output.write(chunk)

    async def cancel(self, prompt_id: str, timeout: int = 120) -> bool:
        try:
            response = await self.http.post(f"/api/jobs/{prompt_id}/cancel")
            response.raise_for_status()
        except httpx.HTTPError:
            await self.http.post("/interrupt", json={"prompt_id": prompt_id})
            await self.http.post("/queue", json={"delete": [prompt_id]})
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            response = await self.http.get("/queue")
            response.raise_for_status()
            queue = response.json()
            active = False
            for key in ("queue_running", "queue_pending"):
                for row in queue.get(key, []):
                    if isinstance(row, (list, tuple)) and len(row) > 1 and row[1] == prompt_id:
                        active = True
                        break
            if not active:
                return True
            await asyncio.sleep(1)
        return False

    async def forget_prompt(self, prompt_id: str):
        response = await self.http.post("/history", json={"delete": [prompt_id]})
        response.raise_for_status()


def new_client_id() -> str:
    return uuid.uuid4().hex
