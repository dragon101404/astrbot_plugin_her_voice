import asyncio
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import wave

import aiohttp


EMOTION_ALIASES = {
    "平静": "中立", "温和": "中立", "高兴": "开心", "愉快": "开心",
    "悲伤": "一点悲伤", "难过": "一点悲伤", "谨慎": "中立谨慎",
}


def profile(root, character, emotion="", default_emotion="中立"):
    if not re.fullmatch(r"[\w -]{1,40}", character):
        raise ValueError("角色名无效。")
    gpt = sorted((root / "GPT_weights_v4").glob(f"{character}-*.ckpt"))
    sovits = sorted((root / "SoVITS_weights_v4").glob(f"{character}_e*.pth"))
    refs = {}
    for file in sorted((root / "参考音频" / character).glob("*.wav")):
        match = re.fullmatch(r"【([^】]+)】(.+)", file.stem)
        if match:
            refs[match[1]] = (file, match[2].rstrip("."))
    if len(gpt) != 1 or len(sovits) != 1 or not refs:
        raise ValueError("该角色需要一套 V4 GPT、SoVITS 权重以及带语气标签的参考音频。")
    emotion = emotion.strip()
    alias = EMOTION_ALIASES.get(emotion)
    label = emotion if emotion in refs else alias if alias in refs else emotion
    if not label:
        label = next(value for value in (default_emotion, "中立", next(iter(refs))) if value in refs)
    if label not in refs:
        raise ValueError("可选参考语气：" + "、".join(refs))
    return (gpt[0], sovits[0]), refs[label]


class Voice:
    def __init__(self, root, port, directory):
        self.root = Path(root).resolve()
        self.port = int(port)
        self.directory = Path(directory)
        self.base = f"http://127.0.0.1:{self.port}"
        self.lock = asyncio.Lock()
        self.process = None
        self.weights = None
        self.idle = None

    async def _stop(self):
        process, self.process = self.process, None
        self.weights = None
        if process and process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 8)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)

    async def close(self):
        if self.idle:
            self.idle.cancel()
        async with self.lock:
            await self._stop()

    async def _idle_stop(self):
        await asyncio.sleep(300)
        async with self.lock:
            await self._stop()

    async def _ready(self, http, weights):
        if self.process and self.process.poll() is None and self.weights == weights:
            return
        await self._stop()
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", self.port), 1)
        except (OSError, TimeoutError):
            pass
        else:
            writer.close()
            await writer.wait_closed()
            raise RuntimeError("本插件专用端口已被其他进程占用。")
        cfg = self.directory / "sovits-runtime.json"
        pretrained = self.root / "GPT_SoVITS" / "pretrained_models"
        cfg.write_text(json.dumps({"custom": {
            "version": "v4", "device": "cuda", "is_half": True,
            "t2s_weights_path": str(weights[0]), "vits_weights_path": str(weights[1]),
            "bert_base_path": str(pretrained / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(pretrained / "chinese-hubert-base"),
        }}, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with (self.directory / "sovits.log").open("wb") as log:
            self.process = subprocess.Popen(
                [str(self.root / "runtime/python.exe"), "-I", "-u", "-X", "utf8", "api_v2.py",
                 "-a", "127.0.0.1", "-p", str(self.port), "-c", str(cfg)],
                cwd=self.root, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and self.process.poll() is None:
            try:
                async with http.get(self.base + "/openapi.json", timeout=2, allow_redirects=False) as response:
                    if response.status == 200 and "/tts" in (await response.json()).get("paths", {}):
                        self.weights = weights
                        return
            except (aiohttp.ClientError, TimeoutError, ValueError):
                pass
            await asyncio.sleep(1)
        await self._stop()
        raise RuntimeError("GPT-SoVITS 启动失败，请查看插件目录中的 sovits.log。")

    async def synthesize(self, text, character, emotion, pause, *, speed=1.0, default_emotion="中立"):
        weights, (reference, prompt) = profile(self.root, character, emotion, default_emotion)
        async with self.lock:
            if self.idle:
                self.idle.cancel()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as http:
                try:
                    await self._ready(http, weights)
                    async with http.post(self.base + "/tts", allow_redirects=False, json={
                        "text": text, "text_lang": "zh", "ref_audio_path": str(reference),
                        "prompt_text": prompt, "prompt_lang": "zh",
                        "speed_factor": max(0.7, min(1.3, float(speed))),
                        "text_split_method": "cut3", "fragment_interval": max(0, min(2, float(pause))),
                        "batch_size": 1, "media_type": "wav", "streaming_mode": False,
                        "parallel_infer": False, "seed": -1,
                    }) as response:
                        if response.status != 200:
                            raise RuntimeError(f"GPT-SoVITS 合成失败（HTTP {response.status}）。")
                        audio = await response.read()
                    try:
                        with wave.open(io.BytesIO(audio), "rb") as reader:
                            if reader.getnframes() <= 0:
                                raise RuntimeError("GPT-SoVITS 返回了空音频。")
                    except (wave.Error, EOFError) as exc:
                        raise RuntimeError("GPT-SoVITS 返回的内容不是可用的 WAV。") from exc
                    return audio
                except aiohttp.ClientError as exc:
                    raise RuntimeError("无法与本机 GPT-SoVITS 通信。") from exc
                except asyncio.CancelledError:
                    await self._stop()
                    raise
                finally:
                    if self.process:
                        self.idle = asyncio.create_task(self._idle_stop())
