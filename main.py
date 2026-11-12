import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
from time import monotonic
from zoneinfo import ZoneInfo

from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Record
from astrbot.api.star import Star, StarTools

from .voice import Voice


class HerVoice(Star):
    def __init__(self, context, config):
        super().__init__(context)
        self.config = config
        self.data = Path(StarTools.get_data_dir("astrbot_plugin_her_voice"))
        self.voice = Voice(config["repo_path"], config["port"], self.data)
        self.last_voice = None
        self.send_lock = asyncio.Lock()

    def _is_owner_chat(self, event):
        registered = self.context.get_registered_star("astrbot_plugin_proactive_chat")
        proactive = getattr(registered, "star_cls", None)
        if not proactive or event.get_platform_name() != "webchat":
            return False
        allowed = proactive.config.get("friend_settings", {}).get("session_list", [])
        return event.unified_msg_origin in allowed and event.is_private_chat()

    async def _send(self, event, text, emotion, limit, proactive=False):
        text = text.strip()
        if not 1 <= len(text) <= limit:
            return f"正文需要在 1 到 {limit} 字之间。"
        async with self.send_lock:
            if proactive and not self.config.get("proactive_enabled", True):
                return "已关闭自主短语音，用文字回复即可。"
            cooldown = max(1, int(self.config.get("proactive_cooldown", 5))) * 60
            if proactive and self.last_voice is not None and monotonic() - self.last_voice < cooldown:
                return "短语音冷却中，用文字回复即可。"
            try:
                audio = await self.voice.synthesize(
                    text, self.config["character"], emotion, self.config["pause"],
                    speed=self.config.get("speed", 1.0),
                    default_emotion=self.config.get("default_emotion", "中立"),
                )
            except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
                return f"语音合成失败：{exc}"
            with tempfile.TemporaryDirectory(dir=self.data) as folder:
                path = Path(folder) / "voice.wav"
                path.write_bytes(audio)
                try:
                    await event.send(MessageChain([Record(file=str(path))]))
                except Exception:
                    return "网页语音投递未确认，请勿自动重发。"
            self.last_voice = monotonic()
            return "已提交网页语音。不要再发送相同正文或汇报发送结果。"

    @filter.llm_tool(name="her_voice_speak")
    async def speak(self, event: AstrMessageEvent, text: str, emotion: str = "", proactive: bool = False):
        """用当前角色发一条语音。可在日常私聊主动选用短语音，或按用户要求朗读真实文本。

        Args:
            text(string): 要念的正文。主动语音最多100字，点名朗读最多500字。
            emotion(string): 可选参考语气，例如中立、开心、一点悲伤、中立谨慎。拿不准留空。
            proactive(boolean): 自己决定用语音时为 true，用户明确要求时为 false。
        """
        if not self._is_owner_chat(event):
            return "仅在已绑定的主人网页私聊可使用语音。"
        return await self._send(event, text, emotion, 100 if proactive else 500, proactive)

    @filter.llm_tool(name="her_bedtime_radio")
    async def radio(self, event: AstrMessageEvent, text: str, emotion: str = ""):
        """发送一期晚安电台。主人想听完整电台时使用，单说晚安或只要一句话时用短语音。

        Args:
            text(string): 约300–500字完整节目正文。可用当前聊天事实、心情或明确虚构的睡前故事。
            emotion(string): 参考语气，默认用当前角色的中立参考。
        """
        if not self.config["radio_enabled"] or not self._is_owner_chat(event):
            return "当前会话未启用晚安电台。"
        return await self._send(event, text, emotion, 500)

    async def terminate(self):
        await self.voice.close()
