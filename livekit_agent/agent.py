import asyncio
import json
import logging
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    ToolError,
    cli,
    function_tool,
    inference,
    room_io,
    utils,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent-plant-biotech-assistant")

# For local development. On LiveKit Cloud, set env vars via project secrets.
load_dotenv(".env.local")

BACKEND_ASK_URL = "https://physical-ai-backend-production-1c69.up.railway.app/ask"


class DefaultAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                'You are Plant AI Assistant for the book "AI Revolution in Plant Biotechnology".\n\n'
                'Your role:\n'
                '- Answer questions about ML in agriculture, computer vision for plants, genomics\n'
                '- Explain concepts from the textbook in a clear, educational manner\n'
                '- Be helpful and concise in voice conversations\n'
                "- If you don't know something, say so honestly\n\n"
                'Keep responses brief (2-3 sentences) since this is a voice conversation.\n'
                'Speak naturally as if talking to a student.\n\n'
                'Important: For any content question, call the tool ask_textbook and speak its returned answer. '
                'If the tool fails, apologize and ask the user to retry.'
            )
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Hi! I'm your Plant Biotechnology AI assistant. "
                "Ask me anything about machine learning in agriculture, computer vision for plants, or genomics from the textbook!"
            ),
            allow_interruptions=True,
        )

    @function_tool(name="ask_textbook")
    async def _http_tool_ask_textbook(
        self,
        context: RunContext,
        question: Optional[str] = None,
        selected_text: Optional[str] = None,
    ) -> str:
        """Search the Plant Biotechnology textbook for relevant information."""

        context.disallow_interruptions()

        headers = {"Content-Type": "application/json"}
        payload = {"question": question, "selected_text": selected_text}

        try:
            session = utils.http_context.http_session()
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.post(BACKEND_ASK_URL, timeout=timeout, headers=headers, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise ToolError(f"error: HTTP {resp.status}: {body}")

                # Backend returns JSON: { answer, context_used, model }
                try:
                    data = await resp.json(content_type=None)
                    if isinstance(data, dict) and data.get("answer"):
                        return str(data["answer"])
                    return json.dumps(data)
                except Exception:
                    return await resp.text()
        except ToolError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise ToolError(f"error: {e!s}") from e


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="plant-biotech-assistant")
async def entrypoint(ctx: JobContext):
    # Ensure we are connected to the room before starting the session.
    await ctx.connect()

    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",
            language="en",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=(
                    lambda params: noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
