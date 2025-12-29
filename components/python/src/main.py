import asyncio
import contextlib
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.checkpoint.memory import InMemorySaver
from starlette.staticfiles import StaticFiles

# from assemblyai_stt import AssemblyAISTT
# from components.python.src.cartesia_tts import CartesiaTTS
from events import (
    AgentChunkEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    VoiceAgentEvent,
    event_to_dict,
)
from utils import merge_async_iters
from whisper_stt import LocalWhisperSTT 
from whisper_pytorch import WhisperPytorchSTT
from kokoro_tts import KokoroTTS
from models import get_ollama_model, get_groq_model
from vibevoice_tts import VibeVoiceAsyncTTS
load_dotenv()

# Static files are served from the shared web build output
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Web build not found at {STATIC_DIR}. "
        "Run 'make build-web' or 'make dev-py' from the project root."
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."


def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


system_prompt = """
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.

The price for any sandwich is $5 plus $1 for each topping, meat, or cheese.

${CARTESIA_TTS_SYSTEM_PROMPT}
"""

# 1. Check which provider to use (Defaults to "groq" if not set)
provider = os.getenv("LLM_PROVIDER", "groq").lower()

if provider == "ollama":
    print("--> Using LLM Provider: Ollama")
    llm = get_ollama_model()
else:
    print("--> Using LLM Provider: Groq")
    # 2. Get Key from Environment (Don't hardcode "gsk_...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables!")
    
    llm = get_groq_model(api_key=api_key)

from data_visualisation.main import main2 as make_agent
# agent = create_agent(
#     model=llm,
#     tools=[add_to_order, confirm_order],
#     system_prompt=system_prompt,
#     checkpointer=InMemorySaver(),
# )


agent = make_agent(llm)




async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (Bytes) → Voice Events (VoiceAgentEvent)

    This function takes a stream of audio chunks and sends them to AssemblyAI for STT.

    It uses a producer-consumer pattern where:
    - Producer: A background task reads audio chunks from audio_stream and sends
      them to AssemblyAI via WebSocket. This runs concurrently with the consumer,
      allowing transcription to begin before all audio has arrived.
    - Consumer: The main coroutine receives transcription events from AssemblyAI
      and yields them downstream. Events include both partial results (stt_chunk)
      and final transcripts (stt_output).

    Args:
        audio_stream: Async iterator of PCM audio bytes (16-bit, mono, 16kHz)

    Yields:
        STT events (stt_chunk for partials, stt_output for final transcripts)
    """
    # stt = WhisperPytorchSTT(
    #         model_size="large-v3-turbo",
    #         sample_rate=16000,          # <= IMPORTANT: use the WAV's SR (likely 24000)
    #         device="cuda",           # or "cpu" if you want CPU
    #         compute_type="float16",  # safe
    #         silence_threshold=50.0,  # make VAD more permissive
    #         min_silence_chunks=3,    # detect utterance quickly
    #     )
    stt = LocalWhisperSTT(
        model_size="large-v3-turbo", # or "distil-large-v3" for 3x speed
        device="cpu",         # FORCE CUDA
        compute_type="int8" # FORCE FLOAT16
    )

    async def send_audio():
        """
        Background task that pumps audio chunks to AssemblyAI.

        This runs concurrently with the main coroutine, continuously reading
        audio chunks from the input stream and forwarding them to AssemblyAI.
        When the input stream ends, it signals completion by closing the
        WebSocket connection.
        """
        try:
            # Stream each audio chunk to AssemblyAI as it arrives
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            # Signal to AssemblyAI that audio streaming is complete
            await stt.close()

    # Launch the audio sending task in the background
    # This allows us to simultaneously receive transcripts in the main coroutine
    send_task = asyncio.create_task(send_audio())

    try:
        # Consumer loop: receive and yield transcription events as they arrive
        # from AssemblyAI. The receive_events() method listens on the WebSocket
        # for transcript events and yields them as they become available.
        async for event in stt.receive_events():
            yield event
    finally:
        # Cleanup: ensure the background task is cancelled and awaited
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        # Ensure the WebSocket connection is closed
        await stt.close()


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    FIXED: Uses message.content instead of message.text
    """
    thread_id = str(uuid4())

    async for event in event_stream:
        # 1. Pass through all events (User Input, STT, etc.)
        yield event

        if event.type == "stt_output":
            print(f"DEBUG: [1] STT Output received: {event.transcript}") 
            # Invoke LangChain Agent
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            async for message, metadata in stream:
                # --- PROCESS AI MESSAGES (TEXT) ---
                if isinstance(message, AIMessage):
                    # FIX 1: Use .content, not .text
                    content = message.content
                    
                    # FIX 2: LangChain sometimes yields empty chunks or list-based content
                    if isinstance(content, str) and content:
                        yield AgentChunkEvent.create(content)
                    
                    # --- PROCESS TOOL CALLS ---
                    # Note: handling streaming tool calls can be tricky.
                    # This assumes message.tool_calls is populated fully or accumulatively.
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            yield ToolCallEvent.create(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )

                # --- PROCESS TOOL RESULTS ---
                if isinstance(message, ToolMessage):
                    yield ToolResultEvent.create(
                        tool_call_id=getattr(message, "tool_call_id", ""),
                        name=getattr(message, "name", "unknown"),
                        result=str(message.content) if message.content else "",
                    )

            # Signal end of turn
            yield AgentEndEvent.create()
# async def _agent_stream(
#     event_stream: AsyncIterator[VoiceAgentEvent],
# ) -> AsyncIterator[VoiceAgentEvent]:
#     """
#     Transform stream: Voice Events → Voice Events (with Agent Responses)

#     This function takes a stream of upstream voice agent events and processes them.
#     When an stt_output event arrives, it passes the transcript to the LangChain agent.
#     The agent streams back its response tokens as agent_chunk events.
#     Tool calls and results are also emitted as separate events.
#     All other upstream events are passed through unchanged.

#     The passthrough pattern ensures downstream stages (like TTS) can observe all
#     events in the pipeline, not just the ones this stage produces. This enables
#     features like displaying partial transcripts while the agent is thinking.

#     Args:
#         event_stream: An async iterator of upstream voice agent events

#     Yields:
#         All upstream events plus agent_chunk, tool_call, and tool_result events
#     """
#     # Generate a unique thread ID for this conversation session
#     # This allows the agent to maintain conversation context across multiple turns
#     # using the checkpointer (InMemorySaver) configured in the agent
#     thread_id = str(uuid4())

#     # Process each event as it arrives from the upstream STT stage
#     async for event in event_stream:
#         # Pass through all events to downstream consumers
#         yield event

#         # When we receive a final transcript, invoke the agent
#         if event.type == "stt_output":
#             # Stream the agent's response using LangChain's astream method.
#             # stream_mode="messages" yields message chunks as they're generated.
#             stream = agent.astream(
#                 {"messages": [HumanMessage(content=event.transcript)]},
#                 {"configurable": {"thread_id": thread_id}},
#                 stream_mode="messages",
#             )

#             # Iterate through the agent's streaming response. The stream yields
#             # tuples of (message, metadata), but we only need the message.
#             async for message, metadata in stream:
#                 # Emit agent chunks (AI messages)
#                 if isinstance(message, AIMessage):
#                     # Extract and yield the text content from each message chunk
#                     yield AgentChunkEvent.create(message.text)
#                     # Emit tool calls if present
#                     if hasattr(message, "tool_calls") and message.tool_calls:
#                         for tool_call in message.tool_calls:
#                             yield ToolCallEvent.create(
#                                 id=tool_call.get("id", str(uuid4())),
#                                 name=tool_call.get("name", "unknown"),
#                                 args=tool_call.get("args", {}),
#                             )

#                 # Emit tool results (tool messages)
#                 if isinstance(message, ToolMessage):
#                     yield ToolResultEvent.create(
#                         tool_call_id=getattr(message, "tool_call_id", ""),
#                         name=getattr(message, "name", "unknown"),
#                         result=str(message.content) if message.content else "",
#                     )

#             # Signal that the agent has finished responding for this turn
#             yield AgentEndEvent.create()


# async def _tts_stream(
#     event_stream: AsyncIterator[VoiceAgentEvent],
# ) -> AsyncIterator[VoiceAgentEvent]:
#     """
#     Transform stream: Voice Events → Voice Events (with Audio)

#     This function takes a stream of upstream voice agent events and processes them.
#     When agent_chunk events arrive, it sends the text to Cartesia for TTS synthesis.
#     Audio is streamed back as tts_chunk events as it's generated.
#     All upstream events are passed through unchanged.

#     It uses merge_async_iters to combine two concurrent streams:
#     - process_upstream(): Iterates through incoming events, yields them for
#       passthrough, and sends agent text chunks to Cartesia for synthesis.
#     - tts.receive_events(): Yields audio chunks from Cartesia as they are
#       synthesized.

#     The merge utility runs both iterators concurrently, yielding items from
#     either stream as they become available. This allows audio generation to
#     begin before the agent has finished generating all text, minimizing latency.

#     Args:
#         event_stream: An async iterator of upstream voice agent events

#     Yields:
#         All upstream events plus tts_chunk events for synthesized audio
#     """
#     # tts = VibeVoiceAsyncTTS(model_path = "/app/models/VibeVoice-Realtime-0.5B")

#     tts = KokoroTTS()

#     async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
#         """
#         Process upstream events, yielding them while sending text to Cartesia.

#         This async generator serves two purposes:
#         1. Pass through all upstream events (stt_chunk, stt_output, agent_chunk)
#            so downstream consumers can observe the full event stream.
#         2. Buffer agent_chunk text and send to Cartesia when agent_end arrives.
#            This ensures the full response is sent at once for better TTS quality.
#         """
#         buffer: list[str] = []
#         async for event in event_stream:
#             # Pass through all events to downstream consumers
#             yield event
#             # Buffer agent text chunks
#             if event.type == "agent_chunk":
#                 buffer.append(event.text)
#             # Send all buffered text to Cartesia when agent finishes
#             if event.type == "agent_end":
#                 await tts.send_text("".join(buffer))
#                 buffer = []

#     try:
#         # Merge the processed upstream events with TTS audio events
#         # Both streams run concurrently, yielding events as they arrive
#         async for event in merge_async_iters(process_upstream(), tts.receive_events()):
#             yield event
#     finally:
#         # Cleanup: close the WebSocket connection to Cartesia
#         await tts.close()


import re # Add this import at the top

async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    
    # Initialize your TTS (VibeVoice or Kokoro)
    # tts = VibeVoiceAsyncTTS(model_path="/app/models/VibeVoice-Realtime-0.5B")
    tts = VibeVoiceAsyncTTS(model_path="/home/robust/models/VibeVoice-Realtime-0.5B")
    # tts = KokoroTTS() 

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        # Buffer to accumulate partial text chunks
        text_buffer = ""
        
        async for event in event_stream:
            # 1. Pass ALL events to the UI immediately (So text bubbles appear)
            yield event

            # 2. Process Text for TTS
            if event.type == "agent_chunk":
                text_buffer += event.text
                
                # Check if we have a full sentence (ends in . ? ! followed by space or newline)
                # We split iteratively to handle multiple sentences in one chunk
                while True:
                    # Regex: Find punctuation [.?!] followed by whitespace or end of string
                    match = re.search(r'([.?!]+)(\s+|$)', text_buffer)
                    if match:
                        end_idx = match.end()
                        sentence = text_buffer[:end_idx]
                        
                        # Send the complete sentence to TTS
                        if sentence.strip():
                            await tts.send_text(sentence)
                        
                        # Remove processed sentence from buffer
                        text_buffer = text_buffer[end_idx:]
                    else:
                        # No end of sentence found yet, keep buffering
                        break
            
            # 3. Flush remaining text when agent is done
            elif event.type == "agent_end":
                if text_buffer.strip():
                    await tts.send_text(text_buffer)
                text_buffer = "" # Reset for next turn

    try:
        # Merge the upstream (Agent) and downstream (TTS Audio) streams
        async for event in merge_async_iters(process_upstream(), tts.receive_events()):
            yield event
    finally:
        await tts.close()


pipeline = (
    RunnableGenerator(_stt_stream)  # Audio -> STT events
    | RunnableGenerator(_agent_stream)  # STT events -> STT + Agent events
    | RunnableGenerator(_tts_stream)  # STT + Agent events -> All events
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        """Async generator that yields audio bytes from the websocket."""
        while True:
            data = await websocket.receive_bytes()
            yield data

    output_stream = pipeline.atransform(websocket_audio_stream())

    # Process all events from the pipeline, sending events back to the client
    async for event in output_stream:
        await websocket.send_json(event_to_dict(event))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8112, reload=True)
    # uvicorn.run(
    #     app, 
    #     host="0.0.0.0", 
    #     port=8000,
    #     # Point to where they were copied in the container
    #     ssl_keyfile="/app/key.pem", 
    #     ssl_certfile="/app/cert.pem"
    # )