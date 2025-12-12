"""
Local Whisper Real-Time-ish STT Adapter

Drop-in-ish replacement for AssemblyAISTT, but using a local Whisper model
via `faster-whisper`.

Input: PCM 16-bit mono audio chunks (bytes) at 16 kHz
Output: STT events (currently STTOutputEvent for each detected utterance)

You can extend this to emit STTChunkEvent for partials if desired.
"""

import asyncio
import contextlib
from typing import AsyncIterator, Optional, List

import numpy as np
from faster_whisper import WhisperModel

from events import STTChunkEvent, STTEvent, STTOutputEvent


from faster_whisper import WhisperModel

class LocalWhisperSTT:
    def __init__(
        self,
        model_size: str = "large-v3",
        sample_rate: int = 16000,
        device: str = "cpu",              # <-- force CPU
        compute_type: str = "int8",       # <-- or "float32" for max compatibility
        silence_threshold: float = 500.0,
        min_silence_chunks: int = 8,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_silence_chunks = min_silence_chunks
        self.device = device
        self.compute_type = compute_type

        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._close_signal = asyncio.Event()

        print(f"[INFO] Loading Whisper model '{model_size}' on {self.device} ({self.compute_type})")
        self._model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
        )



    # -------------------------------------------------------------------------
    # Public API (parallel to AssemblyAISTT)
    # -------------------------------------------------------------------------

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        buffer = bytearray()
        silence_chunks = 0

        while True:
            # Wait indefinitely for next chunk (or sentinel)
            try:
                chunk = await self._audio_queue.get()
            except asyncio.CancelledError:
                break

            # Sentinel: None means flush + exit
            if chunk is None:
                if buffer:
                    text = await self._transcribe_async(bytes(buffer))
                    if text.strip():
                        yield STTOutputEvent.create(text)
                break

            # (Optional) if you really want to respect _close_signal,
            # you can check it here but still process until sentinel.
            # if self._close_signal.is_set():
            #     ...

            # Normal audio path
            buffer.extend(chunk)

            if self._is_silence(chunk):
                silence_chunks += 1
            else:
                silence_chunks = 0

            # When we detect enough silence, treat buffer as one utterance
            if silence_chunks >= self.min_silence_chunks and buffer:
                pcm = bytes(buffer)
                buffer.clear()
                silence_chunks = 0

                text = await self._transcribe_async(pcm)
                if text.strip():
                    yield STTOutputEvent.create(text)


    async def close(self) -> None:
        """
        Signal the STT adapter to finish processing and exit receive_events().
        """
        self._close_signal.set()
        # Put a sentinel into the queue so receive_events can flush & exit
        with contextlib.suppress(asyncio.QueueFull):
            await self._audio_queue.put(None)


    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Queue an audio chunk (PCM 16-bit mono @ sample_rate Hz).
        Mirrors AssemblyAISTT.send_audio(), but instead of sending to WebSocket,
        we put it into a local queue consumed by receive_events().
        """
        await self._audio_queue.put(audio_chunk)

    async def close(self) -> None:
        """
        Signal the STT adapter to finish processing and exit receive_events().
        Mirrors AssemblyAISTT.close().
        """
        self._close_signal.set()
        # Put sentinel to unblock queue consumer
        with contextlib.suppress(asyncio.QueueFull):
            await self._audio_queue.put(None)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _is_silence(self, audio_chunk: bytes) -> bool:
        """
        Very simple energy-based VAD: if RMS < threshold, call it silence.
        """
        if not audio_chunk:
            return True

        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return True

        rms = np.sqrt(np.mean(samples**2))
        return rms < self.silence_threshold

    async def _transcribe_async(self, pcm_bytes: bytes) -> str:
        """
        Run Whisper transcription in a threadpool so we don't block the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_blocking, pcm_bytes)

    def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
        """
        Blocking transcription call. Converts PCM16 bytes → float32 → WhisperModel.
        """
        if not pcm_bytes:
            return ""

        # PCM 16-bit mono -> float32 [-1, 1]
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # faster-whisper can take numpy arrays directly
        segments, info = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            without_timestamps=True,
            vad_filter=True,                # <-- Enable VAD
            condition_on_previous_text=False, # <-- Prevent loops
        )

        texts: List[str] = []
        for seg in segments:
            if seg.text:
                texts.append(seg.text.strip())
        
        raw_text = " ".join(texts).strip()
        return self._filter_hallucinations(raw_text)

    def _filter_hallucinations(self, text: str) -> str:
        """
        Filter out common Whisper hallucinations.
        """
        # Common hallucinations on silence
        hallucinations = {
            "you", "you.", "You", "You.",
            "Thank you.", "Thank you", "Thanks.",
            "MBC News", "MBC",
            "you you you",
            "Bye.", "Bye",
        }
        
        cleaned = text.strip()
        
        # If the entire text is just a hallucination, drop it
        if cleaned in hallucinations:
            return ""
            
        # Check for repetitive "you you you" patterns specifically if they are part of a longer string but dominate it?
        # For now, let's just be simple. If it's *only* repetitions of "you", kill it.
        if cleaned.lower().replace(" ", "").replace(".", "") == "you" * (len(cleaned.lower().replace(" ", "").replace(".", "")) // 3):
             return ""

        return cleaned


git config --global user.name "anycookie112"
git config --global user.email "anycookiefor@gmail.com"


