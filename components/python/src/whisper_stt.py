

# import asyncio
# import logging
# import re
# import numpy as np
# from typing import AsyncIterator, Optional, List
# from faster_whisper import WhisperModel
# from events import STTEvent, STTOutputEvent

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("LocalWhisperSTT")

# class LocalWhisperSTT:
#     def __init__(
#         self,
#         model_size: str = "large-v3", 
#         sample_rate: int = 16000,
#         device: str = "cuda",         
#         compute_type: str = "float16",
#         silence_threshold: float = 100.0, 
#         min_silence_duration: float = 0.6, 
#         min_audio_duration: float = 0.5,   
#         max_buffer_duration: float = 8.0, 
#     ):
#         self.sample_rate = sample_rate
#         self.silence_threshold = silence_threshold
#         self.device = device
#         self.compute_type = compute_type
        
#         # FIX: Store these as instance variables so receive_events can see them
#         self.min_audio_duration = min_audio_duration
#         self.max_buffer_duration = max_buffer_duration
#         self.min_silence_seconds = min_silence_duration

#         # Calculate bytes per second (16-bit mono = 2 bytes per sample)
#         self.bytes_per_second = sample_rate * 2 
        
#         self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
#         self._close_signal = asyncio.Event()

#         logger.info(f"Loading Whisper model '{model_size}' on {self.device}...")
#         self._model = WhisperModel(
#             model_size,
#             device=self.device,
#             compute_type=self.compute_type,
#         )
#         logger.info("Whisper model loaded.")

#     async def receive_events(self) -> AsyncIterator[STTEvent]:
#         buffer = bytearray()
        
#         has_speech = False 
#         silence_start_time = None
        
#         logger.info("Local STT listening... (Speak now)")

#         while True:
#             try:
#                 chunk = await self._audio_queue.get()
#             except asyncio.CancelledError:
#                 break

#             # Sentinel: Stop
#             if chunk is None:
#                 if buffer and has_speech:
#                     text = await self._transcribe_async(bytes(buffer))
#                     if text: yield STTOutputEvent.create(text)
#                 break

#             buffer.extend(chunk)

#             # --- VAD ---
#             rms = self._calculate_rms(chunk)
#             is_silent = rms < self.silence_threshold
            
#             # Uncomment for debugging volume levels
#             # logger.info(f"[DEBUG] RMS: {rms:.2f}")

#             current_time = asyncio.get_running_loop().time()

#             if not is_silent:
#                 has_speech = True
#                 silence_start_time = None 
#             else:
#                 if silence_start_time is None:
#                     silence_start_time = current_time 
            
#             # --- TRIGGER LOGIC ---
#             buffer_duration = len(buffer) / self.bytes_per_second
#             silence_duration = (current_time - silence_start_time) if silence_start_time else 0.0

#             should_transcribe = False
            
#             # 1. Speech finished (Speech detected + Silence detected + Min Duration met)
#             # FIX: Used self.min_audio_duration instead of undefined variable
#             if (buffer_duration >= self.min_audio_duration and 
#                 has_speech and 
#                 silence_duration >= self.min_silence_seconds):
#                 should_transcribe = True
                
#             # 2. Buffer overflow (Safety valve)
#             # FIX: Used self.max_buffer_duration
#             elif buffer_duration >= self.max_buffer_duration:
#                 should_transcribe = True

#             if should_transcribe:
#                 # If the buffer is strictly silence (overflow triggered), drop it
#                 if not has_speech:
#                     buffer.clear()
#                     silence_start_time = current_time
#                     continue

#                 pcm_data = bytes(buffer)
#                 buffer.clear()
#                 has_speech = False
#                 silence_start_time = None 

#                 text = await self._transcribe_async(pcm_data)
#                 if text:
#                     yield STTOutputEvent.create(text)

#     async def send_audio(self, audio_chunk: bytes) -> None:
#         await self._audio_queue.put(audio_chunk)

#     async def close(self) -> None:
#         self._close_signal.set()
#         await self._audio_queue.put(None)

#     # -------------------------------------------------------------------------
#     # Helpers
#     # -------------------------------------------------------------------------

#     def _calculate_rms(self, audio_chunk: bytes) -> float:
#         if not audio_chunk: return 0.0
#         samples = np.frombuffer(audio_chunk, dtype=np.int16)
#         if samples.size == 0: return 0.0
#         sq = samples.astype(np.float32) ** 2
#         return float(np.sqrt(np.mean(sq)))

#     async def _transcribe_async(self, pcm_bytes: bytes) -> str:
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(None, self._transcribe_blocking, pcm_bytes)

#     def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
#         audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

#         try:
#             segments, _ = self._model.transcribe(
#                 audio,
#                 language="en",
#                 beam_size=1, 
#                 without_timestamps=True,
#                 condition_on_previous_text=False,
#                 initial_prompt="Use English. ", 
#                 vad_filter=True, 
#                 vad_parameters=dict(min_silence_duration_ms=500)
#             )

#             texts = [s.text.strip() for s in segments if s.text]
#             raw_text = " ".join(texts).strip()
            
#             return self._filter_hallucinations(raw_text)
#         except Exception as e:
#             logger.error(f"Transcribe Error: {e}")
#             return ""

#     def _filter_hallucinations(self, text: str) -> str:
#         if not text: return ""
#         clean = text.strip()
        
#         blocklist = {
#             "you", "You.", "you.", "You", 
#             "No, no, no, no, no.", "No, no, no.", "no no no",
#             "Thank you.", "Thanks.", 
#             "MBC News", "Amara.org", "Subtitle by"
#         }
        
#         if clean in blocklist:
#             return ""

#         if re.search(r'\b(\w+)( \1){2,}', clean, re.IGNORECASE):
#             return ""

#         return clean


import asyncio
import logging
import re
import numpy as np
from typing import AsyncIterator, Optional
from faster_whisper import WhisperModel
from events import STTEvent, STTOutputEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LocalWhisperSTT")

class LocalWhisperSTT:
    def __init__(
        self,
        model_size: str = "large-v3", 
        sample_rate: int = 16000,
        device: str = "cuda",         
        compute_type: str = "float16",
        silence_threshold: float = 100.0,  # Adjusted for sensitivity
        min_silence_duration: float = 0.6, # Wait 0.6s of silence before finalizing
        min_audio_duration: float = 0.5,   # Don't transcribe bursts shorter than 0.5s
        max_buffer_duration: float = 8.0,  # Force transcribe after 8s
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.device = device
        self.compute_type = compute_type
        
        # Save these settings (Fixes the NameError)
        self.min_audio_duration = min_audio_duration
        self.max_buffer_duration = max_buffer_duration
        self.min_silence_seconds = min_silence_duration

        # Calculate bytes per second (16-bit mono = 2 bytes per sample)
        self.bytes_per_second = sample_rate * 2 
        
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._close_signal = asyncio.Event()

        logger.info(f"Loading Whisper model '{model_size}' on {self.device}...")
        self._model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Whisper model loaded.")

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        """
        Drop-in replacement for AssemblyAI.receive_events()
        Instead of reading from a WebSocket, we read from the local audio queue.
        """
        buffer = bytearray()
        
        # State tracking for VAD
        has_speech = False 
        silence_start_time = None
        
        logger.info("Local STT listening...")

        while not self._close_signal.is_set():
            try:
                # Wait for next audio chunk
                chunk = await self._audio_queue.get()
            except asyncio.CancelledError:
                break

            # Sentinel: None means stream is finishing
            if chunk is None:
                # Flush remaining audio if there was speech in it
                if buffer and has_speech:
                    text = await self._transcribe_async(bytes(buffer))
                    if text: yield STTOutputEvent.create(text)
                break

            # 1. Add chunk to buffer
            buffer.extend(chunk)

            # 2. Analyze Audio Energy (VAD)
            rms = self._calculate_rms(chunk)
            is_silent = rms < self.silence_threshold
            
            # Debugging (Optional: Uncomment to tune sensitivity)
            # logger.info(f"RMS: {rms}")

            current_time = asyncio.get_running_loop().time()

            if not is_silent:
                # Speech Detected
                has_speech = True
                silence_start_time = None # Reset silence timer
            else:
                # Silence Detected
                if silence_start_time is None:
                    silence_start_time = current_time # Start silence timer
            
            # 3. Decision Logic: Should we transcribe now?
            buffer_duration = len(buffer) / self.bytes_per_second
            silence_duration = (current_time - silence_start_time) if silence_start_time else 0.0

            should_transcribe = False
            
            # Condition A: Normal Sentence End
            # We have audio, we heard speech, and we've had enough silence after the speech
            if (buffer_duration >= self.min_audio_duration and 
                has_speech and 
                silence_duration >= self.min_silence_seconds):
                should_transcribe = True
                
            # Condition B: Buffer Overflow (Safety Valve)
            # User has been talking too long without pausing
            elif buffer_duration >= self.max_buffer_duration:
                should_transcribe = True

            # 4. Transcribe Execution
            if should_transcribe:
                # Edge Case: If buffer is full but it was ALL silence, drop it.
                if not has_speech:
                    buffer.clear()
                    silence_start_time = current_time
                    continue

                # Prepare audio data
                pcm_data = bytes(buffer)
                
                # Reset state immediately so we can capture next phrase
                buffer.clear()
                has_speech = False
                silence_start_time = None 

                # Run Inference (Non-blocking)
                text = await self._transcribe_async(pcm_data)
                
                # Yield the event (This matches AssemblyAI's OutputEvent)
                if text:
                    yield STTOutputEvent.create(text)

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Queue audio for processing."""
        await self._audio_queue.put(audio_chunk)

    async def close(self) -> None:
        """Stop processing."""
        self._close_signal.set()
        await self._audio_queue.put(None)

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _calculate_rms(self, audio_chunk: bytes) -> float:
        if not audio_chunk: return 0.0
        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        if samples.size == 0: return 0.0
        # Cast to float32 to prevent overflow
        sq = samples.astype(np.float32) ** 2
        return float(np.sqrt(np.mean(sq)))

    async def _transcribe_async(self, pcm_bytes: bytes) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_blocking, pcm_bytes)

    def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
        # Normalize to float32
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=1, # Greedy search (Fastest)
                without_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt="Use English. ", 
                vad_filter=True, # Internal VAD as second line of defense
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            texts = [s.text.strip() for s in segments if s.text]
            raw_text = " ".join(texts).strip()
            
            return self._filter_hallucinations(raw_text)
        except Exception as e:
            logger.error(f"Transcribe Error: {e}")
            return ""

    def _filter_hallucinations(self, text: str) -> str:
        if not text: return ""
        clean = text.strip()
        
        # Block common Whisper hallucinations
        blocklist = {
            "you", "You.", "you.", "You", 
            "No, no, no, no, no.", "No, no, no.", "no no no",
            "Thank you.", "Thanks.", 
            "MBC News", "Amara.org", "Subtitle by"
        }
        
        if clean in blocklist:
            return ""

        # Block repeats "word word word word"
        if re.search(r'\b(\w+)( \1){2,}', clean, re.IGNORECASE):
            return ""

        return clean