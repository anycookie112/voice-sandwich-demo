# """
# Local Whisper Real-Time-ish STT Adapter (Standard PyTorch Version)

# Drop-in replacement for AssemblyAISTT, using the standard OpenAI Whisper model.
# Fixes compatibility with non-AVX/Specific GPU architectures where faster-whisper fails.
# """

# import asyncio
# import contextlib
# from typing import AsyncIterator, Optional

# import numpy as np
# import torch
# import whisper  # pip install openai-whisper

# from events import STTEvent, STTOutputEvent

# class WhisperPytorchSTT:
#     def __init__(
#         self,
#         model_size: str = "base",
#         sample_rate: int = 16000,
#         device: str = "cpu",
#         compute_type: str = "int8",       # Accepted for compatibility, but ignored by standard whisper
#         silence_threshold: float = 500.0,
#         min_silence_chunks: int = 8,
#     ):
#         self.sample_rate = sample_rate
#         self.silence_threshold = silence_threshold
#         self.min_silence_chunks = min_silence_chunks
        
#         # 1. Device Logic: Check if CUDA is actually available
#         if device == "cuda" and not torch.cuda.is_available():
#             print("[WARN] LocalWhisperSTT: CUDA requested but not available. Falling back to CPU.")
#             self.device = "cpu"
#         else:
#             self.device = device

#         # 2. Compute Type Logic: 
#         # Standard Whisper doesn't use "int8"/"float16" strings in the same way faster-whisper does.
#         # It relies on the 'fp16' boolean during transcription.
#         if compute_type:
#             print(f"[INFO] LocalWhisperSTT: 'compute_type={compute_type}' argument ignored (using standard PyTorch precision).")

#         self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
#         self._close_signal = asyncio.Event()

#         print(f"[INFO] Loading Standard Whisper model '{model_size}' on {self.device}...")
#         self._model = whisper.load_model(model_size, device=self.device)
#         print("[INFO] Model loaded.")

#     # -------------------------------------------------------------------------
#     # Public API (Mirrors AssemblyAISTT)
#     # -------------------------------------------------------------------------

#     async def receive_events(self) -> AsyncIterator[STTEvent]:
#         """
#         Consumes audio from the queue, buffers it until silence is detected,
#         then runs inference and yields events.
#         """
#         buffer = bytearray()
#         silence_chunks = 0

#         while True:
#             try:
#                 # Wait for audio data
#                 chunk = await self._audio_queue.get()
#             except asyncio.CancelledError:
#                 break

#             # Sentinel: None means flush + exit
#             if chunk is None:
#                 if buffer:
#                     text = await self._transcribe_async(bytes(buffer))
#                     if text.strip():
#                         yield STTOutputEvent.create(text)
#                 break

#             # Add data to buffer
#             buffer.extend(chunk)

#             # Simple VAD (Voice Activity Detection)
#             if self._is_silence(chunk):
#                 silence_chunks += 1
#             else:
#                 silence_chunks = 0

#             # If we have enough audio and hit a silence gap, transcribe what we have
#             if silence_chunks >= self.min_silence_chunks and len(buffer) > 0:
#                 pcm = bytes(buffer)
#                 buffer.clear()
#                 silence_chunks = 0

#                 text = await self._transcribe_async(pcm)
#                 if text.strip():
#                     yield STTOutputEvent.create(text)

#     async def send_audio(self, audio_chunk: bytes) -> None:
#         """
#         Input point for the stream. Puts audio into the processing queue.
#         """
#         await self._audio_queue.put(audio_chunk)

#     async def close(self) -> None:
#         """
#         Signals the loop to finish processing and exit.
#         """
#         self._close_signal.set()
#         # Put sentinel to unblock queue consumer
#         with contextlib.suppress(asyncio.QueueFull):
#             await self._audio_queue.put(None)

#     # -------------------------------------------------------------------------
#     # Internal Processing
#     # -------------------------------------------------------------------------

#     def _is_silence(self, audio_chunk: bytes) -> bool:
#         """
#         Simple RMS energy check. 
#         """
#         if not audio_chunk:
#             return True

#         # Convert bytes to numpy to calculate energy
#         samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
#         if samples.size == 0:
#             return True

#         rms = np.sqrt(np.mean(samples**2))
#         return rms < self.silence_threshold

#     async def _transcribe_async(self, pcm_bytes: bytes) -> str:
#         """
#         Offload the heavy model inference to a separate thread so asyncio isn't blocked.
#         """
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(None, self._transcribe_blocking, pcm_bytes)

#     def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
#         """
#         Standard OpenAI Whisper inference logic.
#         """
#         if not pcm_bytes:
#             return ""

#         # 1. Normalize PCM 16-bit to Float32 [-1, 1]
#         audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

#         # 2. Determine precision settings
#         # CPU generally crashes with fp16=True in PyTorch unless specifically supported
#         use_fp16 = (self.device == "cuda")

#         try:
#             # 3. Transcribe
#             result = self._model.transcribe(
#                 audio_np,
#                 language="en",
#                 fp16=use_fp16,
#                 beam_size=5,             # Standard beam size
#                 no_speech_threshold=0.6, # Helps detect silence vs speech
#                 condition_on_previous_text=False # Prevents looping hallucinations
#             )
            
#             raw_text = result.get("text", "").strip()
#             return self._filter_hallucinations(raw_text)

#         except Exception as e:
#             print(f"[ERROR] LocalWhisperSTT Transcription failed: {e}")
#             return ""

#     def _filter_hallucinations(self, text: str) -> str:
#         """
#         Basic cleanup for common Whisper artifacts.
#         """
#         hallucinations = {
#             "you", "you.", "You", "You.",
#             "Thank you.", "Thank you", "Thanks.",
#             "MBC News", "MBC", "Bye.", "Bye",
#             "I'm sorry.", "I'm sorry",
#         }
        
#         cleaned = text.strip()
        
#         # 1. Exact match removal
#         if cleaned in hallucinations:
#             return ""

#         # 2. Repetitive pattern removal (e.g., "you you you you")
#         # Simple heuristic: if the string is just one word repeated many times
#         clean_lower = cleaned.lower().replace(".", "").replace("!", "").replace("?", "")
#         words = clean_lower.split()
#         if len(words) > 3 and all(w == words[0] for w in words):
#             return ""

#         return cleaned


# """
# Local Whisper Real-Time-ish STT Adapter (Multilingual / Chinese Support)

# Drop-in replacement for AssemblyAISTT using OpenAI Whisper.
# Supports Chinese and other languages by removing English-only constraints.
# """

# import asyncio
# import contextlib
# import re
# from typing import AsyncIterator, Optional, Any

# import numpy as np
# import torch
# import whisper

# from events import STTEvent, STTOutputEvent

# class WhisperPytorchSTT:
#     def __init__(
#         self,
#         # AssemblyAI compatibility args
#         api_key: Optional[str] = None,
#         format_turns: bool = True,
        
#         # Whisper Settings
#         model_size: str = "base",    # Use "large-v3" for best Chinese accuracy
#         sample_rate: int = 16000,
#         device: str = "cuda",
#         language: str = None,        # <--- NEW: Set to "zh" for Chinese, or None to auto-detect
        
#         # VAD / Sensitivity Settings
#         silence_threshold: float = 2000.0, 
#         min_silence_chunks: int = 4,       
#         logprob_threshold: float = -1.0,   
#         no_speech_threshold: float = 0.6,
        
#         **kwargs: Any,
#     ):
#         self.sample_rate = sample_rate
#         self.silence_threshold = silence_threshold
#         self.min_silence_chunks = min_silence_chunks
#         self.logprob_threshold = logprob_threshold
#         self.no_speech_threshold = no_speech_threshold
#         self.language = language

#         # Device Logic
#         if device == "cuda" and not torch.cuda.is_available():
#             print("[WARN] CUDA requested but not available. Falling back to CPU.")
#             self.device = "cpu"
#         else:
#             self.device = device

#         self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
#         self._close_signal = asyncio.Event()

#         print(f"[INFO] Loading Whisper '{model_size}' on {self.device}...")
#         try:
#             self._model = whisper.load_model(model_size, device=self.device)
#             print(f"[INFO] Model loaded. Language set to: {self.language if self.language else 'Auto-detect'}")
#         except Exception as e:
#             print(f"[ERROR] Failed to load model: {e}")
#             raise e

#     # -------------------------------------------------------------------------
#     # Public API
#     # -------------------------------------------------------------------------

#     async def receive_events(self) -> AsyncIterator[STTEvent]:
#         buffer = bytearray()
#         silence_counter = 0

#         while True:
#             try:
#                 if self._close_signal.is_set() and self._audio_queue.empty():
#                     break
#                 chunk = await self._audio_queue.get()
#             except asyncio.CancelledError:
#                 break

#             if chunk is None:
#                 if buffer:
#                     text = await self._transcribe_async(bytes(buffer))
#                     if text: yield STTOutputEvent.create(text)
#                 break

#             buffer.extend(chunk)

#             # Energy-based VAD
#             if self._is_silence(chunk):
#                 silence_counter += 1
#             else:
#                 silence_counter = 0

#             if silence_counter >= self.min_silence_chunks and len(buffer) > 0:
#                 pcm = bytes(buffer)
#                 buffer.clear()
#                 silence_counter = 0

#                 text = await self._transcribe_async(pcm)
#                 if text:
#                     yield STTOutputEvent.create(text)

#     async def send_audio(self, audio_chunk: bytes) -> None:
#         if not self._close_signal.is_set():
#             await self._audio_queue.put(audio_chunk)

#     async def close(self) -> None:
#         self._close_signal.set()
#         with contextlib.suppress(asyncio.QueueFull):
#             await self._audio_queue.put(None)

#     # -------------------------------------------------------------------------
#     # Internal Logic
#     # -------------------------------------------------------------------------

#     async def _ensure_connection(self): pass

#     def _is_silence(self, audio_chunk: bytes) -> bool:
#         if not audio_chunk: return True
#         samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
#         if samples.size == 0: return True
        
#         rms = np.sqrt(np.mean(samples**2))
#         return rms < self.silence_threshold

#     async def _transcribe_async(self, pcm_bytes: bytes) -> str:
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(None, self._transcribe_blocking, pcm_bytes)

#     def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
#         if not pcm_bytes: return ""

#         audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#         use_fp16 = (self.device == "cuda")

#         try:
#             result = self._model.transcribe(
#                 audio_np,
#                 language=self.language,  # <--- CHANGED: Uses self.language ("zh" or None)
#                 fp16=use_fp16,
#                 beam_size=5,
#                 no_speech_threshold=self.no_speech_threshold, 
#                 logprob_threshold=self.logprob_threshold,
#                 condition_on_previous_text=False
#             )
            
#             raw_text = result.get("text", "").strip()
#             return self._filter_hallucinations(raw_text)

#         except Exception as e:
#             print(f"[ERROR] Transcribe error: {e}")
#             return ""

#     def _filter_hallucinations(self, text: str) -> str:
#         """
#         Multilingual hallucination filtering.
#         """
#         cleaned = text.strip()
        
#         # 1. Remove bracketed noise markers like [silence], (music)
#         cleaned = re.sub(r'\[.*?\]', '', cleaned)
#         cleaned = re.sub(r'\(.*?\)', '', cleaned)
#         cleaned = cleaned.strip()

#         if not cleaned:
#             return ""

#         # 2. Filter known Hallucinations (English & Chinese)
#         # Note: Whisper sometimes outputs these specific Chinese phrases during silence
#         hallucinations = {
#             "you", "you.", "thank you.", "thanks.", 
#             "mbc news", "bye.", "bye", 
#             "字幕", "字幕 by", "谢谢", "谢谢观看", "不客气"
#         }
        
#         if cleaned.lower() in hallucinations:
#             return ""

#         # 3. Filter repetitive loops
#         if len(cleaned) > 20:
#              mid = len(cleaned) // 2
#              if cleaned[:mid] == cleaned[mid:]:
#                  return ""

#         # 4. REMOVED the [a-zA-Z] check. 
#         # Instead, just ensure there is *some* content that isn't just punctuation.
#         # This regex matches any word character in any language (including Chinese Kanji/Hanzi)
#         if not re.search(r'\w', cleaned): 
#             return ""

#         return cleaned


"""
Local Whisper Real-Time-ish STT Adapter (Hallucination Fix)

Drop-in replacement for AssemblyAISTT using OpenAI Whisper.
Fixes the "We'll be right back" loop and other silence hallucinations.
"""

import asyncio
import contextlib
import re
from typing import AsyncIterator, Optional, Any

import numpy as np
import torch
import whisper

from events import STTEvent, STTOutputEvent

class WhisperPytorchSTT:
    def __init__(
        self,
        # AssemblyAI compatibility args
        api_key: Optional[str] = None,
        format_turns: bool = True,
        
        # Whisper Settings
        model_size: str = "base",
        sample_rate: int = 16000,
        device: str = "cuda",
        language: str = None, 
        
        # VAD / Sensitivity Settings
        silence_threshold: float = 2000.0, 
        min_silence_chunks: int = 4,       
        logprob_threshold: float = -1.0,   
        no_speech_threshold: float = 0.4,  # <-- LOWERED: More aggressive at detecting silence (default 0.6)
        
        **kwargs: Any,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_silence_chunks = min_silence_chunks
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.language = language

        # State for deduplication
        self._last_text = ""

        # Device Logic
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._close_signal = asyncio.Event()

        print(f"[INFO] Loading Whisper '{model_size}' on {self.device}...")
        try:
            self._model = whisper.load_model(model_size, device=self.device)
            print(f"[INFO] Model loaded. Language: {self.language if self.language else 'Auto'}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise e

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        buffer = bytearray()
        silence_counter = 0

        while True:
            try:
                if self._close_signal.is_set() and self._audio_queue.empty():
                    break
                chunk = await self._audio_queue.get()
            except asyncio.CancelledError:
                break

            if chunk is None:
                if buffer:
                    text = await self._transcribe_async(bytes(buffer))
                    if self._is_valid_output(text):
                        yield STTOutputEvent.create(text)
                break

            buffer.extend(chunk)

            # Energy-based VAD
            if self._is_silence(chunk):
                silence_counter += 1
            else:
                silence_counter = 0

            # Process buffer when silence is detected
            if silence_counter >= self.min_silence_chunks and len(buffer) > 0:
                pcm = bytes(buffer)
                buffer.clear()
                silence_counter = 0

                text = await self._transcribe_async(pcm)
                
                # Yield only if valid and not a duplicate
                if self._is_valid_output(text):
                    yield STTOutputEvent.create(text)

    async def send_audio(self, audio_chunk: bytes) -> None:
        if not self._close_signal.is_set():
            await self._audio_queue.put(audio_chunk)

    async def close(self) -> None:
        self._close_signal.set()
        with contextlib.suppress(asyncio.QueueFull):
            await self._audio_queue.put(None)

    # -------------------------------------------------------------------------
    # Internal Logic
    # -------------------------------------------------------------------------

    async def _ensure_connection(self): pass

    def _is_silence(self, audio_chunk: bytes) -> bool:
        if not audio_chunk: return True
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        if samples.size == 0: return True
        
        rms = np.sqrt(np.mean(samples**2))
        return rms < self.silence_threshold

    async def _transcribe_async(self, pcm_bytes: bytes) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_blocking, pcm_bytes)

    def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
        if not pcm_bytes: return ""

        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        use_fp16 = (self.device == "cuda")

        try:
            result = self._model.transcribe(
                audio_np,
                language=self.language,
                fp16=use_fp16,
                beam_size=5,
                no_speech_threshold=self.no_speech_threshold, 
                logprob_threshold=self.logprob_threshold,
                condition_on_previous_text=False
            )
            return result.get("text", "").strip()

        except Exception as e:
            print(f"[ERROR] Transcribe error: {e}")
            return ""

    def _is_valid_output(self, text: str) -> bool:
        """
        Validates text against hallucinations and duplicates.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        # 1. Check Blocklist (Case insensitive)
        # These are standard hallucinations from the training data
        blocklist = {
            "we'll be right back", 
            "we'll be right back.",
            "you", "you.", "thank you.", "thanks.", 
            "mbc news", "bye.", "bye", 
            "字幕", "字幕 by", "谢谢", "谢谢观看", "不客气",
            "copyright", "all rights reserved",
            "uncaptioned"
        }
        
        # Remove punctuation for the check to catch "We'll be right back!"
        check_str = re.sub(r'[^\w\s]', '', cleaned).lower()
        if check_str in blocklist or cleaned.lower() in blocklist:
            return False

        # 2. Check for bracketed noise like [silence]
        if re.match(r'^\[.*?\]$', cleaned) or re.match(r'^\(.*?\)$', cleaned):
            return False

        # 3. Deduplication (Prevent same phrase looping)
        # If the exact same text comes out twice in a row, block the second one.
        if cleaned == self._last_text:
            return False
            
        self._last_text = cleaned
        return True