# import asyncio
# import numpy as np
# import threading
# from typing import AsyncIterator, Optional, List
# from concurrent.futures import ThreadPoolExecutor

# # Assuming TTSChunkEvent is imported from your events module
# # If not, here is a dummy definition for context:
# try:
#     from events import TTSChunkEvent
# except ImportError:
#     class TTSChunkEvent:
#         @staticmethod
#         def create(audio_chunk: bytes):
#             return type("Event", (), {"audio": audio_chunk})()

# # Import your local VibeVoice class (assuming it's in a file named vibe_voice.py)
# from demo.vibevoice_tts_main import StreamingTTSService

# class VibeVoiceAsyncTTS:
#     def __init__(
#         self,
#         model_path: str,
#         device: str = "cuda",
#         voice_preset: str = None,  # Corresponds to voice_id
#         inference_steps: int = 5,
#         output_format: str = "pcm_24000", # Used for metadata, VibeVoice is usually 24k
#     ):
#         """
#         An Async wrapper for VibeVoiceTTS to match the ElevenLabsTTS interface.
#         """
#         self.model_path = model_path
#         self.device = device
#         self.voice_key = voice_preset
#         self.inference_steps = inference_steps
        
#         # Initialize the synchronous VibeVoice service
#         print(f"[VibeVoiceAsync] Initializing model from {model_path}...")
#         self.service = StreamingTTSService(
#             model_path=model_path,
#             device=device,
#             inference_steps=inference_steps
#         )
#         self.service.load()
        
#         # Ensure sample_rate exists (Fixing the issue from previous turn)
#         if not hasattr(self.service, 'sample_rate'):
#             # Default to 24k or try to fetch from processor if available
#             self.service.sample_rate = 24000 
            
#         print("[VibeVoiceAsync] Model loaded.")

#         # Queues for communicating between Async World and Sync Thread
#         self._input_queue = asyncio.Queue()
#         self._output_queue = asyncio.Queue()
        
#         # Control flags
#         self._stop_event = threading.Event()
#         self._processing_task: Optional[asyncio.Task] = None
        
#         # Start the background worker
#         self._processing_task = asyncio.create_task(self._generation_worker())

#     async def send_text(self, text: Optional[str]) -> None:
#         """
#         Accepts text and queues it for the background worker.
#         Matches ElevenLabsTTS.send_text signature.
#         """
#         if text is None:
#             return

#         # ElevenLabs sends empty strings or specific flush commands.
#         # We filter for actual content.
#         if not text.strip():
#             return

#         # Put text into queue for the worker to pick up
#         await self._input_queue.put(text)

#     async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
#         """
#         Yields audio chunks as they are generated.
#         Matches ElevenLabsTTS.receive_events signature.
#         """
#         try:
#             while True:
#                 # Get the next chunk from the output queue
#                 # We wait for either a chunk or a cancellation
#                 chunk = await self._output_queue.get()
                
#                 if chunk is None: # Sentinel value indicating stream end/close
#                     break
                
#                 # Check if it's an error passed from the worker
#                 if isinstance(chunk, Exception):
#                     print(f"[VibeVoiceAsync] Generation Error: {chunk}")
#                     continue

#                 yield TTSChunkEvent.create(chunk)
#                 self._output_queue.task_done()
                
#         except asyncio.CancelledError:
#             pass

#     async def close(self) -> None:
#         """
#         Stops the worker and cleans up resources.
#         """
#         self._stop_event.set()
        
#         # Signal worker to stop accepting new input
#         await self._input_queue.put(None)
        
#         if self._processing_task:
#             await self._processing_task
            
#         print("[VibeVoiceAsync] Closed.")

#     async def _generation_worker(self):
#         """
#         Runs in the background, pulling text from input_queue,
#         running the blocking VibeVoice stream, and putting bytes into output_queue.
#         """
#         loop = asyncio.get_running_loop()
#         executor = ThreadPoolExecutor(max_workers=1)

#         while not self._stop_event.is_set():
#             # Wait for text input
#             text = await self._input_queue.get()
            
#             if text is None: # Shutdown signal
#                 break

#             # Run the blocking Torch generation in a separate thread
#             # so we don't block the AsyncIO event loop
#             await loop.run_in_executor(
#                 executor, 
#                 self._run_sync_stream, 
#                 text, 
#                 loop
#             )
            
#             self._input_queue.task_done()

#         # Send None to output queue to break the receive_events loop
#         await self._output_queue.put(None)
#         executor.shutdown(wait=False)

#     def _run_sync_stream(self, text: str, loop: asyncio.AbstractEventLoop):
#         """
#         Executed inside a ThreadPool. Runs the synchronous VibeVoice stream
#         and thread-safely puts results back into the async queue.
#         """
#         try:
#             # Call the synchronous stream method from VibeVoiceTTS
#             stream_iterator = self.service.stream(
#                 text=text,
#                 voice_key=self.voice_key,
#                 inference_steps=self.inference_steps,
#                 temperature=0.7, # Default params
#                 cfg_scale=1.5
#             )

#             for chunk_numpy in stream_iterator:
#                 if self._stop_event.is_set():
#                     break
                
#                 # Convert Numpy Array -> PCM16 Bytes
#                 # Using the helper method from your VibeVoiceTTS class
#                 pcm_bytes = self.service.chunk_to_pcm16(chunk_numpy)
                
#                 # We are in a thread, so we must use call_soon_threadsafe 
#                 # to put data into the asyncio queue
#                 loop.call_soon_threadsafe(
#                     self._output_queue.put_nowait, 
#                     pcm_bytes
#                 )

#         except Exception as e:
#             # Pass errors back to main loop
#             loop.call_soon_threadsafe(
#                 self._output_queue.put_nowait, 
#                 e
#             )



import asyncio
import os
import threading
import logging
from typing import AsyncIterator, Optional
from concurrent.futures import ThreadPoolExecutor

# Import Hugging Face Hub for auto-downloading
from huggingface_hub import snapshot_download

# Assuming TTSChunkEvent is imported from your events module
try:
    from events import TTSChunkEvent
except ImportError:
    class TTSChunkEvent:
        @staticmethod
        def create(audio_chunk: bytes):
            return type("Event", (), {"audio": audio_chunk})()

# Import your local VibeVoice class
from demo.vibevoice_tts_main import StreamingTTSService

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoiceAsync")

class VibeVoiceAsyncTTS:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        voice_preset: str = None, 
        inference_steps: int = 5,
        hf_repo_id: str = "microsoft/VibeVoice-Realtime-0.5B", # Default official repo
    ):
        """
        Async wrapper for VibeVoiceTTS.
        Auto-downloads model/tokenizer if model_path does not exist.
        """
        self.model_path = model_path
        self.device = device
        self.voice_key = voice_preset
        self.inference_steps = inference_steps
        
        # 1. Check and Download Model + Tokenizer if missing
        self._ensure_model_downloaded(model_path, hf_repo_id)
        
        # 2. Initialize the synchronous VibeVoice service
        logger.info(f"Initializing VibeVoice service from {model_path}...")
        try:
            self.service = StreamingTTSService(
                model_path=model_path,
                device=device,
                inference_steps=inference_steps
            )
            self.service.load()
        except Exception as e:
            logger.error(f"Failed to load VibeVoice service: {e}")
            raise e
        
        # 3. Ensure sample_rate exists
        if not hasattr(self.service, 'sample_rate'):
            self.service.sample_rate = 24000 
            
        logger.info("VibeVoice Model loaded successfully.")

        # 4. Setup Async/Sync Bridge
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        self._stop_event = threading.Event()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Start the background worker
        self._processing_task = asyncio.create_task(self._generation_worker())

    def _ensure_model_downloaded(self, local_path: str, repo_id: str):
        """
        Checks if the local directory exists and is populated.
        If not, downloads the VibeVoice model (includes Qwen tokenizer config) from HF.
        """
        # Check if folder exists
        if not os.path.exists(local_path):
            logger.warning(f"Model path '{local_path}' not found. Downloading {repo_id}...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_path,
                    local_dir_use_symlinks=False, # Copy actual files so it persists cleanly
                    ignore_patterns=["*.git*"] # Clean download
                )
                logger.info(f"Download complete. Model saved to {local_path}")
            except Exception as e:
                logger.error(f"Failed to download model from Hugging Face: {e}")
                raise RuntimeError(
                    f"Model not found at {local_path} and download failed. "
                    "Please check your internet connection or HF token."
                ) from e
        else:
            # Optional: Check if the folder is empty
            if not os.listdir(local_path):
                logger.warning(f"Directory '{local_path}' is empty. Downloading {repo_id}...")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_path,
                    local_dir_use_symlinks=False
                )

    async def send_text(self, text: Optional[str]) -> None:
        if text is None: return
        if not text.strip(): return
        await self._input_queue.put(text)

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        try:
            while True:
                chunk = await self._output_queue.get()
                
                if chunk is None: 
                    break
                
                if isinstance(chunk, Exception):
                    logger.error(f"Generation Error: {chunk}")
                    continue

                yield TTSChunkEvent.create(chunk)
                self._output_queue.task_done()
                
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        self._stop_event.set()
        await self._input_queue.put(None)
        if self._processing_task:
            await self._processing_task
        logger.info("Service closed.")

    async def _generation_worker(self):
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        while not self._stop_event.is_set():
            text = await self._input_queue.get()
            if text is None: break

            await loop.run_in_executor(
                executor, 
                self._run_sync_stream, 
                text, 
                loop
            )
            self._input_queue.task_done()

        await self._output_queue.put(None)
        executor.shutdown(wait=False)

    def _run_sync_stream(self, text: str, loop: asyncio.AbstractEventLoop):
        try:
            stream_iterator = self.service.stream(
                text=text,
                voice_key=self.voice_key,
                inference_steps=self.inference_steps,
                temperature=0.7,
                cfg_scale=1.5
            )

            for chunk_numpy in stream_iterator:
                if self._stop_event.is_set(): break
                
                # Convert Numpy -> PCM Bytes
                if hasattr(self.service, 'chunk_to_pcm16'):
                    pcm_bytes = self.service.chunk_to_pcm16(chunk_numpy)
                else:
                    # Fallback if method missing: convert float32 [-1,1] to int16 bytes
                    pcm_bytes = (chunk_numpy * 32767).astype(np.int16).tobytes()

                loop.call_soon_threadsafe(
                    self._output_queue.put_nowait, 
                    pcm_bytes
                )

        except Exception as e:
            loop.call_soon_threadsafe(
                self._output_queue.put_nowait, 
                e
            )