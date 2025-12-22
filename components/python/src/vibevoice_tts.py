import asyncio
import numpy as np
import threading
from typing import AsyncIterator, Optional, List
from concurrent.futures import ThreadPoolExecutor

# Assuming TTSChunkEvent is imported from your events module
# If not, here is a dummy definition for context:
try:
    from events import TTSChunkEvent
except ImportError:
    class TTSChunkEvent:
        @staticmethod
        def create(audio_chunk: bytes):
            return type("Event", (), {"audio": audio_chunk})()

# Import your local VibeVoice class (assuming it's in a file named vibe_voice.py)
from demo.vibevoice_tts_main import StreamingTTSService

class VibeVoiceAsyncTTS:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        voice_preset: str = None,  # Corresponds to voice_id
        inference_steps: int = 5,
        output_format: str = "pcm_24000", # Used for metadata, VibeVoice is usually 24k
    ):
        """
        An Async wrapper for VibeVoiceTTS to match the ElevenLabsTTS interface.
        """
        self.model_path = model_path
        self.device = device
        self.voice_key = voice_preset
        self.inference_steps = inference_steps
        
        # Initialize the synchronous VibeVoice service
        print(f"[VibeVoiceAsync] Initializing model from {model_path}...")
        self.service = StreamingTTSService(
            model_path=model_path,
            device=device,
            inference_steps=inference_steps
        )
        self.service.load()
        
        # Ensure sample_rate exists (Fixing the issue from previous turn)
        if not hasattr(self.service, 'sample_rate'):
            # Default to 24k or try to fetch from processor if available
            self.service.sample_rate = 24000 
            
        print("[VibeVoiceAsync] Model loaded.")

        # Queues for communicating between Async World and Sync Thread
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        
        # Control flags
        self._stop_event = threading.Event()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Start the background worker
        self._processing_task = asyncio.create_task(self._generation_worker())

    async def send_text(self, text: Optional[str]) -> None:
        """
        Accepts text and queues it for the background worker.
        Matches ElevenLabsTTS.send_text signature.
        """
        if text is None:
            return

        # ElevenLabs sends empty strings or specific flush commands.
        # We filter for actual content.
        if not text.strip():
            return

        # Put text into queue for the worker to pick up
        await self._input_queue.put(text)

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        """
        Yields audio chunks as they are generated.
        Matches ElevenLabsTTS.receive_events signature.
        """
        try:
            while True:
                # Get the next chunk from the output queue
                # We wait for either a chunk or a cancellation
                chunk = await self._output_queue.get()
                
                if chunk is None: # Sentinel value indicating stream end/close
                    break
                
                # Check if it's an error passed from the worker
                if isinstance(chunk, Exception):
                    print(f"[VibeVoiceAsync] Generation Error: {chunk}")
                    continue

                yield TTSChunkEvent.create(chunk)
                self._output_queue.task_done()
                
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        """
        Stops the worker and cleans up resources.
        """
        self._stop_event.set()
        
        # Signal worker to stop accepting new input
        await self._input_queue.put(None)
        
        if self._processing_task:
            await self._processing_task
            
        print("[VibeVoiceAsync] Closed.")

    async def _generation_worker(self):
        """
        Runs in the background, pulling text from input_queue,
        running the blocking VibeVoice stream, and putting bytes into output_queue.
        """
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        while not self._stop_event.is_set():
            # Wait for text input
            text = await self._input_queue.get()
            
            if text is None: # Shutdown signal
                break

            # Run the blocking Torch generation in a separate thread
            # so we don't block the AsyncIO event loop
            await loop.run_in_executor(
                executor, 
                self._run_sync_stream, 
                text, 
                loop
            )
            
            self._input_queue.task_done()

        # Send None to output queue to break the receive_events loop
        await self._output_queue.put(None)
        executor.shutdown(wait=False)

    def _run_sync_stream(self, text: str, loop: asyncio.AbstractEventLoop):
        """
        Executed inside a ThreadPool. Runs the synchronous VibeVoice stream
        and thread-safely puts results back into the async queue.
        """
        try:
            # Call the synchronous stream method from VibeVoiceTTS
            stream_iterator = self.service.stream(
                text=text,
                voice_key=self.voice_key,
                inference_steps=self.inference_steps,
                temperature=0.7, # Default params
                cfg_scale=1.5
            )

            for chunk_numpy in stream_iterator:
                if self._stop_event.is_set():
                    break
                
                # Convert Numpy Array -> PCM16 Bytes
                # Using the helper method from your VibeVoiceTTS class
                pcm_bytes = self.service.chunk_to_pcm16(chunk_numpy)
                
                # We are in a thread, so we must use call_soon_threadsafe 
                # to put data into the asyncio queue
                loop.call_soon_threadsafe(
                    self._output_queue.put_nowait, 
                    pcm_bytes
                )

        except Exception as e:
            # Pass errors back to main loop
            loop.call_soon_threadsafe(
                self._output_queue.put_nowait, 
                e
            )