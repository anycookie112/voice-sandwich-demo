# # # test_kokoro_wav.py
# # import asyncio
# # from kokoro_tts import KokoroTTS  # adjust import path

# # async def main():
# #     tts = KokoroTTS(
# #         lang_code="a",
# #         voice="af_heart",
# #         sample_rate=16000,
# #         chunk_ms=50,
# #     )

# #     await tts.synthesize_to_wav(
# #         "Hello, this is Kokoro speaking from your local TTS adapter.",
# #         "kokoro_test.wav",
# #     )

# # asyncio.run(main())
# #    python -m ensurepip --default-pip



# import asyncio
# import wave

# from whisper_stt import LocalWhisperSTT
# from events import STTOutputEvent  # just to type-check / compare

# INPUT_WAV = "kokoro_test.wav"  # 16kHz, mono, 16-bit PCM

# async def main():
#     stt = LocalWhisperSTT(
#         model_size="small.en",
#         sample_rate=16000,
#         device="cuda",        # force GPU
#         compute_type="default"  # or "float32" if you want to be explicit
#     )


#     # 1. Read a WAV file and push its frames as chunks
#     with wave.open(INPUT_WAV, "rb") as f:
#         assert f.getnchannels() == 1, "WAV must be mono"
#         assert f.getframerate() == 16000, "WAV must be 16kHz"
#         assert f.getsampwidth() == 2, "WAV must be 16-bit"

#         while True:
#             data = f.readframes(320)  # ~20ms at 16kHz
#             if not data:
#                 break
#             await stt.send_audio(data)

#     # 2. Close stream to flush and end receive_events
#     await stt.close()

#     # 3. Collect transcripts
#     async for event in stt.receive_events():
#         print("Got STT event:", event)
#         # If you want just text:
#         if isinstance(event, STTOutputEvent):
#             print("Final transcript:", event.text)

# asyncio.run(main())




# import asyncio
# import wave
# from pathlib import Path

# from whisper_stt import LocalWhisperSTT   # <-- adjust if filename is different
# from events import STTOutputEvent, STTChunkEvent  # just for isinstance checks

# INPUT_WAV = "kokoro_test.wav"  # file written by Kokoro test


# async def main():
#     wav_path = Path(INPUT_WAV)
#     if not wav_path.exists():
#         print(f"[ERROR] WAV file not found: {wav_path.resolve()}")
#         return

#     print(f"[INFO] Using WAV file: {wav_path.resolve()}")

#     # Open WAV and inspect format
#     with wave.open(str(wav_path), "rb") as f:
#         channels = f.getnchannels()
#         sr = f.getframerate()
#         width = f.getsampwidth()
#         nframes = f.getnframes()

#         print(f"[INFO] WAV channels: {channels}")
#         print(f"[INFO] WAV sample rate: {sr}")
#         print(f"[INFO] WAV sample width: {width} bytes")
#         print(f"[INFO] WAV frames: {nframes}")

#         if channels != 1:
#             print("[ERROR] WAV must be mono (1 channel)")
#             return
#         if width != 2:
#             print("[ERROR] WAV must be 16-bit (2 bytes per sample)")
#             return

#         # Create STT with audio sample rate from file (Whisper will resample to 16k internally)
#         stt = LocalWhisperSTT(
#             model_size="large-v3",
#             sample_rate=sr,          # <= IMPORTANT: use the WAV's SR (likely 24000)
#             device="cpu",           # or "cpu" if you want CPU
#             compute_type="float32",  # safe
#             silence_threshold=50.0,  # make VAD more permissive
#             min_silence_chunks=3,    # detect utterance quickly
#         )

#         print("[INFO] Feeding audio chunks to LocalWhisperSTT...")

#         # Read the whole WAV as chunks (~20ms per chunk at given SR)
#         chunk_size_frames = int(sr * 0.02)  # ~20ms
#         while True:
#             data = f.readframes(chunk_size_frames)
#             if not data:
#                 break
#             await stt.send_audio(data)

#         print("[INFO] Done sending audio. Closing STT stream...")
#         await stt.close()

#         print("[INFO] Collecting STT events:")
#         got_any = False

#         async for event in stt.receive_events():
#             got_any = True
#             print(f"[DEBUG] Got STT event: {event}")

#             if isinstance(event, STTOutputEvent):
#                 print(f"[RESULT] Final transcript: {event.transcript!r}")
#             elif isinstance(event, STTChunkEvent):
#                 print(f"[PARTIAL] Chunk: {event.transcript!r}")


#         if not got_any:
#             print("[WARN] No STT events were produced at all.")


# if __name__ == "__main__":
#     asyncio.run(main())
from vibevoice_tts import VibeVoiceAsyncTTS

services = VibeVoiceAsyncTTS(model_path = "/home/robust/models/VibeVoice-Realtime-0.5B")
# services.test()