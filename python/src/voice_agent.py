from time import sleep
from typing import Any, Iterator
from typing_extensions import AsyncIterator
from langchain_core.runnables import RunnableGenerator
from langchain_core.messages import AIMessage
from langchain.agents import create_agent


# this ideally should be coming from a websocket or something else
# that is streaming data
async def _input_stream(input: AsyncIterator[Any]) -> AsyncIterator[str]:
    async for token in input:
        yield token
        sleep(1)


# this is a simple buffer that emits a string when the buffer reaches max size of 2
# this should be replaced with some more meaningful VAD buffer
async def _buffer_stream(input: AsyncIterator[str]) -> AsyncIterator[str]:
    buffer = []
    async for token in input:
        buffer.append(token)
        if len(buffer) >= 2:
            yield "".join(buffer)
            buffer = []

    # Emit any remaining tokens in buffer
    if buffer:
        # Flatten buffer if it contains lists
        flattened = []
        for item in buffer:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        yield "".join(flattened)


# this is where we would call openai/11labs/etc. to transcribe the stream
# (imagine the input is an audio buffer)
async def _transcribe_stream(input: AsyncIterator[str]) -> AsyncIterator[dict]:
    async for token in input:
        print(f"got token {token}")
        yield {"messages": [AIMessage(content=token)]}


# this is where we would call openai/11labs/etc. to generate text to speech
async def _tts_stream(input: str) -> AsyncIterator[str]:
    print(f"got input {input}")
    yield "hello"


agent = create_agent(
    model="openai:gpt-5", system_prompt="You are a helpful audio assistant"
)

audio_stream = (
    RunnableGenerator(_input_stream)
    | RunnableGenerator(_buffer_stream)
    | RunnableGenerator(_transcribe_stream)
    | agent
    | RunnableGenerator(_tts_stream)
)

stream_instance = audio_stream.astream(["hey", " there", " delilah"])


async def main():
    async for token in stream_instance:
        print("output: ", token)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
