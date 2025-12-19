from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import os

def get_ollama_model():
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0,
        base_url =  "http://192.168.15.199:11434",
        # other params...
    )
    return llm


def get_groq_model(
    api_key: str | None = None,
    model: str = "openai/gpt-oss-20b",
    temperature: float = 0.0,
    max_tokens: int | None = None,
):
    """
    Create a LangChain ChatGroq LLM.

    api_key:
        Groq API key. If None, falls back to GROQ_API_KEY env var.
    model:
        Groq model name (e.g. llama-3.1-70b-versatile, mixtral-8x7b-32768).
    """

    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "Groq API key not provided. "
            "Pass api_key=... or set GROQ_API_KEY."
        )

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm
# response = llm.invoke("Write a poem about a robot learning to love.")
# print(response)