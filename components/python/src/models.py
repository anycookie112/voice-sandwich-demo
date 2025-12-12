from langchain_ollama import ChatOllama


def get_ollama_model():
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0,
        # other params...
    )
    return llm

