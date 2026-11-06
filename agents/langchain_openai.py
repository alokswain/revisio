from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = "You are a helpful assistant. Give clear and precise answers."


def ask(messages, *, model, api_key):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), MessagesPlaceholder("history")]
    )
    chat_model = ChatOpenAI(
        model=model,
        api_key=api_key,
        max_completion_tokens=2048,
        timeout=90,
        max_retries=0,
    )
    chain = prompt | chat_model | StrOutputParser()
    answer = chain.invoke(
        {"history": [(message["role"], message["content"]) for message in messages]}
    )
    if not answer.strip():
        raise ValueError("Empty model response")
    return answer