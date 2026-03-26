import os
import json
import sys
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
# FastAPI app
app = FastAPI(title="KavachMitra API")

print("Python being used:", sys.executable)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferMemory
# from langchain_community.memory import ConversationBufferMemory
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage


def _load_env():
    load_dotenv()


def _configure_model():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise SystemExit("Missing GROQ_API_KEY in .env")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.2,
        max_tokens=800
    )

    return llm


# Initialize memory (stores conversation history)
memory = ConversationBufferMemory(return_messages=True)


system_prompt = """
You are KavachMitra, the official AI cybersecurity assistant for the KavachX platform developed by team BItWin Init.

Your role:
- Answer ONLY questions related to Cyber Security.
- Topics allowed include:
  phishing, malware, ransomware, deepfakes, social engineering,
  data breaches, cyber attacks, authentication, encryption,
  secure coding, network security, AI security, threat detection,
  digital privacy, cyber laws, security best practices.

If the question is NOT related to cybersecurity,
respond with:

"This is not my domain. I am KavachMitra, the cybersecurity assistant of KavachX. I only answer questions related to Cyber Security."

Return output strictly in JSON with the following keys:
- topic
- explanation
- prevention_tips
- resources

If the question is outside cybersecurity scope, return JSON with all values containing the above message.

Do NOT include markdown.
"""


def kavach_mitra_agent(llm, user_query: str) -> Dict:

    # Get previous conversation
    history = memory.load_memory_variables({})["history"]

    messages = [SystemMessage(content=system_prompt)]

    # Add previous chat history
    messages.extend(history)

    # Add new question
    messages.append(HumanMessage(content=user_query))

    response = llm.invoke(messages)

    text = response.content.strip()

    # Save conversation into memory
    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "raw": text,
            "note": "Model output was not valid JSON"
        }

class ChatRequest(BaseModel):
    message: str

# Load environment and configure model once for API
_load_env()
llm = _configure_model()

@app.get("/")
def health_check():
    return {"status": "KavachMitra API running"}


@app.post("/chat")
def chat_endpoint(req: ChatRequest):

    result = kavach_mitra_agent(llm, req.message)

    return result

def main():

    llm = _configure_model()

    print("\nKavachMitra Cybersecurity Assistant")
    print("Type 'exit' to stop\n")

    while True:

        user_question = input("Ask KavachMitra: ")

        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye from KavachMitra.")
            break

        result = kavach_mitra_agent(llm, user_question)

        print("\n=== KavachMitra Response ===")
        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    _load_env()
    main()