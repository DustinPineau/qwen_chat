#!/usr/bin/env python3
import sys
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"
DEFAULT_SYSTEM = "You are a helpful assistant."



def stream_response(messages: list) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": True},
        stream=True,
    )

    full_response = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            print(token, end="", flush=True)
            full_response += token
            if chunk.get("done"):
                break
    print()
    return full_response



def main():
    system_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_SYSTEM
    messages = [{"role": "system", "content": system_prompt}]

    print(f"Model: {MODEL}")
    print(f"System: {system_prompt}\n")

    while True:
        try:
                user_input = input("you: ").strip()
        except  (EOFError, KeyboardInterrupt):
            print()
            break
        
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "system":
            system_prompt = input("new system prompt: ").strip()
            messages = [{"role": "system", "content": system_prompt}]
            print(f"System prompt updated. Conversation reset.\n")
            continue

        messages.append({"role": "user", "content": user_input})
        print("qwen: ", end="", flush=True)
        reply = stream_response(messages)
        messages.append({"role": "assistant", "content": reply})



if __name__ == "__main__":
    main()
