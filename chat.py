#!/usr/bin/env python3
import sys
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"
DEFAULT_SYSTEM = "You are a helpful assistant."

DIM = "\033[2m"
RESET = "\033[0m"


def stream_response(messages: list, think: bool) -> str | None:
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": True, "think": think},
        stream=True,
    )

    if response.status_code != 200:
        error = response.json().get("error", response.text)
        print(f"\n[error] {error}\n")
        return None

    full_response = ""
    thinking = False

    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)

        if "error" in chunk:
            print(f"\n[error] {chunk['error']}\n")
            return None

        message = chunk.get("message", {})
        thought = message.get("thinking", "")
        token = message.get("content", "")

        if thought:
            if not thinking:
                print(DIM, end="", flush=True)
                thinking = True
            print(thought, end="", flush=True)

        if token:
            if thinking:
                print(f"{RESET}\n", end="", flush=True)
                thinking = False
            print(token, end="", flush=True)
            full_response += token

        if chunk.get("done"):
            break

    print(RESET)
    return full_response


def main():
    system_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_SYSTEM
    messages = [{"role": "system", "content": system_prompt}]
    think_mode = False

    print(f"Model: {MODEL}")
    print(f"System: {system_prompt}\n")

    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "!system":
            system_prompt = input("new system prompt: ").strip()
            messages = [{"role": "system", "content": system_prompt}]
            print("System prompt updated. Conversation reset.\n")
            continue
        if user_input.lower() == "!think":
            think_mode = not think_mode
            state = "on" if think_mode else "off"
            print(f"Thinking mode {state}.\n")
            continue

        messages.append({"role": "user", "content": user_input})
        print("qwen: ", end="", flush=True)
        reply = stream_response(messages, think_mode)

        if reply is None:
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
