from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "conversation_log.csv"
REQUEST_TIMEOUT_SECONDS = 10
MAX_TOOL_ROUNDS = 10

load_dotenv(BASE_DIR / ".env")
API_BASE_URL = os.getenv("INVENTORY_API_URL", "http://127.0.0.1:8000").rstrip("/")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


SYSTEM_PROMPT = """You are Carla's conversational inventory assistant for a small coffee-shop supply store.

Rules:
- Use the provided tools for all inventory facts and all inventory changes. Never invent stock values.
- A delivery or received stock is a positive delta.
- A sale, usage, loss, or outgoing stock is a negative delta.
- The update_stock tool requires a product_id. If the user gives only a product name, call list_inventory first to find the correct id.
- If a product name is ambiguous or does not exist, ask a short clarifying question rather than guessing.
- For low-stock questions, use get_low_stock_alerts. The default threshold is 10 unless the user specifies another threshold.
- Keep answers direct and conversational. Include the product name, resulting quantity, and unit after a successful stock change.
- You may make multiple tool calls before giving the final answer.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_inventory",
            "description": "Return the complete inventory with product ids, names, quantities, and units. Use this to answer stock questions or to find a product id before updating stock.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_product",
            "description": "Register a new inventory product. Use only when the product does not already exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Product name, for example Oat Milk."},
                    "quantity": {"type": "number", "minimum": 0, "description": "Starting quantity."},
                    "unit": {"type": "string", "description": "Unit of measure, for example units, bags, kg, or liters."},
                },
                "required": ["name", "quantity", "unit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_stock",
            "description": "Change stock for an existing product by id. Use a positive delta for deliveries/incoming stock and a negative delta for sales/outgoing stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "minimum": 1, "description": "Existing product id."},
                    "delta": {"type": "number", "description": "Stock change. Positive adds stock; negative removes stock."},
                },
                "required": ["product_id", "delta"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_low_stock_alerts",
            "description": "Return all products whose quantity is below a threshold. Use 10 when the user does not specify a threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 0, "description": "Low-stock threshold. Defaults to 10."}
                },
                "additionalProperties": False,
            },
        },
    },
]


def log_event(actor: str, message: str, tool_call: str = "") -> None:
    """Append one observable loop event to conversation_log.csv."""
    file_exists = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0

    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["actor", "message", "tool_call", "timestamp"],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "actor": actor,
                "message": message,
                "tool_call": tool_call,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


def api_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Call the FastAPI service and return a JSON-safe result for the LLM."""
    url = f"{API_BASE_URL}{path}"

    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status_code": None,
            "error": f"Could not reach the inventory API at {API_BASE_URL}: {exc}",
        }

    try:
        body: Any = response.json()
    except ValueError:
        body = response.text

    if response.ok:
        return {"ok": True, "status_code": response.status_code, "data": body}

    if isinstance(body, dict) and "detail" in body:
        error = body["detail"]
    else:
        error = body

    return {
        "ok": False,
        "status_code": response.status_code,
        "error": error,
    }


def list_inventory() -> dict[str, Any]:
    return api_request("GET", "/inventory")


def add_product(name: str, quantity: float, unit: str) -> dict[str, Any]:
    return api_request(
        "POST",
        "/inventory",
        json={"name": name, "quantity": quantity, "unit": unit},
    )


def update_stock(product_id: int, delta: float) -> dict[str, Any]:
    return api_request(
        "PATCH",
        f"/inventory/{product_id}",
        json={"delta": delta},
    )


def get_low_stock_alerts(threshold: float = 10) -> dict[str, Any]:
    return api_request(
        "GET",
        "/inventory/alerts",
        params={"threshold": threshold},
    )


TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "list_inventory": list_inventory,
    "add_product": add_product,
    "update_stock": update_stock,
    "get_low_stock_alerts": get_low_stock_alerts,
}


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    function = TOOL_FUNCTIONS.get(tool_name)
    if function is None:
        return {"ok": False, "error": f"Unknown tool: {tool_name}"}

    try:
        return function(**arguments)
    except TypeError as exc:
        return {"ok": False, "error": f"Invalid arguments for {tool_name}: {exc}"}
    except Exception as exc:  # Defensive boundary: send a clean error back to the LLM.
        return {"ok": False, "error": f"Tool {tool_name} failed: {exc}"}


def check_api_is_running() -> bool:
    result = list_inventory()
    if result.get("ok"):
        return True

    print("\nCannot start the agent because the inventory API is not available.")
    print(f"Start it first with: uvicorn api.app:app --reload")
    print(f"API error: {result.get('error')}\n")
    return False


def run_agent_turn(
    client: OpenAI,
    messages: list[dict[str, Any]],
    user_input: str,
) -> str:
    """Run Observe -> Think -> Act -> Update -> Repeat for one user message."""

    # OBSERVE: store the new user input in both session memory and the audit log.
    messages.append({"role": "user", "content": user_input})
    log_event("user", user_input)

    for _ in range(MAX_TOOL_ROUNDS):
        # THINK: ask the LLM what to do next using full in-memory history + tool definitions.
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message

        # Preserve the assistant turn in the in-memory session history.
        messages.append(assistant_message.model_dump(exclude_none=True))

        # If there are no tool calls, the loop is complete.
        if not assistant_message.tool_calls:
            final_text = assistant_message.content or "I don't have a response yet."
            log_event("agent", final_text)
            return final_text

        # ACT + UPDATE: execute every tool selected by the LLM and add results to context.
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            log_event(
                "agent",
                json.dumps(arguments, ensure_ascii=False),
                tool_call=tool_name,
            )

            tool_result = execute_tool(tool_name, arguments)
            tool_result_text = json.dumps(tool_result, ensure_ascii=False)

            log_event("tool", tool_result_text, tool_call=tool_name)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result_text,
                }
            )

        # REPEAT: the next loop iteration sends the updated context back to the LLM.

    final_text = "I stopped this turn because too many tool calls were requested. Please try again."
    log_event("agent", final_text)
    return final_text


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        print("Missing GROQ_API_KEY. Copy groq.env.example to .env and add your API key.")
        sys.exit(1)

    if not check_api_is_running():
        sys.exit(1)

    # Groq exposes an OpenAI-compatible API, but the OpenAI SDK does not
    # automatically read GROQ_API_KEY or point to Groq's endpoint.
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
    )

    # Full conversation history is kept here for the lifetime of this terminal session.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("\nInventory Agent is ready.")
    print("Talk naturally. Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("Carla: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.casefold() in {"exit", "quit"}:
            print("Session ended.")
            break

        try:
            answer = run_agent_turn(client, messages, user_input)
            print(f"Agent: {answer}\n")
        except Exception as exc:
            error_message = f"The LLM request failed: {exc}"
            log_event("agent", error_message)
            print(f"Agent: {error_message}\n")


if __name__ == "__main__":
    main()
