# AI Basic Inventory Agent Loop

A standalone beginner-friendly project with two parts:

1. A FastAPI REST API that stores inventory in `api/products.csv`.
2. A plain-Python AI agent in `agent.py` that uses the API as LLM tools.

No agent framework (LangChain, LlamaIndex, AutoGen, etc.) is used. The agent loop is implemented manually.

## Project structure

```text
inventory-agent-loop/
├── api/
│   ├── __init__.py
│   ├── app.py
│   └── products.csv
├── agent.py
├── conversation_log.csv
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/inventory` | Return the full inventory |
| POST | `/inventory` | Add a product with `name`, `quantity`, and `unit` |
| PATCH | `/inventory/{product_id}` | Apply a stock `delta` (positive = incoming, negative = outgoing) |
| GET | `/inventory/alerts?threshold=10` | Return products below the threshold |

The API also exposes `/docs` for Swagger UI and `/` as a simple health response.

## 1. Create and activate a virtual environment

From the project root:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure Groq

The agent uses Groq through its OpenAI-compatible API. Copy `groq.env.example` to `.env` and add your Groq API key.

### Windows PowerShell

```powershell
Copy-Item groq.env.example .env
```

### macOS / Linux

```bash
cp groq.env.example .env
```

Then edit `.env`:

```env
GROQ_API_KEY=your_real_groq_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1
INVENTORY_API_URL=http://127.0.0.1:8000
```

`agent.py` passes `GROQ_API_KEY` explicitly to the OpenAI Python SDK and sets the base URL to Groq's API. The SDK package is named `openai` because Groq implements its API format; this does not mean the agent sends requests to OpenAI. Change `GROQ_MODEL` to another tool-calling model currently supported by Groq if needed.

## 3. Start the API first

Open Terminal 1 in the project root:

```bash
uvicorn api.app:app --reload
```

Useful URLs:

- API: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

The agent intentionally checks that the API is reachable before starting.

## 4. Start the agent

Open Terminal 2 in the same project root, activate the same virtual environment, then run:

```bash
python agent.py
```

Example conversation:

```text
Carla: What products are running low?
Agent: Paper Cups, Vanilla Syrup, and Cold Brew Concentrate are below 10.

Carla: We just received 30 units of oat milk.
Agent: Done. Oat Milk now has 54 units.

Carla: We sold 12 bags of arabica today.
Agent: Done. Arabica Coffee Beans now has 6 bags.
```

## How the manual agent loop works

For every user turn, `agent.py` follows:

```text
Observe
  -> add Carla's message to in-memory history
Think
  -> send the full history + tool definitions to the LLM
Act
  -> call the FastAPI endpoint selected by the LLM
Update
  -> add the tool result back to the LLM message history
Repeat
  -> continue until the LLM returns a final response with no tool calls
```

The `messages` list remains in memory for the duration of the process, so follow-up questions can use earlier context.

## Conversation log

`conversation_log.csv` is append-only. Existing rows are never cleared by `agent.py`.

It contains exactly these columns:

```text
actor,message,tool_call,timestamp
```

Observable events are logged as:

- `user`: Carla's input
- `agent`: a tool request or final response
- `tool`: the API/tool result

The `tool_call` column is empty when no tool applies and contains the tool name for tool-call and tool-result rows.

## Important behavior

- Positive `delta` = stock came in.
- Negative `delta` = stock went out.
- Stock cannot be reduced below zero.
- Duplicate product names return HTTP 409.
- Missing products return HTTP 404.
- Validation errors return HTTP 422.
- Low-stock threshold defaults to 10 and can be changed per request.
- The LLM never edits `products.csv` directly; only the FastAPI API reads/writes inventory data.
