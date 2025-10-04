feat(cli): initial public release of yak-ctx-client 🐼 — context-engineered OpenAI-compatible REPL

## Motivation
This CLI client implements practical “effective context engineering” patterns inspired by
Anthropic’s *Effective Context Engineering for AI Agents* (2025). It aims to make small,
local models more reliable over long conversations by structuring prompts and managing
context explicitly rather than hoping the model “just remembers.”

## Purpose
Provide an ergonomic, batteries-included REPL for OpenAI-compatible /v1 endpoints
(e.g., vLLM) that:
- Preserves conversation history on disk and exports transcripts
- Compacts older turns into a durable, model-visible summary
- Surfaces “relevant memory” (simple KV facts) inline
- Keeps a single, sectioned system message so servers/templates don’t drop later system prompts

## Key Features
- Single, sectioned **system message** with:
  - 🐼 Role
  - 📏 Ground Rules
  - 🧠 Relevant Memory
  - 🧾 Conversation Summary
  - ✍️ Output Style

- **Adaptive compaction**: summarize older turns into a concise brief that’s re-injected into
  the system block (not as a low-weight assistant message).
- **Tokenizer-aware budgeting**: prefers HF tokenizer for Qwen (falls back to tiktoken, then heuristic)
  to reduce “phantom trims.”
- **Memory KV store** with naive relevance selection.
- **REPL UX** with emoji commands:
  `/help, /mem, /reset, /export, /stats, /history, /tail, /setsys, /loadsys, /saveas, /debug`
- **Transcripts** written as JSONL in `./yak_client_data/transcripts/`.
- **Config via env**: `OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_MODEL`, token budgets, etc.
- **Debug mode** prints the exact system block + first N messages sent to the server.

## Design Choices
- **Single system message**: avoids server/template quirks where additional system entries
  are ignored or down-weighted.
- **Summary lives in system**: makes the synopsis “authoritative” for the model.
- **Keep tail pairs** adaptively (3–6 pairs by default), then summarize the head.
- **Explicit logging** and `/stats` for visibility into what the model actually sees.

## Getting Started

1) Configure your endpoint (example):
- export OPENAI_API_BASE="http://yak:8000/v1"
- export OPENAI_API_KEY="sk-local-123"
- export OPENAI_MODEL="Qwen/Qwen2.5-7B-Instruct"

2) Run:

- python yc.py "What's the good word for the day?" # one time query

or interactive:

- python yc.py

3) Explore `/help` for commands.

## References
- Anthropic, “Effective Context Engineering for AI Agents” (2025)

## Notes
- Works well on Qwen2.5-7B-Instruct; larger variants (14B/32B) improve long-horizon fidelity.
- The client is endpoint-agnostic as long as it speaks the OpenAI `/chat/completions` API.


## Install
WSL/Linux:
- git clone https://github.com/jrtorrez31337/yakchat.git
- cd yakchat
- python -m venv yakchat 
- source .yakchat/bin/activate  
- pip install -r requirements.txt
