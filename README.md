# yakchat🐼

A context-engineered, memory-aware client for OpenAI-compatible `/v1` endpoints (e.g., vLLM).  
It uses a **single, sectioned system message**, **adaptive compaction with summaries**, and a **KV memory** store.

## Features
- Single system message with sections (Role, Rules, Relevant Memory, Conversation Summary, Output Style)
- Conversation history persistence + export
- Memory KV (`/mem key=val`, `/mem -key`, `/mem`)
- HF tokenizer for Qwen (fallbacks to tiktoken, then heuristic)
- Debug mode to print what the model actually receives
- Handy REPL commands

## Install
WSL/Linux:
git clone https://github.com/jrtorrez31337/yakchat.git
cd yakchat
python -m venv .yakchat && source .yakchat/bin/activate   # Windows: .yakchat\Scripts\activate
pip install -r requirements.txt
# optional: pip install -e .   # enables the yakchat console command
