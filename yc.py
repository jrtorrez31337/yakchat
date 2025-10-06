# yakchat - A command-line REPL for OpenAI-compatible endpoints
# Copyright (C) 2025 Jon Torrez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yc.py

Emoji-powered 🐼 context-savvy client for an OpenAI-compatible /v1 endpoint (e.g., vLLM),
designed around "single system message + structured sections" and safe compaction.

Features
- Single sectioned system message with: role, rules, relevant memory, conversation summary, output style
- Conversation history persistence (JSON) + pretty /history dump + /export to .jsonl
- Memory KV store (/mem key=val, /mem -key, /mem to list)
- Adaptive token budgeting with compaction/summarization
- HF tokenizer for Qwen (fallbacks to tiktoken, then heuristic)
- REPL with emoji commands and helpful stats
- Debug mode to print what is sent to the server

Usage
  python yc.py "What's the good word for the day?" # one-shot CLI
  python yc.py  # interactive REPL

Env overrides
  OPENAI_API_BASE (default: http://yak:8000/v1)
  OPENAI_API_KEY  (default: sk-local-123)
  OPENAI_MODEL    (default: Qwen/Qwen2.5-7B-Instruct)
  YAK_CLIENT_DATA_DIR (default: ./yak_client_data)
  YAK_HARD_LIMIT_TOKENS (default: 7000)
  YAK_COMPACT_AT_TOKENS (default: 5500)
  YAK_SUMMARY_TOKENS (default: 300)
  YAK_SYSTEM_PROMPT (base system instructions; merged into system block)
  YAK_DEBUG_INPUT (1 to enable debug payload printing)

"""

import os
import re
import sys
import json
import time
import shutil
import requests
#from datetime import datetime
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# =========================
# Config defaults
# =========================
DEFAULT_API_BASE = os.getenv("OPENAI_API_BASE", "http://yak:8000/v1")
DEFAULT_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-local-123")
DEFAULT_MODEL    = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")

DATA_DIR         = Path(os.getenv("YAK_CLIENT_DATA_DIR", "./yak_client_data"))
HISTORY_FILE     = DATA_DIR / "session_history.json"
MEMORY_FILE      = DATA_DIR / "memory.json"
TRANSCRIPTS_DIR  = DATA_DIR / "transcripts"
PROFILES_DIR     = DATA_DIR / "profiles"

HARD_LIMIT_TOKENS   = int(os.getenv("YAK_HARD_LIMIT_TOKENS", "7000"))
COMPACT_AT_TOKENS   = int(os.getenv("YAK_COMPACT_AT_TOKENS", "5500"))
SUMMARY_TARGET_TOKENS = int(os.getenv("YAK_SUMMARY_TOKENS", "300"))

BASE_SYSTEM_PROMPT = os.getenv("YAK_SYSTEM_PROMPT", """
You are a helpful, concise assistant. When asked for code or commands, respond with immediately usable snippets and minimal fluff.
""").strip()

DEBUG_FLAG = os.getenv("YAK_DEBUG_INPUT", "0") == "1"

# =========================
# Tokenizer: prefer HF for Qwen; fallback to tiktoken; else heuristic
# =========================
def _build_token_counter():
    # Try HF tokenizer for configured model
    try:
        from transformers import AutoTokenizer  # type: ignore
        _tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
        def estimate_tokens(text: str) -> int:
            return len(_tok(text, add_special_tokens=False).input_ids)
        return estimate_tokens, "HF"
    except Exception:
        pass

    # Try tiktoken
    try:
        import tiktoken  # type: ignore
        _enc = tiktoken.get_encoding("cl100k_base")
        def estimate_tokens(text: str) -> int:
            return len(_enc.encode(text))
        return estimate_tokens, "tiktoken(cl100k_base)"
    except Exception:
        pass

    # Last resort heuristic
    def estimate_tokens(text: str) -> int:
        n = max(1, len(text))
        return max(1, int(n / 4))
    return estimate_tokens, "heuristic(~4 chars/token)"

ESTIMATE_TOKENS, TOKENIZER_KIND = _build_token_counter()


class YakContextClient:
    def __init__(self,
                 api_base: str = DEFAULT_API_BASE,
                 api_key: str = DEFAULT_API_KEY,
                 model: str = DEFAULT_MODEL,
                 data_dir: Path = DATA_DIR,
                 history_file: Path = HISTORY_FILE,
                 memory_file: Path = MEMORY_FILE,
                 transcripts_dir: Path = TRANSCRIPTS_DIR,
                 profiles_dir: Path = PROFILES_DIR,
                 hard_limit_tokens: int = HARD_LIMIT_TOKENS,
                 compact_at_tokens: int = COMPACT_AT_TOKENS,
                 summary_target_tokens: int = SUMMARY_TARGET_TOKENS,
                 base_system_prompt: str = BASE_SYSTEM_PROMPT,
                 debug: bool = DEBUG_FLAG):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.data_dir = data_dir
        self.history_file = history_file
        self.memory_file = memory_file
        self.transcripts_dir = transcripts_dir
        self.profiles_dir = profiles_dir
        self.hard_limit = hard_limit_tokens
        self.compact_at = compact_at_tokens
        self.summary_target = summary_target_tokens
        self.base_system_prompt = base_system_prompt
        self.debug = debug

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self._write_json(self.history_file, {"messages": []})
        if not self.memory_file.exists():
            self._write_json(self.memory_file, {"facts": {}})
        # keep the latest used system block cached for stats
        self._last_system_block: Optional[str] = None

    # ---------- public memory ops ----------
    def add_memory(self, key: str, value: str) -> None:
        mem = self._read_json(self.memory_file)
        mem.setdefault("facts", {})[key] = value
        self._write_json(self.memory_file, mem)

    def delete_memory(self, key: str) -> bool:
        mem = self._read_json(self.memory_file)
        existed = key in mem.get("facts", {})
        mem.get("facts", {}).pop(key, None)
        self._write_json(self.memory_file, mem)
        return existed

    def list_memory(self) -> Dict[str, str]:
        return self._read_json(self.memory_file).get("facts", {})

    # ---------- profiles (alternate system prompts) ----------
    def save_profile(self, name: str, system_prompt: str) -> None:
        p = self.profiles_dir / f"{name}.txt"
        p.write_text(system_prompt, encoding="utf-8")

    def load_profile(self, name: str) -> Optional[str]:
        p = self.profiles_dir / f"{name}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    # ---------- chat API ----------
    def chat(self, user_text: str, system_prompt_override: Optional[str] = None) -> str:
        history = self._read_json(self.history_file).get("messages", [])
        mem_facts = self.list_memory()
        memory_snippet = self._memory_snippet(user_text, mem_facts, max_items=6)

        # Build initial system block (no summary yet)
        sys_block = self._build_system_block(
            base_system=(system_prompt_override or self.base_system_prompt),
            memory_snippet=memory_snippet,
            convo_summary=None
        )
        self._last_system_block = sys_block

        # Candidate message list: single system + history + new user
        messages = [{"role": "system", "content": sys_block}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        # Fit within token budget; compaction will regenerate system block with a summary
        messages = self._fit_into_budget(messages, memory_snippet)

        # Call model
        reply = self._chat_api(messages)

        # Persist
        history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply}
        ])
        self._write_json(self.history_file, {"messages": history})

        # Transcript (append jsonl)
        #ts = datetime.utcnow().strftime("%Y%m%d")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        tr_path = self.transcripts_dir / f"session_{ts}.jsonl"
        with open(tr_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "role": "user", "content": user_text}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"ts": time.time(), "role": "assistant", "content": reply}, ensure_ascii=False) + "\n")

        return reply

    def reset_session(self) -> None:
        self._write_json(self.history_file, {"messages": []})

    # ---------- compaction & system block ----------
    def _build_system_block(self, base_system: str, memory_snippet: str, convo_summary: Optional[str]) -> str:
        mem_text = memory_snippet or "(none)"
        summary_text = convo_summary or "(not available yet; rely on recent turns below)"
        # Clean extra blank lines
        base_system = base_system.strip()
        return (
f"""# 🐼 Role
{base_system}

# 📏 Ground Rules
- Use the conversation summary and recent turns to stay consistent with prior decisions.
- If unsure, ask a brief clarifying question before proceeding.
- Prefer concise answers, code blocks for code, and bullet points for steps.

# 🧠 Relevant Memory
{mem_text}

# 🧾 Conversation Summary
{summary_text}

# ✍️ Output Style
- Tight and actionable; minimal fluff.
- Use fenced ``` code blocks for scripts/commands.
- When listing steps, prefer short bullets."""
        ).strip()

    def _extract_memory_from_sys(self, sys_block: str) -> str:
        # simple regex to capture memory block between headers
        m = re.search(r"# 🧠 Relevant Memory\s+(.*?)\s+# 🧾 Conversation Summary", sys_block, re.S)
        if m:
            return m.group(1).strip()
        return "(none)"

    def _fit_into_budget(self, messages: List[Dict[str, str]], memory_snippet: str) -> List[Dict[str, str]]:
        def toks(msgs: List[Dict[str, str]]) -> int:
            return sum(ESTIMATE_TOKENS(f"{m.get('role','')}:{m.get('content','')}") for m in msgs)

        total = toks(messages)
        if total <= self.compact_at:
            if self.debug:
                print(f"🧮 No compaction: est_tokens={total} (limit={self.compact_at}/{self.hard_limit})")
            return messages

        # Split out system and non-system
        sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        non_sys = [m for m in messages if m.get("role") != "system"]

        # Keep last K msgs (3 pairs = 6). Make it a bit adaptive (min=6, max=12).
        K_MIN = 6
        K_MAX = 12
        # crude adaptivity: shorter history => keep more
        avg_len = max(1, int(sum(len(m.get("content", "")) for m in non_sys) / max(1, len(non_sys))))
        if avg_len < 400:   # short messages -> keep more
            N_KEEP = K_MAX
        else:
            N_KEEP = K_MIN

        if len(non_sys) <= N_KEEP + 1:
            # Not much to compact, may need truncation
            if self.debug:
                print(f"🧮 Low history; attempting truncate-if-needed (total={total})")
            return self._truncate_if_needed(messages)

        head = non_sys[:-N_KEEP]
        tail = non_sys[-N_KEEP:]

        # Summarize head with a small temporary prompt
        summary_text = self._summarize_turns(head, target_tokens=self.summary_target)

        # Rebuild single system with the new summary
        base_system = BASE_SYSTEM_PROMPT if sys_msg is None else self._strip_headers_from_base(sys_msg.get("content", ""))
        new_sys = self._build_system_block(base_system.strip(), memory_snippet, summary_text)
        self._last_system_block = new_sys

        compacted = [{"role": "system", "content": new_sys}] + tail

        # Hard truncate oldest tail pairs if still above hard limit
        while toks(compacted) > self.hard_limit and len(tail) > 2:
            # drop oldest pair (assumes user/assistant alternation; if not, drop two oldest)
            drop = min(2, len(tail))
            tail = tail[drop:]
            compacted = [{"role": "system", "content": new_sys}] + tail

        if self.debug:
            print(f"🧮 Compacted. Tokens≈{toks(compacted)} (hard≤{self.hard_limit}); kept {len(tail)} non-system msgs.")
        return compacted

    def _truncate_if_needed(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        def toks(msgs: List[Dict[str, str]]) -> int:
            return sum(ESTIMATE_TOKENS(f"{m.get('role','')}:{m.get('content','')}") for m in msgs)

        sys_msgs = [m for m in messages if m.get("role") == "system"]
        non_sys = [m for m in messages if m.get("role") != "system"]

        while toks(sys_msgs + non_sys) > self.hard_limit and len(non_sys) > 2:
            non_sys.pop(0)
        return sys_msgs + non_sys

    def _strip_headers_from_base(self, system_block: str) -> str:
        """
        Recover the original base system prompt from a prior system block by removing the sections
        we added. If not found, just return the full content (harmless).
        """
        # Attempt to capture the part right after "# 🐼 Role" until next header
        m = re.search(r"# 🐼 Role\s+(.*?)(?:# 📏 Ground Rules|# 🧠 Relevant Memory|# 🧾 Conversation Summary|# ✍️ Output Style)", system_block, re.S)
        if m:
            return m.group(1).strip()
        return system_block.strip()

    # ---------- summarization ----------
    def _summarize_turns(self, turns: List[Dict[str, str]], target_tokens: int) -> str:
        if not turns:
            return "(no prior context)"
        text = []
        for t in turns:
            role = t.get("role", "user").upper()
            content = t.get("content", "")
            text.append(f"{role}: {content}")
        joined = "\n".join(text)[:20000]  # safety cap

        sys_p = (
            "You are a precise summarizer. Summarize the following dialogue into a concise brief, "
            f"preserving key facts, decisions, actions, and open questions. Aim for ~{target_tokens} tokens. "
            "Use short bullet points."
        )
        msgs = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"Summarize this dialogue:\n\n{joined}"}
        ]
        try:
            return self._chat_api(msgs, temperature=0.2, max_tokens=target_tokens + 120)
        except Exception:
            return "(summary unavailable)"

    def _memory_snippet(self, user_text: str, facts: Dict[str, str], max_items: int = 6) -> str:
        if not facts:
            return ""
        utoks = set(w.lower() for w in re.findall(r"\w+", user_text))
        scored: List[Tuple[int,str,str]] = []
        for k, v in facts.items():
            kv_toks = set(re.findall(r"\w+", (k + " " + v).lower()))
            score = len(utoks.intersection(kv_toks))
            scored.append((score, k, v))
        scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
        top = [(k, v) for s, k, v in scored if s > 0][:max_items]
        if not top:
            top = list(facts.items())[:min(2, len(facts))]
        return "\n".join([f"- {k}: {v}" for k, v in top])

    # ---------- low-level HTTP ----------
    def _chat_api(self,
                  messages: List[Dict[str, str]],
                  temperature: float = 0.7,
                  max_tokens: Optional[int] = None) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if self.debug:
            est_total = sum(ESTIMATE_TOKENS(m.get("content","")) for m in messages)
            print(f"🔍 DEBUG POST {url}  messages={len(messages)}  est_tokens≈{est_total}  tokenizer={TOKENIZER_KIND}")
            for i, m in enumerate(messages[:10]):
                head = m.get("content","")[:700]
                print(f"—[{i}] {m['role'].upper()}—\n{head}\n")

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected API response: {data}")

    # ---------- IO ----------
    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, obj: Dict[str, Any]) -> None:
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        shutil.move(tmp, path)

    # ---------- convenience ----------
    def export_history_jsonl(self) -> Path:
        history = self._read_json(self.history_file).get("messages", [])
        #ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = self.transcripts_dir / f"export_{ts}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for m in history:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        return out

    def stats(self) -> Dict[str, Any]:
        hist = self._read_json(self.history_file).get("messages", [])
        sys_tokens = ESTIMATE_TOKENS(self._last_system_block or "")
        non_sys_tokens = sum(ESTIMATE_TOKENS(m.get("content","")) for m in hist)
        return {
            "model": self.model,
            "api_base": self.api_base,
            "tokenizer": TOKENIZER_KIND,
            "hard_limit": self.hard_limit,
            "compact_at": self.compact_at,
            "summary_target_tokens": self.summary_target,
            "messages_in_history": len(hist),
            "system_block_tokens": sys_tokens,
            "history_tokens_est": non_sys_tokens,
            "total_tokens_est": sys_tokens + non_sys_tokens,
            "data_dir": str(self.data_dir),
        }


def _banner():
    print("🐼 yak ctx client PLUS — context-engineered, memory-aware")
    print("Type /help for commands; Ctrl+C to exit.\n")


def _help():
    print("""✨ Commands
  /help                 show this help
  /reset                clear chat history (keeps memory)
  /mem                  list memory facts
  /mem key=val          add/update memory
  /mem -key             delete memory key
  /stats                show model & token stats
  /export               export history to transcripts/export_*.jsonl
  /history              pretty-print current conversation history
  /tail N               show last N messages (default 10)
  /setsys               set/replace base system prompt (multi-line; end with a single '.')
  /loadsys NAME         load system prompt profile from profiles/NAME.txt
  /saveas NAME          save current base system prompt to profiles/NAME.txt
  /debug on|off         toggle debug payload printing
  /quit                 exit
""")


def _pretty_print_history(messages: List[Dict[str,str]], limit: Optional[int]=None):
    if limit is not None:
        messages = messages[-limit:]
    for i, m in enumerate(messages, 1):
        role = m.get("role","").upper()
        content = m.get("content","")
        print(f"—— {i}. {role} ——")
        print(content if len(content) <= 2000 else content[:2000] + " …(truncated)…")
        print()


def _read_multiline_until_dot() -> str:
    print("✍️  Enter new system prompt. End input with a single '.' on its own line.")
    lines = []
    while True:
        line = input("")
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _main():
    client = YakContextClient()
    # one-shot CLI
    if len(sys.argv) > 1:
        user_msg = " ".join(sys.argv[1:])
        reply = client.chat(user_msg)
        print(reply)
        return

    _banner()
    while True:
        try:
            user_msg = input("🗣️  > ").strip()
            if not user_msg:
                continue

            if user_msg in ("/help", "/?"):
                _help()
                continue
            if user_msg == "/quit":
                print("👋 Bye!")
                break
            if user_msg == "/reset":
                client.reset_session()
                print("🧹 (history cleared)")
                continue
            if user_msg.startswith("/mem"):
                parts = user_msg.split(" ", 1)
                if len(parts) == 1:
                    print(json.dumps(client.list_memory(), indent=2, ensure_ascii=False))
                else:
                    arg = parts[1].strip()
                    if arg.startswith("-"):
                        key = arg[1:].strip()
                        ok = client.delete_memory(key)
                        print("🗑️  (deleted)" if ok else "⚠️  (no such key)")
                    elif "=" in arg:
                        k, v = arg.split("=", 1)
                        client.add_memory(k.strip(), v.strip())
                        print("💾 (saved)")
                    else:
                        print("Usage: /mem key=val   or   /mem -key")
                continue
            if user_msg == "/export":
                path = client.export_history_jsonl()
                print(f"📦 Exported → {path}")
                continue
            if user_msg == "/stats":
                st = client.stats()
                print(json.dumps(st, indent=2, ensure_ascii=False))
                continue
            if user_msg == "/history":
                hist = client._read_json(client.history_file).get("messages", [])
                _pretty_print_history(hist, None)
                continue
            if user_msg.startswith("/tail"):
                try:
                    n = int(user_msg.split(" ",1)[1].strip())
                except Exception:
                    n = 10
                hist = client._read_json(client.history_file).get("messages", [])
                _pretty_print_history(hist, n)
                continue
            if user_msg == "/setsys":
                new_sys = _read_multiline_until_dot()
                if new_sys:
                    client.base_system_prompt = new_sys
                    print("✅ System prompt updated.")
                else:
                    print("⚠️  Empty system prompt; unchanged.")
                continue
            if user_msg.startswith("/loadsys"):
                parts = user_msg.split(" ", 1)
                if len(parts) == 2 and parts[1].strip():
                    name = parts[1].strip()
                    txt = client.load_profile(name)
                    if txt is None:
                        print(f"⚠️  No profile named '{name}'.")
                    else:
                        client.base_system_prompt = txt
                        print(f"✅ Loaded profile '{name}'.")
                else:
                    print("Usage: /loadsys NAME")
                continue
            if user_msg.startswith("/saveas"):
                parts = user_msg.split(" ", 1)
                if len(parts) == 2 and parts[1].strip():
                    name = parts[1].strip()
                    client.save_profile(name, client.base_system_prompt)
                    print(f"💾 Saved current system prompt as profiles/{name}.txt")
                else:
                    print("Usage: /saveas NAME")
                continue
            if user_msg.startswith("/debug"):
                arg = user_msg.split(" ",1)[1].strip().lower() if " " in user_msg else ""
                if arg in ("on","1","true","yes"):
                    client.debug = True
                    print("🐞 Debug: ON")
                elif arg in ("off","0","false","no"):
                    client.debug = False
                    print("🐞 Debug: OFF")
                else:
                    print(f"Debug is {'ON' if client.debug else 'OFF'}")
                continue

            # normal chat
            reply = client.chat(user_msg)
            print(reply)

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"💥 [error] {e}", file=sys.stderr)
            time.sleep(0.4)


if __name__ == "__main__":
    _main()
