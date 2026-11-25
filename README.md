# Athena AI Agent

Athena AI Agent is a self-hosted personal AI workspace. Chat with many AI models and providers,
run an agent that can write files, run shell commands, make PDFs and
spreadsheets, search the web, do deep research, and manage local models —
all from one page, on your own machine.

Built in Rust (Axum + Tokio). The web UI is a single page served by the
binary. It runs on a Linux machine or on a phone under Termux (proot
Ubuntu).

## Features

- Chat with any OpenAI-compatible endpoint, plus local Ollama models
- Agent mode: the model plans and uses tools (`run_shell`, `read_file`,
  `write_file`, `list_files`, `search_images`, `create_pdf`) and shows
  every step as a timeline, with a summary when it finishes
- Real web photos under chat replies, image and video generation
- Deep Research: plans, searches (own SearXNG), reads pages, writes a
  sourced report
- Cookbook: search and download GGUF models, import them into Ollama
- PDF tools: upload and read PDFs (including scanned pages via OCR),
  make a PDF of a chat, edit a PDF keeping its layout
- Document skills: make .docx, .xlsx and .pptx files from a chat request
- Skills and memory, personas, themes, chat background, read aloud
- Works with free model tiers (Gemini, Groq, Conea, free-pi and more)

## Run it

```sh
cargo build --release
./target/release/odysseus-rust
```

Then open http://127.0.0.1:8080 in a browser. Your data (chats, endpoints,
memories, settings) is stored as JSON files next to the binary and is
created on first run — add your own API keys in Settings.

## Layout

- `src/main.rs` — the whole server and web UI
- `Cargo.toml` / `Cargo.lock` — Rust dependencies
- `*.py` — small helpers (PDF text/OCR, image conversion, vision
  descriptions)
- `logo.webp` — the app logo

## Acknowledgements

Athena AI Agent is modeled on **Odysseus AI**, the self-hosted AI
workspace it was built to match. The basics come from Odysseus:

- **Chat** — talk to many AI models through endpoints you add yourself.
- **Agent** — the model works, not just answers: it writes files, runs
  shell commands, searches the web, makes PDFs and spreadsheets, and
  shows each step as it goes.
- **Skills** — reusable instructions that teach the agent how to do a
  specific job.
- **Memory** — it remembers things about you and your work, so you do
  not repeat yourself.
- **Workspace** — a folder where the agent reads and writes files.
- **Deep Research and Cookbook** — research that plans, searches and
  writes a sourced report; a model manager for finding and running
  models on your own machine.

That is the foundation Athena AI Agent builds on.

## Note

This is a personal project. The agent runs shell commands on your
machine, so only run it where you trust the models and prompts you use.
