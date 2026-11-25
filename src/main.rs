use axum::{
    extract::{DefaultBodyLimit, Json, Path, Query},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{Html, IntoResponse, sse::{Event, Sse}},
    routing::{delete, get, post, put},
    Router,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::task::AbortHandle;
use tokio_stream::wrappers::ReceiverStream;

const OLLAMA_URL: &str = "http://127.0.0.1:11434/api/chat";
// Mirage on-device vision (Cactus Compute, 22MB VLM). The local model
// (qwen3-0.6b) is text-only, so an attached photo is described here first
// by Mirage and the text description is given to the model instead.
// Helper script: mirage_describe.py (plain-text description on stdout).
const MIRAGE_SCRIPT: &str = "/root/odysseus-rust/mirage_describe.py";
const MIRAGE_PYTHON: &str = "/root/venv/bin/python3";
const PDF_OCR_SCRIPT: &str = "/root/odysseus-rust/pdf_ocr.py";
const OLLAMA_TAGS_URL: &str = "http://127.0.0.1:11434/api/tags";
const DEFAULT_MODEL: &str = "qwen3-abliterated-0.6b";
const CHATS_DIR: &str = "chats";
const WORKSPACE_DIR: &str = "workspace";

const AGENT_SYSTEM: &str = "You are Mawaxi AI Agent, running on Steven's phone. You can use tools to do tasks. To use a tool, reply with ONLY a fenced tool block and nothing else:\n```tool\n<tool name>\n<arguments>\n```\nTools:\n- run_shell: run a shell command. Argument: the command.\n- read_file: read a file. Argument: the file path (absolute, or relative to the workspace).\n- write_file: write a file. First line of the arguments is the file path (absolute, or relative to the workspace), the rest is the file content.\n- list_files: list files in a directory. Argument: the directory path (absolute, or relative to the workspace; use . for the workspace itself).\n- search_images: search the web for REAL photos of something. Argument: the search query, e.g. giraffe. Returns up to 5 real image results, each with the direct image URL, a thumbnail URL, a title, and the source page URL. Use it when a picture would genuinely help the answer (an animal, object, place, person, or a medical image such as a real blood smear). These are real photos from the web — never describe them as AI-generated, and never offer an AI image as a substitute.\n- create_pdf: create a PDF file. First line of the arguments is the filename (without .pdf, e.g. RBC inclusions), the rest is the full text to put in the PDF. Returns the download link for the saved PDF.\nExample — to read hello.txt, reply with exactly:\n```tool\nread_file\nhello.txt\n```\nHARD RULES:\n- CREATE means write_file. When the user asks you to create, build, or make a file, program, app, or system, you MUST use write_file to create the actual file. Pasting code in your reply does NOT create a file. Never answer a create request with only code in your reply. A PDF is the exception: use create_pdf for a PDF, never write_file.\n- RUN means run_shell. After creating a file, you MUST run it with run_shell. First check it compiles with python -m py_compile <file>, then run it. Do NOT stop after writing the file, and do NOT stop after compiling — you must run the program too.\n- If your reply contains code, you have not finished. Create the file first, run it, then summarise in plain text.\n- Only report what the tool output actually showed. Do not invent or guess output. If you did not run the program, do not say it printed anything.\n- If a program waits for input(), it will hang. Run it with piped input so it exits, e.g. printf \"5\\n\" | python hospital.py.\nWORKED EXAMPLE — user asks \"Create and run a hospital management system in python\":\nStep 1 — reply with ONLY a tool block. Write the FULL program inside write_file. Here is the kind of complete program to write:\n```tool\nwrite_file\nhospital.py\npatients = {}\ndef add_patient():\n    pid = input(\"Patient ID: \")\n    name = input(\"Name: \")\n    patients[pid] = name\n    print(\"Added:\", name)\ndef view_patient():\n    pid = input(\"Patient ID: \")\n    print(patients.get(pid, \"Not found\"))\ndef list_patients():\n    for pid, name in patients.items():\n        print(pid, \"-\", name)\nwhile True:\n    print(\"1. Add  2. View  3. List  5. Exit\")\n    choice = input(\"Choose: \")\n    if choice == \"1\":\n        add_patient()\n    elif choice == \"2\":\n        view_patient()\n    elif choice == \"3\":\n        list_patients()\n    elif choice == \"5\":\n        print(\"Goodbye!\")\n        break\n```\nStep 2 — after it saves, reply with ONLY:\n```tool\nrun_shell\npython -m py_compile hospital.py\n```\nStep 3 — then run it with input piped so it exits:\n```tool\nrun_shell\nprintf \"5\\n\" | python hospital.py\n```\nStep 4 — then reply with your final answer as plain text, summarising what you created and what the program printed. Never paste the code as your final answer.\nAfter a tool runs you get its output as a message. Use tools when needed, then reply with your final answer as plain text with no tool block.\nIMAGES: when a picture would genuinely help the answer, use search_images first, then include the best 1-3 results in your final reply as Markdown images, one per line: ![short description](direct image URL) — followed by the source page as a link, e.g. [Source: title](source page URL). At most 3 images per reply. Only use image URLs the tool actually returned — never invent a URL. Never generate an image and never claim a web photo is your own creation.\nPDFS: when the user asks for a PDF, use create_pdf - first line is the filename, the rest is the full text to put in the PDF. Photos referenced in the text as Markdown images ![alt](url) are embedded in the PDF. After it saves, give Steven the download link from the tool output. Never say you cannot create a PDF.\nPaths may be absolute or relative to the workspace. There is no safety check: you run as root and can reach any path, so only do what the user asked.\nA2UI: when the answer is a decision, a comparison, a form, or an approval — something Steven should tap instead of read — answer with a real interface. Put a fenced ```a2ui block in your final reply with a JSON surface: {\"surfaceId\": \"x\", \"title\": \"...\", \"components\": [...]}. Components you may use: Text {text, variant: body|caption|heading}, Card {title, subtitle, badge}, Button {label, action}, ChoiceChips {options: [...]}, Divider. Only use those components — never invent one, never send HTML or code. A tap sends the label back to you as a message, so continue from there. Use A2UI only when tapping beats reading; plain text is fine for everything else.\nWrite your final answer in plain text. Do not use asterisks (*) anywhere in it — no bold, no italic, no asterisk bullets. Use plain words, and a dash (-) for a list item if you need a list.";


#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    video: Option<String>,
    // Source page URL for a real web photo shown in chat (see web_image).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    // Text of a PDF attached in chat (client sends camelCase pdfText/pdfName).
    // Without these fields serde drops them on save, so a reloaded or
    // resent message loses the PDF text and the model sees nothing.
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "pdfText")]
    pdf_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "pdfName")]
    pdf_name: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Chat {
    id: String,
    title: String,
    model: String,
    messages: Vec<ChatMessage>,
    // The browser's new-chat object does not send updated_at; default it so
    // saving a chat never fails to deserialize. put_chat overwrites it anyway.
    #[serde(default)]
    updated_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    skill_id: Option<String>,
}

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    endpoint_id: Option<String>,
    #[serde(default)]
    skill_id: Option<String>,
    #[serde(default)]
    persona: Option<String>,
}

#[derive(Serialize, Clone)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    think: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct Endpoint {
    id: String,
    name: String,
    base_url: String,
    api_key: String,
    model: String,
    #[serde(default)]
    provider: String,
    // Model list fetched from the endpoint's /models route, cached so the
    // picker is instant; empty until the first successful fetch.
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    models_fetched_at: u64,
}

#[derive(Serialize)]
struct EndpointPublic {
    id: String,
    name: String,
    base_url: String,
    model: String,
    provider: String,
    models: Vec<String>,
    models_fetched_at: u64,
}

#[derive(Deserialize)]
struct EndpointCreate {
    name: String,
    base_url: String,
    api_key: String,
    model: String,
    #[serde(default)]
    provider: String,
    #[serde(default)]
    models: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Memory {
    id: String,
    text: String,
}

#[derive(Deserialize)]
struct MemoryCreate {
    text: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct Skill {
    id: String,
    name: String,
    instruction: String,
}

#[derive(Deserialize)]
struct SkillCreate {
    name: String,
    instruction: String,
}

// Notes: a title + free text, stored in notes.json.
#[derive(Serialize, Deserialize, Clone)]
struct Note {
    id: String,
    title: String,
    body: String,
    updated_at: u64,
}

#[derive(Deserialize)]
struct NoteCreate {
    title: String,
    body: String,
}

#[derive(Deserialize)]
struct NoteUpdate {
    title: Option<String>,
    body: Option<String>,
}

// Cookbook: a model manager, mirroring the Odysseus Cookbook (model serving).
// NOT a recipe book. It searches HuggingFace, lists cached models on the
// phone, downloads via Ollama or as a GGUF file, lists running models, and
// stops them. The old recipe data file cookbook.json is left on disk but
// is no longer read or written.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GgufFileInfo {
    filename: String,
    #[serde(default)]
    size: Option<i64>,
}
#[derive(Serialize, Deserialize, Clone)]
struct HfModel {
    id: String,
    author: String,
    downloads: i64,
    likes: i64,
    pipeline_tag: String,
    last_modified: String,
    #[serde(default)]
    size_bytes: Option<i64>,
    #[serde(default)]
    gguf_files: Vec<GgufFileInfo>,
}

#[derive(Serialize, Clone)]
struct CachedOllamaModel {
    name: String,
    size: i64,
    modified_at: String,
    parameter_size: String,
    quantization: String,
}

#[derive(Serialize, Clone)]
struct CachedGguf {
    name: String,
    size: u64,
    modified: u64,
}

#[derive(Serialize, Clone)]
struct CachedModels {
    ollama: Vec<CachedOllamaModel>,
    gguf: Vec<CachedGguf>,
    disk_free_bytes: Option<u64>,
    disk_total_bytes: Option<u64>,
}

#[derive(Serialize, Clone)]
struct RunningModels {
    ollama: Vec<serde_json::Value>,
    llama_server: Option<serde_json::Value>,
}

#[derive(Serialize, Clone)]
struct DownloadTask {
    id: String,
    kind: String,
    name: String,
    status: String,
    detail: String,
}

#[derive(Deserialize)]
struct DownloadRequest {
    kind: String,
    name: Option<String>,
    url: Option<String>,
    filename: Option<String>,
}

#[derive(Deserialize)]
struct CookbookStopRequest {
    kind: String,
    name: Option<String>,
    pid: Option<u32>,
}

static DOWNLOAD_TASKS: OnceLock<Mutex<HashMap<String, DownloadTask>>> = OnceLock::new();

fn download_tasks() -> &'static Mutex<HashMap<String, DownloadTask>> {
    DOWNLOAD_TASKS.get_or_init(|| Mutex::new(HashMap::new()))
}

// ---------- Deep Research ----------
// Iterative research engine, mirroring the Odysseus IterResearch loop
// (Think -> Search -> Extract -> Synthesize), simplified for the phone:
// PLAN (model) -> up to 3 rounds of query-gen -> real web search (Bing
// HTML, DuckDuckGo fallback) -> fetch + extract relevant passages ->
// synthesize -> stop check -> final report with a Sources list. Progress
// streams to the browser as SSE events so the run is visible, not a blank
// wait. The old single-call version (one model call, no web search, no
// sources) is gone: that is why Steven found it less advanced.

const RESEARCH_SYSTEM: &str = "You are a thorough research assistant. Research the topic deeply and write a detailed, structured report with these sections:\n1. Background\n2. Key points\n3. Evidence and details\n4. Open questions\n5. Conclusion\nBe specific and factual. Say clearly when something is uncertain or disputed. Do not invent sources or facts.";

const RESEARCH_PLAN_PROMPT: &str = "You are a research strategist. Before searching, analyze this question and create a research plan.\n\nQuestion: {topic}\n\nBreak this question down:\n1. What are the key sub-topics that need to be covered for a comprehensive answer?\n2. What specific data points, facts, or perspectives should we look for?\n3. What would a complete, high-quality answer include?\n\nReturn a JSON object with:\n- \"sub_questions\": Array of 3-6 specific sub-questions to investigate\n- \"key_topics\": Array of key topics/angles to cover\n- \"success_criteria\": One sentence describing what a complete answer looks like\n\nReturn ONLY the JSON object, nothing else.";

const RESEARCH_QUERY_PROMPT: &str = "You are a research assistant planning web searches.\n\nOriginal question: {topic}\n\nResearch plan:\n{plan}\n\nWhat we know so far:\n{report}\n\nRound: {round}\n\nGenerate {n} focused search queries that will help answer the question. Prefer queries that cover angles not yet covered by the evidence so far.\n\nReturn ONLY a JSON array of query strings, nothing else. Example: [\"query one\", \"query two\"]";

const RESEARCH_SYNTH_PROMPT: &str = "You are updating an evolving research report.\n\nOriginal question: {topic}\n\nCurrent report:\n{report}\n\nNew findings from this round:\n{findings}\n\nIntegrate the new findings into the existing report. Produce an updated, well-organized report that answers the original question as completely as possible given all evidence so far. Remove redundancy, resolve contradictions, and maintain logical flow. Keep source URLs as inline citations where relevant. Call out weak or missing evidence plainly, and avoid overconfident conclusions when the evidence is thin.\n\nWrite only the updated report — no preamble or meta-commentary.";

const RESEARCH_STOP_PROMPT: &str = "You are deciding whether a research report is comprehensive enough.\n\nOriginal question: {topic}\n\nCurrent report:\n{report}\n\nRounds completed: {round} of {max_rounds}\n\nBased on the report so far, do we have enough information to answer the question comprehensively? Consider: are the key aspects addressed, are there obvious gaps, is the evidence from multiple sources?\n\nReply with ONLY \"YES\" or \"NO\" followed by a brief one-sentence reason.";

const RESEARCH_FINAL_PROMPT: &str = "Write a long, detailed, comprehensive research report answering this question:\n\nQuestion: {topic}\n\nAll collected evidence and analysis:\n{report}\n\nRequirements:\n- Write a thorough report with clear ## headings and ### subheadings\n- Use the Background / Key points / Evidence and details / Open questions / Conclusion structure\n- Synthesize and analyze — explain WHY things matter, draw comparisons, provide context\n- Include specific data points, numbers, and statistics from the evidence\n- Include source URLs as inline citations [like this](url)\n- Note where sources agree and where they disagree\n- If evidence is weak or missing, say that directly instead of filling gaps with guesses\n- Add a brief executive summary at the top\n- End with a clear conclusion that directly answers the question";

const RESEARCH_FINAL_PROMPT_BRIEF: &str = "Write a brief summary answering this question in a few short paragraphs:\n\nQuestion: {topic}\n\nAll collected evidence and analysis:\n{report}\n\nRequirements:\n- Keep it short: an executive summary of 3-5 sentences, then 2-3 short paragraphs covering the key findings\n- Include the most important facts and numbers from the evidence\n- Include source URLs as inline citations [like this](url)\n- If evidence is weak or missing, say that directly instead of filling gaps with guesses\n- End with one clear sentence that directly answers the question";

const RESEARCH_FINAL_PROMPT_BULLETS: &str = "Write the key findings answering this question as a bullet list:\n\nQuestion: {topic}\n\nAll collected evidence and analysis:\n{report}\n\nRequirements:\n- Start with one sentence stating the direct answer\n- Then a bullet list of the key findings, one finding per bullet, each with its source URL as an inline citation [like this](url)\n- Include the most important facts and numbers from the evidence\n- Add a short 'Open questions' bullet section at the end for anything the evidence did not settle\n- If evidence is weak or missing, say that directly instead of filling gaps with guesses";

const RESEARCH_UA: &str = "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36";

#[derive(Deserialize)]
struct ResearchRequest {
    topic: String,
    model: Option<String>,
    endpoint_id: Option<String>,
    // Options from the Deep Research view (all optional — server defaults
    // copy Odysseus, see research_rounds_choice / research_engine_choice).
    rounds: Option<u32>,
    format: Option<String>,
    engine: Option<String>,
}

#[derive(Deserialize)]
struct ResearchCancelRequest {
    run_id: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct ResearchPlan {
    #[serde(default)]
    sub_questions: Vec<String>,
    #[serde(default)]
    key_topics: Vec<String>,
    #[serde(default)]
    success_criteria: String,
}

#[derive(Serialize, Clone)]
struct SearchResult {
    title: String,
    url: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct ResearchSource {
    title: String,
    url: String,
}

// In-flight research runs: run_id -> cancelled flag. The Cancel button sets
// it; the loop checks it before every step.
static RESEARCH_RUNS: OnceLock<Mutex<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>> =
    OnceLock::new();

fn research_runs() -> &'static Mutex<HashMap<String, Arc<std::sync::atomic::AtomicBool>>> {
    RESEARCH_RUNS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn research_new_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("rs-{}", nanos)
}

// ---------- Deep Research history (server-side, survives restarts) ----------
// Every completed run (and failed runs, with their error) is saved to
// research_history.json so the History tab can list, reopen, visualise,
// chat about, and delete past reports.
const RESEARCH_HISTORY_FILE: &str = "research_history.json";

#[derive(Serialize, Deserialize, Clone)]
struct ResearchHistoryEntry {
    id: String,
    topic: String,
    report: String,
    sources: Vec<ResearchSource>,
    duration_secs: u64,
    created_at: u64,
    model: String,
    endpoint: String,
    rounds: usize,
    format: String,
    engine: String,
    status: String, // "completed" | "failed"
    #[serde(default)]
    error: Option<String>,
}

static RESEARCH_HISTORY_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn research_history_lock() -> &'static Mutex<()> {
    RESEARCH_HISTORY_LOCK.get_or_init(|| Mutex::new(()))
}

fn load_research_history() -> Vec<ResearchHistoryEntry> {
    load_json(RESEARCH_HISTORY_FILE)
}

fn research_format_duration(secs: u64) -> String {
    format!("{}:{:02}", secs / 60, secs % 60)
}

fn save_research_history_entry(entry: ResearchHistoryEntry) {
    let _guard = match research_history_lock().lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let mut list = load_research_history();
    list.retain(|e| e.id != entry.id);
    list.insert(0, entry);
    list.truncate(100);
    let _ = save_json(RESEARCH_HISTORY_FILE, &list);
}

async fn list_research_history() -> Json<Vec<ResearchHistoryEntry>> {
    Json(load_research_history())
}

async fn get_research_history_entry(
    Path(id): Path<String>,
) -> Result<Json<ResearchHistoryEntry>, (StatusCode, String)> {
    load_research_history()
        .into_iter()
        .find(|e| e.id == id)
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "research report not found".into()))
}

async fn delete_research_history_entry(
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let _guard = research_history_lock()
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "history locked".to_string()))?;
    let mut list = load_research_history();
    let before = list.len();
    list.retain(|e| e.id != id);
    if list.len() == before {
        return Err((StatusCode::NOT_FOUND, "research report not found".into()));
    }
    save_json(RESEARCH_HISTORY_FILE, &list)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(StatusCode::NO_CONTENT)
}

// Format a Unix timestamp as "YYYY-MM-DD HH:MM UTC" (no chrono dep).
// Civil-from-days algorithm (Howard Hinnant).
fn research_format_date(ts: u64) -> String {
    let secs = ts as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let (hour, min) = ((rem / 3600) as u32, ((rem % 3600) / 60) as u32);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02} {:02}:{:02} UTC", y, m, d, hour, min)
}

// "THE HOUSE FLY" -> "the-house-fly" (for the download filename).
fn research_topic_slug(topic: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for c in topic.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("research");
    }
    out
}

// Render a saved research report as a Markdown file (for Download).
fn research_entry_to_markdown(e: &ResearchHistoryEntry) -> String {
    let mut md = String::new();
    md.push_str(&format!("# {}\n\n", e.topic));
    md.push_str(&format!("- **Date:** {}\n", research_format_date(e.created_at)));
    md.push_str(&format!("- **Model:** {}\n", e.model));
    md.push_str(&format!("- **Endpoint:** {}\n", e.endpoint));
    md.push_str(&format!("- **Engine:** {}\n", e.engine));
    md.push_str(&format!("- **Rounds:** {}\n", e.rounds));
    md.push_str(&format!("- **Duration:** {}\n", research_format_duration(e.duration_secs)));
    md.push_str(&format!("- **Status:** {}\n\n", e.status));
    if let Some(err) = &e.error {
        md.push_str(&format!("> Failed: {}\n\n", err));
    }
    md.push_str(&e.report);
    md.push_str("\n\n## Sources\n\n");
    if e.sources.is_empty() {
        md.push_str("No sources saved.\n");
    } else {
        for (i, src) in e.sources.iter().enumerate() {
            md.push_str(&format!("{}. [{}]({})\n", i + 1, src.title, src.url));
        }
    }
    md
}

// GET /api/research/history/:id/download — the report as a .md file.
async fn download_research_history_entry(
    Path(id): Path<String>,
) -> Result<(HeaderMap, String), (StatusCode, String)> {
    let entry = load_research_history()
        .into_iter()
        .find(|e| e.id == id)
        .ok_or((StatusCode::NOT_FOUND, "research report not found".into()))?;
    let md = research_entry_to_markdown(&entry);
    let filename = format!("{}-research.md", research_topic_slug(&entry.topic));
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/markdown; charset=utf-8"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"{}\"", filename))
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "bad filename".to_string()))?,
    );
    Ok((headers, md))
}

#[derive(Deserialize)]
struct ResearchHistoryDeleteRequest {
    ids: Vec<String>,
}

async fn delete_research_history_bulk(
    Json(payload): Json<ResearchHistoryDeleteRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let _guard = research_history_lock()
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "history locked".to_string()))?;
    let mut list = load_research_history();
    let before = list.len();
    list.retain(|e| !payload.ids.iter().any(|id| id == &e.id));
    let deleted = before - list.len();
    save_json(RESEARCH_HISTORY_FILE, &list)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(serde_json::json!({"deleted": deleted})))
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(h) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(h);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn html_entity_decode(s: &str) -> String {
    let mut out = s
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ");
    // Numeric entities: &#123; and &#x1F;
    loop {
        let start = match out.find("&#") {
            Some(p) => p,
            None => break,
        };
        let rest = &out[start + 2..];
        let semi = match rest.find(';') {
            Some(p) if p <= 8 => p,
            _ => break,
        };
        let num = &rest[..semi];
        let ch: Option<char> = if let Some(hex) = num.strip_prefix("x").or_else(|| num.strip_prefix("X")) {
            u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
        } else {
            num.parse::<u32>().ok().and_then(char::from_u32)
        };
        match ch {
            Some(c) => {
                out.replace_range(start..start + 2 + semi + 1, &c.to_string());
            }
            None => break,
        }
    }
    out
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

fn remove_tag_blocks(html: &str, tag: &str) -> String {
    let lower = html.to_lowercase();
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    let mut rest_lower = lower.as_str();
    loop {
        match rest_lower.find(&open) {
            Some(start) => {
                out.push_str(&rest[..start]);
                match rest_lower[start..].find(&close) {
                    Some(end) => {
                        let skip = start + end + close.len();
                        rest = &rest[skip..];
                        rest_lower = &rest_lower[skip..];
                    }
                    None => break,
                }
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }
    out
}

// Plain text from a fetched HTML page: drop scripts/styles, turn block
// boundaries into spaces, strip tags, decode entities, collapse whitespace.
fn html_to_text(html: &str) -> String {
    let mut t = remove_tag_blocks(html, "script");
    t = remove_tag_blocks(&t, "style");
    t = remove_tag_blocks(&t, "noscript");
    for tag in ["</p>", "</div>", "</li>", "</h1>", "</h2>", "</h3>", "</h4>", "</tr>", "<br>", "<br/>", "</article>", "</section>"] {
        t = t.replace(tag, "\n");
    }
    let t = strip_tags(&t);
    let t = html_entity_decode(&t);
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn research_stopwords() -> &'static [&'static str] {
    &[
        "what", "why", "how", "when", "where", "which", "about", "with", "from", "that", "this",
        "there", "their", "they", "them", "then", "than", "into", "over", "under", "between",
        "does", "does", "been", "being", "have", "has", "will", "would", "should", "could",
        "your", "yours", "more", "most", "some", "such", "only", "also", "very", "much", "many",
    ]
}

// Keywords for passage scoring: words of 4+ letters from the topic and the
// search queries, minus stopwords, deduplicated.
fn research_keywords(topic: &str, extra: &[String]) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut push = |text: &str| {
        for w in text.split(|c: char| !c.is_alphanumeric()) {
            let w = w.to_lowercase();
            if w.len() >= 4 && !research_stopwords().contains(&w.as_str()) && !words.contains(&w) {
                words.push(w);
            }
        }
    };
    push(topic);
    for e in extra {
        push(e);
    }
    words.truncate(24);
    words
}

// Keep the sentences most relevant to the keywords, up to max_chars,
// in their original order. Falls back to the start of the text.
fn extract_relevant(text: &str, keywords: &[String], max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let mut sentences: Vec<(usize, String, usize)> = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c == '.' || c == '!' || c == '?' {
            let end = i + c.len_utf8();
            let s = text[start..end].trim().to_string();
            if s.len() > 20 {
                let low = s.to_lowercase();
                let score = keywords.iter().filter(|k| low.contains(k.as_str())).count();
                sentences.push((sentences.len(), s, score));
            }
            start = end;
        }
    }
    if sentences.is_empty() {
        return text.chars().take(max_chars).collect();
    }
    let mut by_score: Vec<(usize, String, usize)> = sentences.clone();
    by_score.sort_by(|a, b| b.2.cmp(&a.2));
    let mut picked: Vec<(usize, String)> = Vec::new();
    let mut used = 0usize;
    for (idx, s, score) in by_score {
        if score == 0 && !picked.is_empty() {
            break;
        }
        if used + s.len() > max_chars && !picked.is_empty() {
            continue;
        }
        used += s.len() + 1;
        picked.push((idx, s));
        if used >= max_chars {
            break;
        }
    }
    if picked.is_empty() {
        return text.chars().take(max_chars).collect();
    }
    picked.sort_by_key(|(idx, _)| *idx);
    picked
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>()
        .join(" ")
}

// Parse one Bing result block (<li class="b_algo">...</li>): the <h2> link
// is the result. Returns (title, url) or None.
fn bing_parse_block(block: &str) -> Option<(String, String)> {
    // Bing result links look like:
    //   <a class="tilk" ... href="https://example.com/page" ...><h2>Title</h2></a>
    // The href is not necessarily the first attribute, and the <h2> sits
    // INSIDE the anchor — so find the first <a> tag with an absolute href
    // and take its text (tags stripped) as the title.
    let mut rest = block;
    loop {
        let a = rest.find("<a ")?;
        let astart = &rest[a..];
        let tag_end = astart.find('>')?;
        let tag = &astart[..tag_end];
        if let Some(hp) = tag.find("href=\"") {
            let href = tag[hp + 6..].split('"').next()?;
            if href.starts_with("http") {
                let text_start = &astart[tag_end + 1..];
                let text_end = text_start.find("</a>")?;
                let title = html_entity_decode(&strip_tags(&text_start[..text_end]));
                let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
                if !title.is_empty() {
                    return Some((title, href.to_string()));
                }
                return None;
            }
        }
        // This <a> has no usable href — try the next one in the block.
        rest = &astart[tag_end + 1..];
    }
}

// Parse DuckDuckGo HTML results: <a ... class="result__a" href="...">title</a>.
// The href is a //duckduckgo.com/l/?uddg=<urlencoded> redirect; decode it.
fn ddg_parse_results(html: &str) -> Vec<SearchResult> {
    let mut out: Vec<SearchResult> = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find("result__a") {
        let chunk = &rest[pos..];
        let href = match chunk.find("href=\"") {
            Some(p) => &chunk[p + 6..],
            None => break,
        };
        let href = match href.split('"').next() {
            Some(h) => h,
            None => break,
        };
        let gt = match chunk.find('>') {
            Some(p) => p,
            None => break,
        };
        let after = &chunk[gt + 1..];
        let end = match after.find("</a>") {
            Some(p) => p,
            None => break,
        };
        let title = html_entity_decode(&strip_tags(&after[..end]));
        let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
        let url = if href.contains("uddg=") {
            let enc = href.split("uddg=").nth(1).unwrap_or("");
            let enc = enc.split('&').next().unwrap_or(enc);
            url_decode(enc)
        } else if href.starts_with("//") {
            format!("https:{}", href)
        } else {
            href.to_string()
        };
        if !title.is_empty() && url.starts_with("http") && !out.iter().any(|r: &SearchResult| r.url == url) {
            out.push(SearchResult { title, url });
        }
        rest = after;
        if out.len() >= 5 {
            break;
        }
    }
    out
}

fn http_client(secs: u64) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(secs))
        .user_agent(RESEARCH_UA)
        .build()
        .map_err(|e| e.to_string())
}

// IPv4-only client for PDF image fetches: upload.wikimedia.org resolves to
// an IPv6 address unreachable from this phone, and reqwest tries IPv6 first
// and stalls past the timeout. Binding local address 0.0.0.0 forces IPv4.
fn http_client_ipv4(secs: u64) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(secs))
        .user_agent(RESEARCH_UA)
        .local_address(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED))
        .build()
        .map_err(|e| e.to_string())
}

// Real web search, keyless. Primary: Bing HTML (verified working from the
// phone — DuckDuckGo HTML returns a 202 bot-challenge page from this IP).
// Fallback: DuckDuckGo HTML, in case Bing fails.
async fn web_search_bing(query: &str, keywords: &[String]) -> Result<Vec<SearchResult>, String> {
    let client = http_client(20)?;
    let url = format!(
        "https://www.bing.com/search?q={}&count=10",
        url_encode(query)
    );
    let html = client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let mut out: Vec<SearchResult> = Vec::new();
    let mut blocks = 0;
    for part in html.split("<li class=\"b_algo\"").skip(1) {
        blocks += 1;
        let block = part.split("</li>").next().unwrap_or(part);
        if let Some((title, url)) = bing_parse_block(block) {
            if !out.iter().any(|r: &SearchResult| r.url == url) {
                out.push(SearchResult { title, url });
            }
        }
        if out.len() >= 5 {
            break;
        }
    }
    let out = research_filter_relevant(out, keywords);
    if out.is_empty() {
        if blocks > 0 {
            return Err(format!(
                "Bing returned {} result blocks but none were relevant",
                blocks
            ));
        }
        return Err("Bing returned no results".into());
    }
    Ok(out)
}

async fn web_search_ddg(query: &str, keywords: &[String]) -> Result<Vec<SearchResult>, String> {
    let client = http_client(20)?;
    let url = format!(
        "https://html.duckduckgo.com/html/?q={}",
        url_encode(query)
    );
    let html = client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let out = research_filter_relevant(ddg_parse_results(&html), keywords);
    if out.is_empty() {
        return Err("DuckDuckGo returned no relevant results".into());
    }
    Ok(out)
}

// Parse Google HTML results: result links look like
//   <a href="/url?q=https://example.com/page&sa=..."><h3>Title</h3></a>
// (some layouts link directly: <a href="https://example.com/page"><h3>Title</h3></a>).
// Google currently serves this phone a JS redirect page instead of results
// (verified Nov 2026: "Please click here if you are not redirected"), so
// this parser usually finds nothing and the run falls back to Bing — the
// parser is here for when/if Google serves real results.
fn google_parse_results(html: &str) -> Vec<SearchResult> {
    let mut out: Vec<SearchResult> = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find("<a ") {
        let chunk = &rest[pos..];
        let tag_end = match chunk.find('>') {
            Some(p) => p,
            None => break,
        };
        let tag = &chunk[..tag_end];
        let href = match tag.find("href=\"") {
            Some(p) => &tag[p + 6..],
            None => {
                rest = &chunk[tag_end + 1..];
                continue;
            }
        };
        let href = match href.split('"').next() {
            Some(h) => h,
            None => break,
        };
        let url = if href.starts_with("/url?q=") {
            let enc = &href[7..];
            let enc = enc.split('&').next().unwrap_or(enc);
            url_decode(enc)
        } else if href.starts_with("http") {
            href.to_string()
        } else {
            rest = &chunk[tag_end + 1..];
            continue;
        };
        let after = &chunk[tag_end + 1..];
        let title = if let Some(h3) = after.find("<h3") {
            let h3c = &after[h3..];
            let h3gt = match h3c.find('>') {
                Some(p) => p,
                None => break,
            };
            let h3after = &h3c[h3gt + 1..];
            let hend = match h3after.find("</h3>") {
                Some(p) => p,
                None => break,
            };
            html_entity_decode(&strip_tags(&h3after[..hend]))
        } else {
            String::new()
        };
        let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
        if !title.is_empty() && url.starts_with("http") && !out.iter().any(|r: &SearchResult| r.url == url) {
            out.push(SearchResult { title, url });
        }
        rest = after;
        if out.len() >= 5 {
            break;
        }
    }
    out
}

async fn web_search_google(query: &str, keywords: &[String]) -> Result<Vec<SearchResult>, String> {
    let client = http_client(20)?;
    let url = format!(
        "https://www.google.com/search?q={}&num=10",
        url_encode(query)
    );
    let html = client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    // Google serves a JS redirect/consent page to this phone instead of
    // results — detect it so the fallback note says so plainly.
    if html.contains("if you are not redirected") {
        return Err("Google blocked the request (redirect page, no results)".into());
    }
    let out = research_filter_relevant(google_parse_results(&html), keywords);
    if out.is_empty() {
        return Err("Google returned no relevant results".into());
    }
    Ok(out)
}

// Parse Marginalia Search HTML results: result links are plain direct
// anchors, e.g. <a href="https://en.wikipedia.org/wiki/Rayleigh_scattering">Rayleigh scattering</a>.
// Marginalia is an independent index (verified working from the phone Nov
// 2026). Its page also carries nav links (GitHub, Discord, old-search) —
// the relevance filter drops those, and we skip its own hosts outright.
fn marginalia_parse_results(html: &str) -> Vec<SearchResult> {
    let mut out: Vec<SearchResult> = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find("<a ") {
        let chunk = &rest[pos..];
        let tag_end = match chunk.find('>') {
            Some(p) => p,
            None => break,
        };
        let tag = &chunk[..tag_end];
        let href = match tag.find("href=\"") {
            Some(p) => &tag[p + 6..],
            None => {
                rest = &chunk[tag_end + 1..];
                continue;
            }
        };
        let href = match href.split('"').next() {
            Some(h) => h,
            None => break,
        };
        if !href.starts_with("http") {
            rest = &chunk[tag_end + 1..];
            continue;
        }
        // Skip Marginalia's own nav/chrome links.
        if href.contains("marginalia.nu") || href.contains("github.com/MarginaliaSearch") {
            rest = &chunk[tag_end + 1..];
            continue;
        }
        let after = &chunk[tag_end + 1..];
        let end = match after.find("</a>") {
            Some(p) => p,
            None => break,
        };
        let title = html_entity_decode(&strip_tags(&after[..end]));
        let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
        if !title.is_empty() && !out.iter().any(|r: &SearchResult| r.url == href) {
            out.push(SearchResult { title, url: href.to_string() });
        }
        rest = after;
        if out.len() >= 5 {
            break;
        }
    }
    out
}

async fn web_search_marginalia(query: &str, keywords: &[String]) -> Result<Vec<SearchResult>, String> {
    let client = http_client(20)?;
    let url = format!(
        "https://search.marginalia.nu/search?query={}",
        url_encode(query)
    );
    let html = client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let out = research_filter_relevant(marginalia_parse_results(&html), keywords);
    if out.is_empty() {
        return Err("Marginalia returned no relevant results".into());
    }
    Ok(out)
}

// Relevance guard: keyless HTML search from this phone is degraded —
// Bing intermittently serves login pages and dictionary entries instead
// of real results (verified Nov 2026: TikTok/Google login pages and
// "why" dictionary entries for a Rayleigh-scattering query; Brave 429s
// and Ecosia 403s from this IP too). Drop results that share no content
// keyword with the topic/queries, and skip login/social hosts outright,
// so a run never synthesises a report from junk pages.
fn research_blocked_result(url: &str) -> bool {
    let u = url.to_lowercase();
    ["tiktok.com", "whatsapp.com", "accounts.google.com", "facebook.com", "instagram.com", "twitter.com", "linkedin.com"]
        .iter()
        .any(|h| u.contains(h))
        || u.contains("/login")
        || u.contains("/signup")
}

fn research_filter_relevant(results: Vec<SearchResult>, keywords: &[String]) -> Vec<SearchResult> {
    results
        .into_iter()
        .filter(|r| {
            if research_blocked_result(&r.url) {
                return false;
            }
            if keywords.is_empty() {
                return true;
            }
            let hay = format!("{} {}", r.title, r.url).to_lowercase();
            keywords.iter().any(|k| !k.is_empty() && hay.contains(k.as_str()))
        })
        .collect()
}

fn parse_wikipedia_results(json: &str) -> Vec<SearchResult> {
    let v: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    if let Some(hits) = v
        .get("query")
        .and_then(|q| q.get("search"))
        .and_then(|s| s.as_array())
    {
        for h in hits {
            if let Some(title) = h.get("title").and_then(|t| t.as_str()) {
                let url = format!(
                    "https://en.wikipedia.org/wiki/{}",
                    url_encode(&title.replace(' ', "_"))
                );
                out.push(SearchResult {
                    title: title.to_string(),
                    url,
                });
            }
        }
    }
    out
}

async fn web_search_wikipedia(query: &str, keywords: &[String]) -> Result<Vec<SearchResult>, String> {
    // The MediaWiki search API still answers from this phone when the
    // HTML engines are degraded. Factual topics get real articles; the
    // fetcher then reads the article text like any other page.
    let client = http_client(20)?;
    let url = format!(
        "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json&srlimit=5",
        url_encode(query)
    );
    let body = client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let out = research_filter_relevant(parse_wikipedia_results(&body), keywords);
    if out.is_empty() {
        return Err("Wikipedia returned no relevant results".into());
    }
    Ok(out)
}
// - ROUNDS: Odysseus DeepResearcher default is max_rounds=8. The model's
//   stop-decision normally ends a run after 2 rounds (min_rounds=2), so the
//   cap rarely matters; the run is also bounded by the 6-minute deadline.
//   Clamped to 1..=8.
// - FORMAT: Odysseus has no format selector — it always writes the full
//   structured report, so the default is "full".
// - SEARCH ENGINE: the default is SearXNG — a dedicated instance
//   installed for Mawaxi AI on this phone (port 8888, separate from
//   Odysseus; see /root/searxng-config). It aggregates many engines
//   server-side and is the closest match to Odysseus, which defaults to
//   its own SearXNG (http://localhost:8080 in its .env — dead on this
//   phone, that port is this server's own). If the local SearXNG is
//   down, the chain falls back to Bing with a note. Google is listed
//   but Google blocks this phone (JS redirect page), so it falls back
//   to Bing too. Marginalia is an independent index that works from
//   the phone. Note Bing itself is degraded from this phone as of Nov
//   2026 (it serves login pages and dictionary entries for real
//   queries); the search chain therefore filters results for relevance
//   and falls back to the Wikipedia API, which still answers.
//   DuckDuckGo HTML is bot-blocked from this phone (HTTP 202); Brave
//   needs an API key and none is configured in Mawaxi AI, so it falls
//   back to Bing.
fn research_rounds_choice(rounds: Option<u32>) -> usize {
    match rounds {
        Some(n) => (n as usize).clamp(1, 8),
        None => 8,
    }
}

// Parse a SearXNG JSON search response ({"results": [{title, url, ...}]}).
// Used by the dedicated Mawaxi AI SearXNG instance (local, port 8888).
fn parse_searxng_results(json: &str) -> Vec<SearchResult> {
    let v: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<SearchResult> = Vec::new();
    if let Some(results) = v.get("results").and_then(|r| r.as_array()) {
        for r in results {
            let title = r.get("title").and_then(|t| t.as_str()).unwrap_or("").to_string();
            let url = r.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string();
            if !title.is_empty() && url.starts_with("http") && !out.iter().any(|x: &SearchResult| x.url == url) {
                out.push(SearchResult { title, url });
            }
            if out.len() >= 5 {
                break;
            }
        }
    }
    out
}

// Search via the dedicated Mawaxi AI SearXNG instance (local, port 8888).
// SearXNG aggregates many engines server-side — its DuckDuckGo engine
// works where direct DDG HTML scraping is bot-blocked — and returns JSON.
// If the instance is down, the caller falls back to Bing with a note.
async fn web_search_searxng(query: &str, keywords: &[String]) -> Result<Vec<SearchResult>, String> {
    let client = http_client(20)?;
    let url = format!(
        "http://127.0.0.1:8888/search?q={}&format=json&language=en",
        url_encode(query)
    );
    let body = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("SearXNG instance not reachable: {}", e))?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let out = research_filter_relevant(parse_searxng_results(&body), keywords);
    if out.is_empty() {
        return Err("SearXNG returned no relevant results".into());
    }
    Ok(out)
}

// One real web image result: direct image URL, thumbnail, title, source page.
#[derive(Serialize, Clone)]
struct ImageResult {
    image_url: String,
    thumbnail_url: String,
    title: String,
    source_url: String,
}

// Parse a SearXNG JSON image search response (categories=images):
// results carry img_src (full image URL), thumbnail_src, title, url (source page).
fn parse_searxng_image_results(json: &str) -> Vec<ImageResult> {
    let v: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<ImageResult> = Vec::new();
    if let Some(results) = v.get("results").and_then(|r| r.as_array()) {
        for r in results {
            let image_url = r.get("img_src").and_then(|u| u.as_str()).unwrap_or("").to_string();
            let thumbnail_url = r.get("thumbnail_src").and_then(|u| u.as_str()).unwrap_or("").to_string();
            let title = r.get("title").and_then(|t| t.as_str()).unwrap_or("").to_string();
            let source_url = r.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string();
            if image_url.starts_with("http") && !out.iter().any(|x: &ImageResult| x.image_url == image_url) {
                out.push(ImageResult { image_url, thumbnail_url, title, source_url });
            }
            if out.len() >= 5 {
                break;
            }
        }
    }
    out
}

// Search real web images via the dedicated Mawaxi AI SearXNG instance
// (local, port 8888), image category. If it returns nothing useful,
// fall back to a general web search and keep only direct image links.
async fn search_web_images(query: &str) -> Result<Vec<ImageResult>, String> {
    let client = http_client(15)?;
    let url = format!(
        "http://127.0.0.1:8888/search?q={}&format=json&language=en&categories=images",
        url_encode(query)
    );
    let body = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("SearXNG instance not reachable: {}", e))?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let out = parse_searxng_image_results(&body);
    if !out.is_empty() {
        return Ok(out);
    }
    let web = web_search_searxng(query, &[]).await?;
    let mut imgs: Vec<ImageResult> = Vec::new();
    for r in web {
        let lower = r.url.to_lowercase();
        let path = lower.split(['?', '#']).next().unwrap_or("");
        if path.ends_with(".jpg")
            || path.ends_with(".jpeg")
            || path.ends_with(".png")
            || path.ends_with(".webp")
            || path.ends_with(".gif")
        {
            if !imgs.iter().any(|x: &ImageResult| x.image_url == r.url) {
                imgs.push(ImageResult {
                    image_url: r.url.clone(),
                    thumbnail_url: String::new(),
                    title: r.title.clone(),
                    source_url: r.url.clone(),
                });
            }
        }
        if imgs.len() >= 5 {
            break;
        }
    }
    if imgs.is_empty() {
        return Err("No images found for that query".into());
    }
    Ok(imgs)
}

fn research_engine_choice(engine: &str) -> &'static str {
    // Default is SearXNG: the dedicated instance installed for Mawaxi AI
    // on this phone (port 8888, separate from Odysseus). It aggregates
    // many engines server-side and is the closest match to Odysseus.
    match engine.trim().to_lowercase().as_str() {
        "duckduckgo" | "ddg" => "duckduckgo",
        "brave" => "brave",
        "google" => "google",
        "marginalia" => "marginalia",
        "bing" => "bing",
        _ => "searxng",
    }
}

fn research_format_prompt(format: &str) -> &'static str {
    match format.trim().to_lowercase().as_str() {
        "brief" => RESEARCH_FINAL_PROMPT_BRIEF,
        "bullets" => RESEARCH_FINAL_PROMPT_BULLETS,
        _ => RESEARCH_FINAL_PROMPT,
    }
}

async fn web_search(query: &str, engine: &str, keywords: &[String]) -> Result<(Vec<SearchResult>, Option<String>), String> {
    // Chain: chosen engine first, then the others, then Wikipedia (the
    // MediaWiki API still answers from this phone when the HTML engines
    // are degraded). Each step only counts if it yields a relevant hit.
    // The Option<String> is a note for the progress log when the chosen
    // engine failed and a fallback took over (e.g. Google is blocked
    // from this phone, so it falls back to Bing with a note).
    if engine == "searxng" {
        return match web_search_searxng(query, keywords).await {
            Ok(r) => Ok((r, None)),
            Err(e1) => match web_search_bing(query, keywords).await {
                Ok(r) => Ok((r, Some(format!("SearXNG is not answering ({}) — using Bing instead.", e1)))),
                Err(e2) => match web_search_wikipedia(query, keywords).await {
                    Ok(r) => Ok((r, Some(format!("SearXNG is not answering ({}); Bing failed ({}) — using Wikipedia instead.", e1, e2)))),
                    Err(e3) => Err(format!(
                        "search failed (SearXNG: {}; Bing fallback: {}; Wikipedia: {})",
                        e1, e2, e3
                    )),
                },
            },
        };
    }
    if engine == "google" {
        return match web_search_google(query, keywords).await {
            Ok(r) => Ok((r, None)),
            Err(e1) => match web_search_bing(query, keywords).await {
                Ok(r) => Ok((r, Some(format!("Google blocked the request ({}) — using Bing instead.", e1)))),
                Err(e2) => match web_search_ddg(query, keywords).await {
                    Ok(r) => Ok((r, Some(format!("Google blocked the request ({}); Bing also failed ({}) — using DuckDuckGo instead.", e1, e2)))),
                    Err(e3) => match web_search_wikipedia(query, keywords).await {
                        Ok(r) => Ok((r, Some(format!("Google blocked the request ({}); Bing and DuckDuckGo failed — using Wikipedia instead.", e1)))),
                        Err(e4) => Err(format!(
                            "search failed (Google: {}; Bing fallback: {}; DDG: {}; Wikipedia: {})",
                            e1, e2, e3, e4
                        )),
                    },
                },
            },
        };
    }
    if engine == "marginalia" {
        return match web_search_marginalia(query, keywords).await {
            Ok(r) => Ok((r, None)),
            Err(e1) => match web_search_bing(query, keywords).await {
                Ok(r) => Ok((r, Some(format!("Marginalia returned no results ({}) — using Bing instead.", e1)))),
                Err(e2) => match web_search_wikipedia(query, keywords).await {
                    Ok(r) => Ok((r, Some(format!("Marginalia returned no results ({}); Bing failed ({}) — using Wikipedia instead.", e1, e2)))),
                    Err(e3) => Err(format!(
                        "search failed (Marginalia: {}; Bing fallback: {}; Wikipedia: {})",
                        e1, e2, e3
                    )),
                },
            },
        };
    }
    if engine == "duckduckgo" {
        // DuckDuckGo HTML is often bot-blocked from this phone; if it
        // fails, fall back to Bing, then Wikipedia, rather than failing.
        return match web_search_ddg(query, keywords).await {
            Ok(r) => Ok((r, None)),
            Err(e1) => match web_search_bing(query, keywords).await {
                Ok(r) => Ok((r, Some(format!("DuckDuckGo blocked the request ({}) — using Bing instead.", e1)))),
                Err(e2) => match web_search_wikipedia(query, keywords).await {
                    Ok(r) => Ok((r, Some(format!("DuckDuckGo blocked the request ({}); Bing failed ({}) — using Wikipedia instead.", e1, e2)))),
                    Err(e3) => Err(format!(
                        "search failed (DDG: {}; Bing fallback: {}; Wikipedia: {})",
                        e1, e2, e3
                    )),
                },
            },
        };
    }
    // Bing (also the fallback target when Brave was picked with no key).
    // Bing intermittently serves a page with no results (bot-check); one
    // retry after a short pause usually gets a clean page.
    match web_search_bing(query, keywords).await {
        Ok(r) => Ok((r, None)),
        Err(e1) => {
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            match web_search_bing(query, keywords).await {
                Ok(r) => Ok((r, None)),
                Err(e2) => match web_search_ddg(query, keywords).await {
                    Ok(r) => Ok((r, Some(format!("Bing returned no results ({}); Bing retry failed — using DuckDuckGo instead.", e1)))),
                    Err(e3) => match web_search_wikipedia(query, keywords).await {
                        Ok(r) => Ok((r, Some("Bing and DuckDuckGo returned no results — using Wikipedia instead.".to_string()))),
                        Err(e4) => Err(format!(
                            "search failed (Bing: {}; Bing retry: {}; DDG: {}; Wikipedia: {})",
                            e1, e2, e3, e4
                        )),
                    },
                },
            }
        }
    }
}
fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

// Fetch one page and return (final URL after redirects, extracted text).
// Failures are the caller's problem to skip, never fatal to the run.
async fn fetch_page(url: &str, keywords: &[String]) -> Result<(String, String), String> {
    let client = http_client(15)?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let final_url = resp.url().to_string();
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    let capped: Vec<u8> = bytes.iter().take(400_000).cloned().collect();
    let html = String::from_utf8_lossy(&capped).into_owned();
    let text = html_to_text(&html);
    if text.len() < 200 {
        return Err("page had almost no readable text".into());
    }
    Ok((final_url, extract_relevant(&text, keywords, 3000)))
}

fn parse_plan_json(text: &str, topic: &str) -> ResearchPlan {
    if let (Some(s), Some(e)) = (text.find('{'), text.rfind('}')) {
        if e > s {
            if let Ok(p) = serde_json::from_str::<ResearchPlan>(&text[s..=e]) {
                if !p.sub_questions.is_empty() {
                    return p;
                }
            }
        }
    }
    ResearchPlan {
        sub_questions: vec![
            format!("What is {}?", topic),
            format!("Why does {} matter?", topic),
            format!("What are the latest developments in {}?", topic),
        ],
        key_topics: vec![topic.to_string()],
        success_criteria: format!("A thorough, source-backed answer about {}.", topic),
    }
}

fn parse_queries_json(text: &str, fallback: &[String]) -> Vec<String> {
    if let (Some(s), Some(e)) = (text.find('['), text.rfind(']')) {
        if e > s {
            if let Ok(q) = serde_json::from_str::<Vec<String>>(&text[s..=e]) {
                let qs: Vec<String> = q
                    .into_iter()
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty())
                    .collect();
                if !qs.is_empty() {
                    return qs;
                }
            }
        }
    }
    fallback.to_vec()
}

#[derive(Serialize)]
struct WsFileContent {
    path: String,
    content: String,
}

/// Provider presets for the Add Endpoint form: picking one fills in the
/// base URL and model, so the user only types the API key. "custom" is
/// blank on purpose. No Shannon preset: its base URL is not verified.
#[derive(Serialize)]
struct EndpointPreset {
    provider: &'static str,
    label: &'static str,
    base_url: &'static str,
    model: &'static str,
}

const ENDPOINT_PRESETS: &[EndpointPreset] = &[
    EndpointPreset { provider: "gemini", label: "Gemini", base_url: "https://generativelanguage.googleapis.com/v1beta/openai/", model: "gemini-2.5-flash" },
    EndpointPreset { provider: "nvidia", label: "NVIDIA", base_url: "https://integrate.api.nvidia.com/v1", model: "deepseek-v4.1-flash" },
    EndpointPreset { provider: "openai", label: "OpenAI", base_url: "https://api.openai.com/v1", model: "gpt-4o" },
    EndpointPreset { provider: "copilot", label: "GitHub Copilot", base_url: "https://api.githubcopilot.com", model: "" },
    EndpointPreset { provider: "custom", label: "Custom", base_url: "", model: "" },
];

fn preset_for(provider: &str) -> Option<&'static EndpointPreset> {
    ENDPOINT_PRESETS.iter().find(|p| p.provider == provider)
}

async fn list_endpoint_presets() -> Json<&'static [EndpointPreset]> {
    Json(ENDPOINT_PRESETS)
}

// ---------------------------------------------------------------------------
// GitHub Copilot (the same option as Odysseus).
// Copilot is an OpenAI-compatible API at https://api.githubcopilot.com, but it
// needs Copilot-specific headers and a GitHub OAuth token (from the device
// flow) used directly as the Bearer token. The token is fetched by the device
// flow below, never typed by hand.
// ---------------------------------------------------------------------------
const COPILOT_CLIENT_ID: &str = "01ab8ac9400c4e429b23"; // public VS Code client id (allow-listed for Copilot)
const COPILOT_BASE: &str = "https://api.githubcopilot.com";
const COPILOT_API_VERSION: &str = "2026-06-01";
const COPILOT_INTEGRATION_ID: &str = "vscode-chat";
const COPILOT_USER_AGENT: &str = "MawaxiAI/1.0";
const COPILOT_EDITOR_VERSION: &str = "MawaxiAI/1.0";
const COPILOT_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const COPILOT_ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

fn is_copilot_base(base_url: &str) -> bool {
    let lower = base_url.trim().to_lowercase();
    let host = lower
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    let host = host.split('/').next().unwrap_or("");
    host == "api.githubcopilot.com" || host.ends_with(".githubcopilot.com")
}

/// Copilot-specific headers for a chat/models request. The Bearer token is
/// set separately with .bearer_auth(). `agent` marks agent-loop calls,
/// `vision` marks requests carrying an image.
fn with_copilot_headers(
    req: reqwest::RequestBuilder,
    agent: bool,
    vision: bool,
) -> reqwest::RequestBuilder {
    let mut r = req
        .header("X-GitHub-Api-Version", COPILOT_API_VERSION)
        .header("Openai-Intent", "conversation-edits")
        .header("User-Agent", COPILOT_USER_AGENT)
        .header("Editor-Version", COPILOT_EDITOR_VERSION)
        .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
        .header("x-initiator", if agent { "agent" } else { "user" });
    if vision {
        r = r.header("Copilot-Vision-Request", "true");
    }
    r
}

fn messages_have_images(messages: &[OllamaMessage]) -> bool {
    messages
        .iter()
        .any(|m| m.images.as_ref().map(|v| !v.is_empty()).unwrap_or(false))
}

struct CopilotPending {
    device_code: String,
}

static COPILOT_PENDING: OnceLock<Mutex<HashMap<String, CopilotPending>>> = OnceLock::new();

fn copilot_pending() -> &'static Mutex<HashMap<String, CopilotPending>> {
    COPILOT_PENDING.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Serialize)]
struct CopilotDeviceStartResponse {
    poll_id: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: String,
    interval: u64,
    expires_in: u64,
}

/// Start the GitHub device flow: ask GitHub for a user code. The browser
/// shows the code and a link; Steven authorises it on github.com, and the
/// poll route below picks up the token. The device_code stays server-side.
async fn copilot_device_start(
) -> Result<Json<CopilotDeviceStartResponse>, (StatusCode, String)> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|_| (StatusCode::BAD_GATEWAY, "could not start request".to_string()))?;
    let body = format!("client_id={}&scope=read:user", COPILOT_CLIENT_ID);
    let resp = client
        .post(COPILOT_DEVICE_CODE_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .body(body)
        .send()
        .await
        .map_err(|_| (StatusCode::BAD_GATEWAY, "could not reach GitHub".to_string()))?;
    if !resp.status().is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("GitHub returned HTTP {}", resp.status().as_u16()),
        ));
    }
    let v: serde_json::Value = resp
        .json()
        .await
        .map_err(|_| (StatusCode::BAD_GATEWAY, "GitHub did not return a code".to_string()))?;
    let device_code = v
        .get("device_code")
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .to_string();
    let user_code = v
        .get("user_code")
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .to_string();
    if device_code.is_empty() || user_code.is_empty() {
        return Err((StatusCode::BAD_GATEWAY, "GitHub did not return a code".to_string()));
    }
    let poll_id = gen_id();
    if let Ok(mut m) = copilot_pending().lock() {
        m.insert(poll_id.clone(), CopilotPending { device_code });
    }
    Ok(Json(CopilotDeviceStartResponse {
        poll_id,
        user_code,
        verification_uri: v
            .get("verification_uri")
            .and_then(|d| d.as_str())
            .unwrap_or("https://github.com/login/device")
            .to_string(),
        verification_uri_complete: v
            .get("verification_uri_complete")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string(),
        interval: v.get("interval").and_then(|d| d.as_u64()).unwrap_or(5),
        expires_in: v.get("expires_in").and_then(|d| d.as_u64()).unwrap_or(900),
    }))
}

#[derive(Deserialize)]
struct CopilotDevicePollRequest {
    poll_id: String,
}

#[derive(Serialize)]
struct CopilotDevicePollResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    models: Option<Vec<String>>,
}

/// Poll GitHub for the device-flow token. On authorisation, save the Copilot
/// endpoint (with its model list) and report it; while Steven has not
/// authorised yet, report pending so the browser keeps polling.
async fn copilot_device_poll(
    Json(payload): Json<CopilotDevicePollRequest>,
) -> Result<Json<CopilotDevicePollResponse>, (StatusCode, String)> {
    let device_code = {
        let m = copilot_pending()
            .lock()
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "lock error".to_string()))?;
        match m.get(&payload.poll_id) {
            Some(p) => p.device_code.clone(),
            None => {
                return Err((
                    StatusCode::NOT_FOUND,
                    "sign-in session not found (it may have expired)".to_string(),
                ))
            }
        }
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|_| (StatusCode::BAD_GATEWAY, "could not start request".to_string()))?;
    let body = format!(
        "client_id={}&device_code={}&grant_type=urn:ietf:params:oauth:grant-type:device_code",
        COPILOT_CLIENT_ID, device_code
    );
    let resp = client
        .post(COPILOT_ACCESS_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .body(body)
        .send()
        .await
        .map_err(|_| (StatusCode::BAD_GATEWAY, "could not reach GitHub".to_string()))?;
    if !resp.status().is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("GitHub returned HTTP {}", resp.status().as_u16()),
        ));
    }
    let v: serde_json::Value = resp
        .json()
        .await
        .map_err(|_| (StatusCode::BAD_GATEWAY, "GitHub did not return a reply".to_string()))?;
    if let Some(token) = v.get("access_token").and_then(|t| t.as_str()) {
        if !token.is_empty() {
            if let Ok(mut m) = copilot_pending().lock() {
                m.remove(&payload.poll_id);
            }
            let (endpoint_id, models) = provision_copilot_endpoint(token)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
            return Ok(Json(CopilotDevicePollResponse {
                status: "authorized".to_string(),
                error: None,
                endpoint_id: Some(endpoint_id),
                models: Some(models),
            }));
        }
    }
    let err = v
        .get("error")
        .and_then(|e| e.as_str())
        .unwrap_or("")
        .to_string();
    match err.as_str() {
        "authorization_pending" | "slow_down" | "" => Ok(Json(CopilotDevicePollResponse {
            status: "pending".to_string(),
            error: None,
            endpoint_id: None,
            models: None,
        })),
        "expired_token" | "access_denied" => {
            if let Ok(mut m) = copilot_pending().lock() {
                m.remove(&payload.poll_id);
            }
            Ok(Json(CopilotDevicePollResponse {
                status: "failed".to_string(),
                error: Some(err),
                endpoint_id: None,
                models: None,
            }))
        }
        _ => Ok(Json(CopilotDevicePollResponse {
            status: "pending".to_string(),
            error: None,
            endpoint_id: None,
            models: None,
        })),
    }
}

/// Create (or refresh) the GitHub Copilot endpoint from a device-flow token,
/// fetch its model list, and pick a default model. Returns (endpoint_id, models).
async fn provision_copilot_endpoint(token: &str) -> Result<(String, Vec<String>), String> {
    let mut eps: Vec<Endpoint> = load_json(ENDPOINTS_FILE);
    let id = match eps
        .iter_mut()
        .find(|e| is_copilot_base(&e.base_url) || e.provider == "copilot")
    {
        Some(ep) => {
            ep.api_key = token.to_string();
            ep.id.clone()
        }
        None => {
            let id = gen_id();
            eps.push(Endpoint {
                id: id.clone(),
                name: "GitHub Copilot".to_string(),
                base_url: COPILOT_BASE.to_string(),
                api_key: token.to_string(),
                model: String::new(),
                provider: "copilot".to_string(),
                models: Vec::new(),
                models_fetched_at: 0,
            });
            id
        }
    };
    // Fetch the model catalogue (Copilot headers are added by
    // fetch_endpoint_models when the base is a Copilot base).
    let models = fetch_endpoint_models(COPILOT_BASE, token).await.unwrap_or_default();
    if let Some(ep) = eps.iter_mut().find(|e| e.id == id) {
        ep.models = models.clone();
        ep.models_fetched_at = now_secs();
        if ep.model.is_empty() || !models.contains(&ep.model) {
            ep.model = if models.iter().any(|m| m == "gpt-4o") {
                "gpt-4o".to_string()
            } else {
                models.first().cloned().unwrap_or_default()
            };
        }
    }
    save_json(ENDPOINTS_FILE, &eps).map_err(|e| e.to_string())?;
    Ok((id, models))
}


const ENDPOINTS_FILE: &str = "endpoints.json";
const MEMORIES_FILE: &str = "memories.json";
const SKILLS_FILE: &str = "skills.json";
const SETTINGS_FILE: &str = "settings.json";
const NOTES_FILE: &str = "notes.json";
const HF_API_URL: &str = "https://huggingface.co/api/models";
const OLLAMA_PS_URL: &str = "http://127.0.0.1:11434/api/ps";
const OLLAMA_PULL_URL: &str = "http://127.0.0.1:11434/api/pull";
const OLLAMA_GENERATE_URL: &str = "http://127.0.0.1:11434/api/generate";
const MODELS_DIR: &str = "/root/models";

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn gen_id() -> String {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}", n)
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &str) -> Vec<T> {
    fs::read_to_string(path)
        .ok()
        .and_then(|d| serde_json::from_str(&d).ok())
        .unwrap_or_default()
}

fn save_json<T: Serialize>(path: &str, v: &T) -> Result<(), String> {
    let data = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    fs::write(path, data).map_err(|e| e.to_string())
}

// Real web photos in plain chat (Steven's word, 2026-11-21): when a real
// picture would genuinely help the answer, the model ends its reply with
// one [[WEBIMAGE: short search query]] line and the client fetches real
// photos from the web via POST /api/web-image (SearXNG) — never
// AI-generated. (The old [[IMAGE: ...]] AI-image hint was removed the
// same day for the same reason: a generated image can look like a real
// finding, e.g. for medical/lab content.)
const CHAT_WEB_IMAGE_HINT: &str = "Real photos: when a real picture would genuinely help the answer — the answer is about a visible thing (animal, object, place, medical morphology such as a blood smear, or an example scene) — end your reply with exactly one line: [[WEBIMAGE: short search query]] and nothing else on that line. When no picture would help — greetings, maths, code, abstract concepts, purely textual definitions, lists of facts — omit the marker entirely. At most one image search per reply. These are REAL photos from the web, never AI-generated.";
// Steven's word (2026-11-23): chat replies must not include asterisks.
const CHAT_PLAIN_TEXT_HINT: &str = "Write in plain text. Do not use asterisks (*) anywhere in your reply — no bold, no italic, no asterisk bullets. Use plain words, and a dash (-) for a list item if you need a list.";

const CHAT_PDF_HINT: &str = "PDFs: if the user asks for a PDF, say the app can make one - use the PDF button in the header for a PDF of this chat, or ask again for a PDF of a specific answer. Never say you cannot create PDFs.";

// Personas: selectable reply styles. General adds nothing (today's
// behaviour). The client sends the chosen id as `persona` with every
// chat/agent request; the server injects the matching preset.
const PERSONA_MEDICAL: &str = "You are answering as a senior medical laboratory scientist with many years at the bench. Be precise about morphology, stains, and what to look for on the film or smear; give bench-practical detail in the style of ASCP exam material. Name the key features that separate look-alikes, state honest caveats when a finding is not specific, and end with concrete next steps.";
const PERSONA_TEACHER: &str = "You are a patient teacher. Explain step by step in simple language, starting from the basics before the detail. Use a concrete example, keep sentences short, and check understanding at the end with one short question.";
const PERSONA_CODER: &str = "You are a practical software builder. Answer concisely and lead with the code, commands, or file changes when code helps. Name files and commands concretely and keep explanations short.";

fn persona_prompt(id: &str) -> Option<&'static str> {
    match id {
        "medical" => Some(PERSONA_MEDICAL),
        "teacher" => Some(PERSONA_TEACHER),
        "coder" => Some(PERSONA_CODER),
        _ => None, // "general" or unknown: no extra instruction
    }
}

fn build_system_prompt(skill: Option<&Skill>, memories: &[Memory], persona: Option<&str>, local_model: bool) -> Option<String> {
    // The tiny local model (qwen3-abliterated-0.6b) cannot handle a system
    // prompt at all: with one it echoes the stored memories, spits out
    // [[WEBIMAGE: ...]] markers, or answers in fragments instead of
    // answering (2026-11-24: "Give total of 5 and 6" came back as "- 5
    // and 6", "5 - 6", and once as a dump of the memories). With NO system
    // prompt it answers "11" correctly. So the local model gets none.
    if local_model {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if let Some(p) = persona.and_then(persona_prompt) {
        parts.push(p.to_string());
    }
    // NOTE (Steven's word, 2026-11-21): no AI-generated images in normal
    // chat — a generated image can look like a real finding, e.g. for
    // medical/lab content. The model is never asked to emit [[IMAGE: ...]]
    // markers; the client still strips a stray one as a safety net but
    // never generates an image. Real web photos ARE welcome: the model
    // may end a reply with [[WEBIMAGE: <query>]] (see CHAT_WEB_IMAGE_HINT)
    // and the client fetches real photos from the web — never generated.
    // The tiny local model (qwen3-abliterated-0.6b) cannot follow the
    // [[WEBIMAGE: ...]] marker instruction -- it latches onto the marker
    // syntax and repeats it instead of answering (2026-11-24: "Give total
    // of 5 and 6" came back as a stream of [[WEBIMAGE: ...]] markers).
    // So the hint is only sent to capable cloud models.
    if !local_model {
        parts.push(CHAT_WEB_IMAGE_HINT.to_string());
    }
    parts.push(CHAT_PLAIN_TEXT_HINT.to_string());
    parts.push(CHAT_PDF_HINT.to_string());
    if let Some(s) = skill {
        parts.push(s.instruction.clone());
    }
    if !memories.is_empty() {
        let mut m = String::from("Remember these facts about the user:");
        for mem in memories {
            m.push_str(&format!("\n- {}", mem.text));
        }
        parts.push(m);
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

#[derive(Serialize)]
struct ChatSummary {
    id: String,
    title: String,
    updated_at: u64,
}

#[derive(Serialize)]
struct ModelsResponse {
    models: Vec<String>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn chat_path(id: &str) -> PathBuf {
    PathBuf::from(CHATS_DIR).join(format!("{}.json", id))
}

fn load_chat(id: &str) -> Option<Chat> {
    let data = fs::read_to_string(chat_path(id)).ok()?;
    serde_json::from_str(&data).ok()
}

fn save_chat(chat: &Chat) -> Result<(), String> {
    fs::create_dir_all(CHATS_DIR).map_err(|e| e.to_string())?;
    let data = serde_json::to_string_pretty(chat).map_err(|e| e.to_string())?;
    fs::write(chat_path(&chat.id), data).map_err(|e| e.to_string())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn list_chats() -> Json<Vec<ChatSummary>> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(CHATS_DIR) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            if let Ok(data) = fs::read_to_string(&p) {
                if let Ok(c) = serde_json::from_str::<Chat>(&data) {
                    out.push(ChatSummary {
                        id: c.id,
                        title: c.title,
                        updated_at: c.updated_at,
                    });
                }
            }
        }
    }
    out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Json(out)
}

async fn get_chat(Path(id): Path<String>) -> Result<Json<Chat>, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    match load_chat(&id) {
        Some(c) => Ok(Json(c)),
        None => Err((StatusCode::NOT_FOUND, "chat not found".into())),
    }
}

async fn put_chat(
    Path(id): Path<String>,
    Json(mut chat): Json<Chat>,
) -> Result<Json<Chat>, (StatusCode, String)> {
    if !valid_id(&id) || chat.id != id {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    chat.updated_at = now_secs();
    save_chat(&chat).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(chat))
}

async fn delete_chat(Path(id): Path<String>) -> Result<StatusCode, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let p = chat_path(&id);
    if p.exists() {
        fs::remove_file(p).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn list_models() -> Json<ModelsResponse> {
    let client = reqwest::Client::new();
    let mut models: Vec<String> = Vec::new();
    if let Ok(resp) = client.get(OLLAMA_TAGS_URL).send().await {
        if let Ok(v) = resp.json::<serde_json::Value>().await {
            if let Some(arr) = v.get("models").and_then(|m| m.as_array()) {
                for m in arr {
                    if let Some(name) = m.get("name").and_then(|n| n.as_str()) {
                        models.push(name.to_string());
                    }
                }
            }
        }
    }
    if models.is_empty() {
        models.push(DEFAULT_MODEL.to_string());
    }
    Json(ModelsResponse { models })
}

fn percent_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

#[derive(Deserialize)]
struct ImageRequest {
    prompt: String,
    // Data Saver: request a smaller (512px) image instead of 1024px.
    #[serde(default)]
    small: bool,
}

#[derive(Serialize)]
struct ImageResponse {
    url: String,
}

// ---------- Video generation ----------
// Two providers. Magic Hour (text-to-video API) is used when its key is set
// in Settings > Video; otherwise a free fallback runs: a still image from
// Pollinations (free, no key) is animated by a public Hugging Face Wan 2.2
// Space (no key, but its anonymous ZeroGPU quota is small — about one 5s
// clip a day — and the Space can be paused by its owner). Pollinations
// video itself needs a key (HTTP 401 without one), so it is not a free
// route. Videos are saved under videos/ and served at /api/video/file/<name>.
const VIDEOS_DIR: &str = "videos";
const MAGIC_HOUR_API: &str = "https://api.magichour.ai/v1";
const HF_VIDEO_SPACE: &str = "https://prithivmlmods-wan2-2-fast.hf.space";

// Small base64 helpers (no extra crate): the free video path needs to
// base64-encode a JPEG for the Space and decode the MP4 it returns.
const B64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
        out.push(B64_ALPHABET[(n >> 18 & 63) as usize] as char);
        out.push(B64_ALPHABET[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            B64_ALPHABET[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut vals: Vec<u8> = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut nbits = 0;
    for &c in s.as_bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => continue, // skip whitespace/newlines
        };
        acc = (acc << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            vals.push((acc >> nbits) as u8);
        }
    }
    Ok(vals)
}

// Pull the video reference out of the Space's SSE "complete" event data,
// which is JSON like [{"video": "data:video/mp4;base64,...."}].
fn hf_video_ref(data_json: &str) -> Result<String, String> {
    let arr: serde_json::Value =
        serde_json::from_str(data_json).map_err(|e| e.to_string())?;
    let first = arr.get(0).ok_or("empty video result")?;
    let vid = first
        .get("video")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "no video in result".to_string())?;
    if vid.is_empty() {
        return Err("empty video in result".into());
    }
    Ok(vid.to_string())
}

// Decode a "data:video/...;base64,...." URI into raw video bytes.
fn decode_data_uri_video(s: &str) -> Result<Vec<u8>, String> {
    if !s.starts_with("data:video/") {
        return Err("video is not an embedded file".into());
    }
    let b64 = s.splitn(2, ',').nth(1).ok_or("bad video data uri")?;
    let bytes = base64_decode(b64)?;
    if bytes.is_empty() {
        return Err("empty video".into());
    }
    Ok(bytes)
}

// Magic Hour text-to-video: create a job, poll until complete, download the
// MP4. 5 seconds, 480p (the free-tier default), no audio, to keep the free
// credits it costs low. Returns (video bytes, duration seconds).
async fn magic_hour_video(prompt: &str, key: &str) -> Result<(Vec<u8>, u64), String> {
    let client = http_client(30)?;
    let body = serde_json::json!({
        "end_seconds": 5,
        "style": {"prompt": prompt},
        "aspect_ratio": "16:9",
        "resolution": "480p",
        "model": "ltx-2.5",
        "audio": false,
        "name": "Mawaxi AI video",
    });
    let resp = client
        .post(format!("{MAGIC_HOUR_API}/text-to-video"))
        .bearer_auth(key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Magic Hour request failed: {e}"))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        if status.as_u16() == 401 {
            return Err(
                "Magic Hour key was rejected. Check the key in Settings > Video.".into(),
            );
        }
        let msg = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| {
                v.get("message")
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| text.chars().take(200).collect());
        return Err(format!("Magic Hour error ({status}): {msg}"));
    }
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let id = v
        .get("id")
        .and_then(|x| x.as_str())
        .ok_or("Magic Hour gave no video id")?
        .to_string();
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        if Instant::now() > deadline {
            return Err("Magic Hour video took too long (10 minutes). Try again.".into());
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
        let r = client
            .get(format!("{MAGIC_HOUR_API}/video-projects/{id}"))
            .bearer_auth(key)
            .send()
            .await
            .map_err(|e| format!("Magic Hour status check failed: {e}"))?;
        let st = r.status();
        let t = r.text().await.map_err(|e| e.to_string())?;
        if !st.is_success() {
            return Err(format!("Magic Hour status check failed ({st})"));
        }
        let p: serde_json::Value = serde_json::from_str(&t).map_err(|e| e.to_string())?;
        let status_s = p.get("status").and_then(|x| x.as_str()).unwrap_or("");
        match status_s {
            "complete" => {
                let url = p
                    .get("downloads")
                    .and_then(|d| d.get(0))
                    .and_then(|d| d.get("url"))
                    .and_then(|u| u.as_str())
                    .ok_or("Magic Hour finished but gave no download link")?;
                let bytes = client
                    .get(url)
                    .send()
                    .await
                    .map_err(|e| format!("video download failed: {e}"))?
                    .bytes()
                    .await
                    .map_err(|e| e.to_string())?
                    .to_vec();
                if bytes.is_empty() {
                    return Err("Magic Hour returned an empty video".into());
                }
                return Ok((bytes, 5));
            }
            "error" | "canceled" => {
                return Err(
                    "Magic Hour could not make this video. Try a different prompt.".into(),
                );
            }
            _ => {} // draft, queued, rendering — keep waiting
        }
    }
}

// Free video: Pollinations still image, animated by the public Wan 2.2
// Space. Returns (video bytes, duration seconds).
async fn free_video(prompt: &str, small: bool) -> Result<(Vec<u8>, u64), String> {
    let client = http_client(120)?;
    // Step 1: a still image from Pollinations (free, no key). Data Saver
    // uses a smaller still (480x270 instead of 768x432).
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        % 1_000_000;
    let (sw, sh) = if small { (480, 270) } else { (768, 432) };
    let img_url = format!(
        "https://image.pollinations.ai/prompt/{}?width={}&height={}&seed={}&nologo=true",
        percent_encode(prompt),
        sw,
        sh,
        seed
    );
    // Fetch the still with retries: Pollinations sometimes answers 503
    // ("at its limit") and the error body is NOT a JPEG. The old code
    // accepted any non-empty body, so the error text was sent to the Wan
    // Space as image_b64 and PIL failed with "cannot identify image
    // file". Validate the HTTP status and the image magic bytes first.
    let mut img: Vec<u8> = Vec::new();
    let mut last_err = String::new();
    // Pollinations' error JSON says how long to wait ("Retry after 10s").
    // Honour it instead of a fixed 5s gap, or the retries land inside
    // the same 45s-per-IP window and keep failing.
    let mut wait_secs: u64 = 5;
    for attempt in 1..=4 {
        match client.get(&img_url).send().await {
            Ok(resp) => {
                let st = resp.status();
                match resp.bytes().await {
                    Ok(b) => {
                        let b = b.to_vec();
                        let looks_jpeg = b.len() > 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF;
                        let looks_png = b.len() > 3 && b[0] == 0x89 && b[1] == 0x50 && b[2] == 0x4E && b[3] == 0x47;
                        if st.is_success() && (looks_jpeg || looks_png) {
                            img = b;
                            break;
                        }
                        let txt = String::from_utf8_lossy(&b);
                        if let Some(pos) = txt.find("Retry after ") {
                            let rest = &txt[pos + 12..];
                            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                            if let Ok(n) = digits.parse::<u64>() {
                                wait_secs = n + 2;
                            }
                        }
                        last_err = format!("still image came back as HTTP {st}, not a photo");
                    }
                    Err(e) => last_err = e.to_string(),
                }
            }
            Err(e) => last_err = format!("still image failed: {e}"),
        }
        if attempt < 4 {
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
        }
    }
    if img.is_empty() {
        return Err(format!("still image unavailable: {last_err}. Try again in a bit."));
    }
    let img_b64 = format!("data:image/jpeg;base64,{}", base64_encode(&img));
    // Step 2: animate it on the public Wan 2.2 Space (Gradio call/v2 API).
    let body = serde_json::json!({
        "image_b64": img_b64,
        "prompt": prompt,
        "steps": 4,
        "negative_prompt": "",
        "duration_seconds": 5,
        "guidance_scale": 5,
        "guidance_scale_2": 5,
        "seed": (seed % 1_000_000) as i64,
        "randomize_seed": false,
    });
    let resp = client
        .post(format!(
            "{HF_VIDEO_SPACE}/gradio_api/call/v2/generate_video"
        ))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("video request failed: {e}"))?;
    let st = resp.status();
    let t = resp.text().await.map_err(|e| e.to_string())?;
    if !st.is_success() {
        return Err(format!("video service error ({st})"));
    }
    let v: serde_json::Value = serde_json::from_str(&t).map_err(|e| e.to_string())?;
    let event_id = v
        .get("event_id")
        .and_then(|x| x.as_str())
        .ok_or("video service gave no job")?
        .to_string();
    // Step 3: follow the SSE stream until "complete" (up to ~10 minutes).
    // The stream needs a long client timeout: generation takes ~1 minute.
    let sse_client = http_client(660)?;
    let sse = sse_client
        .get(format!(
            "{HF_VIDEO_SPACE}/gradio_api/call/generate_video/{event_id}"
        ))
        .header("Accept", "text/event-stream")
        .send()
        .await
        .map_err(|e| format!("video stream failed: {e}"))?;
    let mut stream = sse.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut last_event = String::new();
    let deadline = Instant::now() + Duration::from_secs(600);
    while let Some(chunk) = stream.next().await {
        if Instant::now() > deadline {
            return Err("Free video took too long (10 minutes). Try again.".into());
        }
        let c = chunk.map_err(|e| e.to_string())?;
        buf.extend_from_slice(&c);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(ev) = line.strip_prefix("event:") {
                last_event = ev.trim().to_string();
            } else if let Some(data) = line.strip_prefix("data:") {
                let data = data.trim();
                if last_event == "error" {
                    let msg = serde_json::from_str::<serde_json::Value>(data)
                        .ok()
                        .and_then(|j| {
                            j.get("error")
                                .and_then(|e| e.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| data.to_string());
                    if msg.contains("ZeroGPU quota") {
                        return Err("Free video limit used up for today (Hugging Face free quota, about one clip a day). Try again tomorrow, or add a Magic Hour key in Settings > Video for more video.".into());
                    }
                    return Err(format!(
                        "Video failed: {}",
                        msg.chars().take(200).collect::<String>()
                    ));
                }
                if last_event == "complete" {
                    let vid = hf_video_ref(data)?;
                    if vid.starts_with("data:video/") {
                        let bytes = decode_data_uri_video(&vid)?;
                        return Ok((bytes, 5));
                    }
                    // A file URL/path on the Space — download it.
                    let full = if vid.starts_with("http") {
                        vid.clone()
                    } else {
                        format!("{HF_VIDEO_SPACE}{vid}")
                    };
                    let bytes = client
                        .get(&full)
                        .send()
                        .await
                        .map_err(|e| format!("video download failed: {e}"))?
                        .bytes()
                        .await
                        .map_err(|e| e.to_string())?
                        .to_vec();
                    if bytes.is_empty() {
                        return Err("video download came back empty".into());
                    }
                    return Ok((bytes, 5));
                }
            }
        }
    }
    Err("Video stream ended before the clip was ready. Try again.".into())
}

#[derive(Deserialize)]
struct VideoRequest {
    prompt: String,
    // Data Saver: the free route uses a smaller still image.
    #[serde(default)]
    small: bool,
}

#[derive(Serialize)]
struct VideoResponse {
    url: String,
    provider: String,
    duration_secs: u64,
    size_bytes: usize,
}

// POST /api/video — generate a 5-second clip for the prompt and return the
// URL it plays from. Magic Hour when its key is set, else the free route.
async fn generate_video(
    Json(payload): Json<VideoRequest>,
) -> Result<Json<VideoResponse>, (StatusCode, String)> {
    let prompt = payload.prompt.trim().to_string();
    if prompt.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty prompt".into()));
    }
    if prompt.len() > 500 {
        return Err((StatusCode::BAD_REQUEST, "prompt too long".into()));
    }
    let key = load_settings().magic_hour_api_key.trim().to_string();
    let (bytes, secs, provider) = if !key.is_empty() {
        match magic_hour_video(&prompt, &key).await {
            Ok((b, s)) => (b, s, "magichour"),
            Err(e) if e.contains("402") => {
                // Out of Magic Hour credits (402): fall back to the free
                // route so he still gets a video instead of an error.
                // Any other Magic Hour error (bad key, etc.) is returned.
                let (b, s) = free_video(&prompt, payload.small).await.map_err(|e2| {
                    (
                        StatusCode::BAD_GATEWAY,
                        format!("Magic Hour out of credits ({e}); free video also failed: {e2}"),
                    )
                })?;
                (b, s, "free")
            }
            Err(e) => return Err((StatusCode::BAD_GATEWAY, e)),
        }
    } else {
        let (b, s) = free_video(&prompt, payload.small)
            .await
            .map_err(|e| (StatusCode::BAD_GATEWAY, e))?;
        (b, s, "free")
    };
    let _ = fs::create_dir_all(VIDEOS_DIR);
    let name = format!("video-{}.mp4", gen_id());
    let path = PathBuf::from(VIDEOS_DIR).join(&name);
    fs::write(&path, &bytes).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(VideoResponse {
        url: format!("/api/video/file/{name}"),
        provider: provider.to_string(),
        duration_secs: secs,
        size_bytes: bytes.len(),
    }))
}

// GET /api/video/file/:name — serve a saved clip.
async fn serve_video_file(
    Path(name): Path<String>,
) -> Result<(HeaderMap, Vec<u8>), (StatusCode, String)> {
    if name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || !name.ends_with(".mp4")
    {
        return Err((StatusCode::BAD_REQUEST, "bad video name".into()));
    }
    let path = PathBuf::from(VIDEOS_DIR).join(&name);
    let bytes = fs::read(&path).map_err(|_| (StatusCode::NOT_FOUND, "video not found".into()))?;
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("video/mp4"),
    );
    Ok((headers, bytes))
}

// Image generation via Pollinations (free, no key). Returns the image URL;
// the browser loads it directly, so the phone does not download it twice.
async fn generate_image(
    Json(payload): Json<ImageRequest>,
) -> Result<Json<ImageResponse>, (StatusCode, String)> {
    let prompt = payload.prompt.trim().to_string();
    if prompt.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty prompt".into()));
    }
    if prompt.len() > 500 {
        return Err((StatusCode::BAD_REQUEST, "prompt too long".into()));
    }
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        % 1_000_000;
    let (w, h) = if payload.small { (512, 512) } else { (1024, 1024) };
    let url = format!(
        "https://image.pollinations.ai/prompt/{}?width={}&height={}&seed={}&nologo=true",
        percent_encode(&prompt),
        w,
        h,
        seed
    );
    Ok(Json(ImageResponse { url }))
}

#[derive(Deserialize)]
struct WebImageRequest {
    query: String,
}

#[derive(Serialize)]
struct WebImageResponse {
    results: Vec<ImageResult>,
}

// POST /api/web-image — real web photos for plain chat replies. Reuses
// search_web_images() (the same SearXNG search the agent's
// search_images tool uses). Returns the top 3 results; never generates
// an image. The client calls this when a reply ends with a
// [[WEBIMAGE: <query>]] marker.
async fn web_image(
    Json(payload): Json<WebImageRequest>,
) -> Result<Json<WebImageResponse>, (StatusCode, String)> {
    let query = payload.query.trim().to_string();
    if query.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty query".into()));
    }
    if query.len() > 300 {
        return Err((StatusCode::BAD_REQUEST, "query too long".into()));
    }
    match search_web_images(&query).await {
        Ok(results) => Ok(Json(WebImageResponse {
            results: results.into_iter().take(3).collect(),
        })),
        Err(e) => Err((StatusCode::BAD_GATEWAY, e)),
    }
}

// ---------- In-app browser ----------
// The client shows a page in an iframe when the site allows framing, and
// falls back to a server-side readable text view when it does not. The
// server fetches the page (same pattern as the research page fetcher),
// reports whether framing is allowed from X-Frame-Options / CSP, and
// returns the title plus readable text.
fn normalize_browser_url(input: &str) -> String {
    let raw = input.trim().to_string();
    if raw.is_empty() {
        return raw;
    }
    let low = raw.to_lowercase();
    if low.starts_with("http://") || low.starts_with("https://") {
        return raw;
    }
    // No scheme: a bare domain gets https://, anything else is a search.
    if raw.contains('.') && !raw.contains(' ') {
        return format!("https://{raw}");
    }
    format!(
        "https://www.bing.com/search?q={}",
        percent_encode(&raw)
    )
}

fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || h.len() < n.len() {
        return None;
    }
    h.windows(n.len()).position(|w| {
        w.iter()
            .zip(n.iter())
            .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
    })
}

fn frameable_from_headers(headers: &reqwest::header::HeaderMap) -> bool {
    // X-Frame-Options: anything present except ALLOWALL blocks cross-origin
    // framing (DENY, SAMEORIGIN and ALLOW-FROM all block us). A CSP
    // frame-ancestors directive also blocks us — no site lists this phone.
    if let Some(v) = headers.get("x-frame-options") {
        let s = v.to_str().unwrap_or("").trim().to_lowercase();
        if !s.is_empty() && s != "allowall" {
            return false;
        }
    }
    if let Some(v) = headers.get("content-security-policy") {
        let s = v.to_str().unwrap_or("").to_lowercase();
        if s.contains("frame-ancestors") {
            return false;
        }
    }
    true
}

fn html_title(html: &str) -> String {
    if let Some(s) = find_ci(html, "<title>") {
        let start = s + 7;
        if let Some(e) = find_ci(&html[start..], "</title>") {
            return html[start..start + e]
                .trim()
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
                .replace("&nbsp;", " ");
        }
    }
    String::new()
}

#[derive(Serialize)]
struct BrowserFetchResponse {
    url: String,
    final_url: String,
    title: String,
    text: String,
    frameable: bool,
    content_type: String,
}

async fn browser_fetch(
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<BrowserFetchResponse>, (StatusCode, String)> {
    let raw = params.get("url").cloned().unwrap_or_default();
    let url = normalize_browser_url(&raw);
    if url.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "no URL given".into()));
    }
    let client = http_client(12).map_err(|e| (StatusCode::BAD_GATEWAY, e))?;
    let resp = client
        .get(&url)
        .header(
            "User-Agent",
            "Mozilla/5.0 (Linux; Android 10) MawaxiAI/1.0",
        )
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("could not fetch page: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        return Err((StatusCode::BAD_GATEWAY, format!("page returned HTTP {status}")));
    }
    let final_url = resp.url().to_string();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let frameable = frameable_from_headers(resp.headers());
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let capped: Vec<u8> = bytes.iter().take(400_000).cloned().collect();
    let is_html =
        content_type.to_lowercase().contains("html") || content_type.is_empty();
    let (title, text) = if is_html {
        let html = String::from_utf8_lossy(&capped).into_owned();
        let t = html_title(&html);
        let txt = html_to_text(&html);
        let capped_txt: String = txt.chars().take(8000).collect();
        (t, capped_txt)
    } else {
        (String::new(), String::new())
    };
    Ok(Json(BrowserFetchResponse {
        url,
        final_url,
        title,
        text,
        frameable,
        content_type,
    }))
}

// ---------- PDF ----------
// Creation: a minimal PDF writer in pure Rust (no new crate) — A4 pages,
// Helvetica, word-wrapped text (used for chat transcripts only).
// Editing: upload a PDF, extract its text AND each word's position with
// the phone's PyMuPDF helper (pdf_edit.py), edit the text, and save it
// as a NEW PDF by redacting only the changed words in the ORIGINAL PDF
// and inserting the replacements at the same position and size — the
// original format (layout, tables, fonts) is kept. Originals are kept.
// Research reports: the Download PDF button opens the coloured VISUAL
// report in a print view (print-color-adjust: exact) so the saved PDF
// keeps the colours and graphics; the plain server writer is not used
// for research reports.
const PDFS_DIR: &str = "pdfs";
const PDF_INDEX_FILE: &str = "pdfs.json";
const MAX_PDF_BYTES: usize = 20 * 1024 * 1024;

#[derive(Serialize, Deserialize, Clone)]
struct PdfEntry {
    id: String,
    name: String,
    size: usize,
}

fn pdf_sanitize_byte(c: char) -> u8 {
    // WinAnsi (CP1252) for the built-in Helvetica font: ASCII as-is,
    // Latin-1 0xA0-0xFF as their byte, everything else becomes '?'.
    let n = c as u32;
    if n < 0x20 {
        b' '
    } else if n < 0x7f {
        n as u8
    } else if (0xa0..=0xff).contains(&n) {
        n as u8
    } else {
        b'?'
    }
}

fn pdf_escape_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '(' => out.extend_from_slice(b"\\("),
            ')' => out.extend_from_slice(b"\\)"),
            '\\' => out.extend_from_slice(b"\\\\"),
            _ => out.push(pdf_sanitize_byte(c)),
        }
    }
    out
}

fn wrap_pdf_lines(text: &str, max_chars: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for para in text.split('\n') {
        let para = para.trim_end();
        if para.trim().is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut cur = String::new();
        for word in para.split_whitespace() {
            if cur.is_empty() {
                cur = word.to_string();
            } else if cur.chars().count() + 1 + word.chars().count() > max_chars {
                lines.push(std::mem::take(&mut cur));
                cur = word.to_string();
            } else {
                cur.push(' ');
                cur.push_str(word);
            }
        }
        if !cur.is_empty() {
            lines.push(cur);
        }
    }
    lines
}

// Build a valid multi-page A4 PDF from a title and plain text.
// Object ids: 1 Catalog, 2 Pages, 3 F1 Helvetica, 4 F2 Helvetica-Bold,
// then per page i (0-based): Page id = 5 + 2*i, Contents id = 6 + 2*i.
// --- PDF images -----------------------------------------------------------
// Chat PDFs embed the photos that show under replies (real web photos and
// generated images). JPEGs are embedded directly as DCTDecode image
// XObjects; anything else is converted to JPEG with the img_to_jpeg.py
// helper (PyMuPDF), and an image that cannot be loaded at all becomes a
// placeholder line with its title and source URL instead of breaking the PDF.

/// A photo resolved and ready to embed. `jpeg` is None when the image
/// could not be loaded or converted — the writer then prints a placeholder.
struct PdfImageReady {
    jpeg: Option<Vec<u8>>,
    width: u32,
    height: u32,
    title: String,
    source: String,
}

enum PdfBlock {
    Text(String),
    Image(PdfImageReady),
}

/// Parse (width, height, components) from a JPEG's SOF marker.
/// Returns None when the bytes are not a JPEG or have no SOF marker.
fn jpeg_dims(data: &[u8]) -> Option<(u32, u32, u8)> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            return None;
        }
        let mut j = i + 1;
        while j < data.len() && data[j] == 0xFF {
            j += 1;
        }
        if j >= data.len() {
            return None;
        }
        let marker = data[j];
        if marker == 0x00 {
            return None;
        }
        if marker == 0xD9 || marker == 0xDA {
            return None; // EOI / SOS: no SOF found before the scan data.
        }
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            i = j + 1; // Standalone marker, no length field.
            continue;
        }
        if j + 3 >= data.len() {
            return None;
        }
        let len = ((data[j + 1] as usize) << 8) | data[j + 2] as usize;
        if len < 8 || j + 1 + len > data.len() {
            return None;
        }
        if matches!(marker, 0xC0 | 0xC1 | 0xC2) {
            // SOF payload: [precision(1)][height(2)][width(2)][components(1)]
            let h = ((data[j + 4] as u32) << 8) | data[j + 5] as u32;
            let w = ((data[j + 6] as u32) << 8) | data[j + 7] as u32;
            let comp = data[j + 8];
            if w == 0 || h == 0 {
                return None;
            }
            return Some((w, h, comp));
        }
        i = j + 1 + len;
    }
    None
}

// Build a valid multi-page A4 PDF from a title and an ordered list of
// blocks (text and images). Object ids: 1 Catalog, 2 Pages, 3 F1 Helvetica,
// 4 F2 Helvetica-Bold, then per page i (0-based): Page id = 5 + 2*i,
// Contents id = 6 + 2*i, then one image XObject per embedded photo.
fn make_pdf_blocks(title: &str, blocks: &[PdfBlock]) -> Vec<u8> {
    const PAGE_W: i32 = 595;
    const PAGE_H: i32 = 842;
    const MARGIN: i32 = 56;
    const CONTENT_W: i32 = PAGE_W - 2 * MARGIN; // 483
    const BODY_SIZE: i32 = 11;
    const LINE_H: i32 = 15;
    const CAP_SIZE: i32 = 10;
    const CAP_H: i32 = 13;
    const IMG_MAX_H: i32 = 600;
    const BODY_TOP_FIRST: i32 = PAGE_H - 106; // below the title on page 1
    const BODY_TOP: i32 = PAGE_H - 72;
    const BODY_BOTTOM: i32 = 56;

    struct PlacedImage {
        img_idx: usize,
        y: i32, // bottom edge of the image
        w: i32,
        h: i32,
        caption: Vec<(String, i32)>, // (text, baseline y)
    }
    #[derive(Default)]
    struct PageLayout {
        lines: Vec<(String, i32)>, // (text, baseline y)
        images: Vec<PlacedImage>,
    }

    // Photos embedded, in block order — their XObject ids follow the pages.
    let embedded: Vec<&PdfImageReady> = blocks
        .iter()
        .filter_map(|b| match b {
            PdfBlock::Image(img) if img.jpeg.is_some() => Some(img),
            _ => None,
        })
        .collect();

    let mut pages: Vec<PageLayout> = vec![PageLayout::default()];
    let mut page_no = 0usize;
    let mut cursor = BODY_TOP_FIRST;
    let mut img_counter = 0usize;
    for block in blocks {
        match block {
            PdfBlock::Text(text) => {
                for line in wrap_pdf_lines(text, 90) {
                    if cursor - LINE_H < BODY_BOTTOM {
                        pages.push(PageLayout::default());
                        page_no += 1;
                        cursor = BODY_TOP;
                    }
                    pages[page_no].lines.push((line, cursor));
                    cursor -= LINE_H;
                }
            }
            PdfBlock::Image(img) => {
                if img.jpeg.is_none() {
                    // Placeholder: title + source as plain text lines.
                    let mut ph = format!("[Image not available: {}]", img.title);
                    if !img.source.is_empty() {
                        ph.push_str(&format!("\nSource: {}", img.source));
                    }
                    for line in wrap_pdf_lines(&ph, 90) {
                        if cursor - LINE_H < BODY_BOTTOM {
                            pages.push(PageLayout::default());
                            page_no += 1;
                            cursor = BODY_TOP;
                        }
                        pages[page_no].lines.push((line, cursor));
                        cursor -= LINE_H;
                    }
                    continue;
                }
                let mut w = img.width as i32;
                let mut h = img.height as i32;
                if w > CONTENT_W {
                    h = (h * CONTENT_W / w).max(1);
                    w = CONTENT_W;
                }
                if h > IMG_MAX_H {
                    w = (w * IMG_MAX_H / h).max(1);
                    h = IMG_MAX_H;
                }
                let mut cap_text = img.title.clone();
                if !img.source.is_empty() {
                    if !cap_text.is_empty() {
                        cap_text.push('\n');
                    }
                    cap_text.push_str(&format!("Source: {}", img.source));
                }
                let caption_lines = if cap_text.trim().is_empty() {
                    Vec::new()
                } else {
                    wrap_pdf_lines(&cap_text, 80)
                };
                let mut need = h + 8 + (caption_lines.len() as i32) * CAP_H + 10;
                if cursor - need < BODY_BOTTOM {
                    pages.push(PageLayout::default());
                    page_no += 1;
                    cursor = BODY_TOP;
                    let avail =
                        BODY_TOP - BODY_BOTTOM - 8 - (caption_lines.len() as i32) * CAP_H - 10;
                    if h > avail {
                        let nh = avail.max(1);
                        w = (w * nh / h).max(1);
                        h = nh;
                        need = h + 8 + (caption_lines.len() as i32) * CAP_H + 10;
                    }
                }
                let y = cursor - h;
                let mut cy = cursor - h - 8 - CAP_SIZE;
                let mut caption: Vec<(String, i32)> = Vec::new();
                for cap in caption_lines {
                    caption.push((cap, cy));
                    cy -= CAP_H;
                }
                pages[page_no].images.push(PlacedImage {
                    img_idx: img_counter,
                    y,
                    w,
                    h,
                    caption,
                });
                img_counter += 1;
                cursor -= need;
            }
        }
    }
    let n_pages = pages.len();
    let img_base = 5 + 2 * n_pages;
    let mut objs: Vec<Vec<u8>> = Vec::new();
    objs.push(b"<< /Type /Catalog /Pages 2 0 R >>".to_vec());
    let kids: Vec<String> = (0..n_pages).map(|i| format!("{} 0 R", 5 + 2 * i)).collect();
    objs.push(
        format!(
            "<< /Type /Pages /Kids [{}] /Count {} >>",
            kids.join(" "),
            n_pages
        )
        .into_bytes(),
    );
    objs.push(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
    );
    objs.push(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
            .to_vec(),
    );
    for (i, page) in pages.iter().enumerate() {
        let cont_id = 6 + 2 * i;
        let mut xobj = String::new();
        for im in &page.images {
            xobj.push_str(&format!(" /Im{} {} 0 R", im.img_idx, img_base + im.img_idx));
        }
        let res = if xobj.is_empty() {
            "<< /Font << /F1 3 0 R /F2 4 0 R >> >>".to_string()
        } else {
            format!(
                "<< /Font << /F1 3 0 R /F2 4 0 R >> /XObject <<{} >> >>",
                xobj
            )
        };
        objs.push(
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Resources {} /Contents {} 0 R >>",
                PAGE_W, PAGE_H, res, cont_id
            )
            .into_bytes(),
        );
        let mut stream: Vec<u8> = Vec::new();
        if i == 0 {
            stream.extend_from_slice(b"BT /F2 16 Tf 56 786 Td (");
            stream.extend_from_slice(&pdf_escape_bytes(title));
            stream.extend_from_slice(b") Tj ET\n");
        }
        for (line, y) in &page.lines {
            stream.extend_from_slice(
                format!("BT /F1 {} Tf 56 {} Td (", BODY_SIZE, y).as_bytes(),
            );
            stream.extend_from_slice(&pdf_escape_bytes(line));
            stream.extend_from_slice(b") Tj ET\n");
        }
        for im in &page.images {
            stream.extend_from_slice(
                format!(
                    "q {} 0 0 {} {} {} cm /Im{} Do Q\n",
                    im.w, im.h, MARGIN, im.y, im.img_idx
                )
                .as_bytes(),
            );
            for (cap, cy) in &im.caption {
                stream.extend_from_slice(
                    format!("BT /F1 {} Tf 56 {} Td (", CAP_SIZE, cy).as_bytes(),
                );
                stream.extend_from_slice(&pdf_escape_bytes(cap));
                stream.extend_from_slice(b") Tj ET\n");
            }
        }
        let mut cont = format!("<< /Length {} >>\nstream\n", stream.len()).into_bytes();
        cont.extend_from_slice(&stream);
        cont.extend_from_slice(b"\nendstream");
        objs.push(cont);
    }
    for img in &embedded {
        let jpeg = img.jpeg.as_ref().expect("embedded image has bytes");
        let cs = match jpeg_dims(jpeg) {
            Some((_, _, 1)) => "/DeviceGray",
            _ => "/DeviceRGB",
        };
        let mut obj = format!(
            "<< /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace {} /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>\nstream\n",
            img.width, img.height, cs, jpeg.len()
        )
        .into_bytes();
        obj.extend_from_slice(jpeg);
        obj.extend_from_slice(b"\nendstream");
        objs.push(obj);
    }
    let mut pdf: Vec<u8> = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");
    let mut offsets: Vec<usize> = vec![0];
    for (idx, body) in objs.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", idx + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref_pos = pdf.len();
    let total = objs.len() + 1;
    pdf.extend_from_slice(format!("xref\n0 {}\n", total).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for off in offsets.iter().skip(1) {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total, xref_pos
        )
        .as_bytes(),
    );
    pdf
}

// Build a valid multi-page A4 PDF from a title and plain text (no images).
/// Kept as the text-only entry point (used by the PDF tests); the live
/// handlers build mixed text/image PDFs with make_pdf_blocks.
#[allow(dead_code)]
fn make_pdf(title: &str, text: &str) -> Vec<u8> {
    make_pdf_blocks(title, &[PdfBlock::Text(text.to_string())])
}

/// One photo for a PDF: a remote https URL (web photo or generated
/// image) or a local file path, plus its caption and source page.
#[derive(Deserialize, Clone, Default)]
struct PdfImageInput {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    source: String,
}

/// One ordered segment of a chat PDF: a text chunk, or a photo that
/// belongs under the text before it. The client sends these in reading
/// order so each picture lands under the text it illustrates.
#[derive(Deserialize, Clone, Default)]
struct PdfSegment {
    #[serde(default)]
    kind: String, // "text" or "image"
    #[serde(default)]
    text: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    source: String,
}

#[derive(Deserialize)]
struct PdfCreateRequest {
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
    /// Photos appended after the text (legacy path; segments win when set).
    #[serde(default)]
    images: Vec<PdfImageInput>,
    /// Ordered text/image segments in reading order (preferred).
    #[serde(default)]
    segments: Vec<PdfSegment>,
}

#[derive(Deserialize)]
struct PdfUploadRequest {
    name: String,
    data: String, // base64 PDF bytes, optionally a data: URL
}

#[derive(Deserialize)]
struct PdfEditRequest {
    id: String,
    #[serde(default)]
    text: String,
}

// The PyMuPDF helper (pdf_edit.py) lives in the project dir next to the
// server binary's working directory. Resolve it from the current working
// directory first, then from the executable's location.
fn pdf_helper_path() -> PathBuf {
    let rel = PathBuf::from("pdf_edit.py");
    if rel.exists() {
        return rel;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
        {
            let p = dir.join("pdf_edit.py");
            if p.exists() {
                return p;
            }
        }
    }
    rel
}

// Run the PyMuPDF helper and parse the JSON it prints on stdout.
fn run_pdf_helper(args: &[&str]) -> Result<serde_json::Value, String> {
    let helper = pdf_helper_path();
    let out = std::process::Command::new("timeout")
        .arg("60")
        .arg("python3")
        .arg(&helper)
        .args(args)
        .output()
        .map_err(|e| format!("could not run the PDF helper: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(if err.is_empty() {
            "the PDF helper failed".to_string()
        } else {
            err
        });
    }
    serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("the PDF helper returned bad data: {e}"))
}

async fn run_pdf_ocr(pdf_path: &str) -> Result<String, String> {
    let run = tokio::process::Command::new(MIRAGE_PYTHON)
        .arg(PDF_OCR_SCRIPT)
        .arg(pdf_path)
        .arg("8")
        .output();
    let out = tokio::time::timeout(std::time::Duration::from_secs(120), run)
        .await
        .map_err(|_| "PDF OCR timed out".to_string())
        .and_then(|r| r.map_err(|e| format!("cannot run PDF OCR: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(if err.is_empty() {
            "the PDF OCR helper failed".to_string()
        } else {
            err
        });
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("the PDF OCR helper returned bad data: {e}"))?;
    Ok(v
        .get("text")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .trim()
        .to_string())
}

fn pdf_path(id: &str) -> PathBuf {
    PathBuf::from(PDFS_DIR).join(format!("{id}.pdf"))
}

fn pdf_slug(name: &str) -> String {
    let mut s = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            s.push(c.to_ascii_lowercase());
        } else if c == ' ' || c == '.' {
            s.push('-');
        }
    }
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    let s = s.trim_matches('-').to_string();
    if s.is_empty() {
        "document".to_string()
    } else {
        s
    }
}

// Create a PDF from plain text (chat or research report) and return the
// file itself for download.
async fn create_pdf(
    Json(payload): Json<PdfCreateRequest>,
) -> Result<(HeaderMap, Vec<u8>), (StatusCode, String)> {
    let title = payload.title.trim().to_string();
    let text = payload.text.trim_end().to_string();
    if text.is_empty() && payload.segments.is_empty() && payload.images.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "nothing to put in the PDF".into()));
    }
    if text.len() > 500_000 {
        return Err((StatusCode::BAD_REQUEST, "text too long for a PDF".into()));
    }
    let title = if title.is_empty() {
        "Mawaxi AI".to_string()
    } else {
        title
    };
    let blocks = pdf_request_blocks(&payload).await;
    let (embedded, failed) = pdf_block_image_counts(&blocks);
    let bytes = make_pdf_blocks(&title, &blocks);
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/pdf"),
    );
    let fname = format!("{}.pdf", pdf_slug(&title));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"{fname}\""))
            .unwrap_or(HeaderValue::from_static("attachment")),
    );
    headers.insert(
        "X-Images-Embedded",
        HeaderValue::from_str(&embedded.to_string())
            .unwrap_or(HeaderValue::from_static("0")),
    );
    headers.insert(
        "X-Images-Failed",
        HeaderValue::from_str(&failed.to_string())
            .unwrap_or(HeaderValue::from_static("0")),
    );
    Ok((headers, bytes))
}

/// Text-only library save (used by the PDF tests); the live handlers save
/// mixed text/image PDFs with save_pdf_blocks_to_library.
#[allow(dead_code)]
fn save_pdf_to_library_at(
    dir: &str,
    index: &str,
    title: &str,
    text: &str,
) -> Result<PdfEntry, String> {
    let text = text.trim_end();
    if text.is_empty() {
        return Err("nothing to put in the PDF".into());
    }
    if text.len() > 500_000 {
        return Err("text too long for a PDF".into());
    }
    let mut title = title.trim().to_string();
    if title.to_lowercase().ends_with(".pdf") {
        title.truncate(title.len() - 4);
        title = title.trim().to_string();
    }
    if title.is_empty() {
        title = "Mawaxi AI".to_string();
    }
    let bytes = make_pdf(&title, text);
    let _ = fs::create_dir_all(dir);
    let id = gen_id();
    let path = PathBuf::from(dir).join(format!("{id}.pdf"));
    fs::write(&path, &bytes).map_err(|e| e.to_string())?;
    let name = format!("{}.pdf", pdf_slug(&title));
    let mut entries: Vec<PdfEntry> = load_json(index);
    entries.push(PdfEntry { id: id.clone(), name: name.clone(), size: bytes.len() });
    save_json(index, &entries)?;
    Ok(PdfEntry { id, name, size: bytes.len() })
}

/// Path to the img_to_jpeg.py helper (same lookup as pdf_helper_path).
fn img_helper_path() -> PathBuf {
    let rel = PathBuf::from("img_to_jpeg.py");
    if rel.exists() {
        return rel;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
        {
            let p = dir.join("img_to_jpeg.py");
            if p.exists() {
                return p;
            }
        }
    }
    rel
}

/// Convert non-JPEG image bytes to JPEG with the PyMuPDF helper.
/// Returns None when the bytes are not a convertible image (WebP and
/// HTML error pages both fail here — they must never become a fake photo).
fn convert_image_bytes(bytes: &[u8]) -> Option<Vec<u8>> {
    let path = std::env::temp_dir().join(format!("mawaxi-img-{}.bin", gen_id()));
    if fs::write(&path, bytes).is_err() {
        return None;
    }
    let out = std::process::Command::new("timeout")
        .arg("30")
        .arg("python3")
        .arg(img_helper_path())
        .arg(&path)
        .output();
    let _ = fs::remove_file(&path);
    let out = match out {
        Ok(o) if o.status.success() => o,
        _ => return None,
    };
    let jpeg = out.stdout;
    if jpeg_dims(&jpeg).is_some() {
        Some(jpeg)
    } else {
        None
    }
}

/// Fetch one image for a PDF and return it ready to embed. `url` is a
/// remote https URL (web photo or generated image) or a local file path.
/// Remote fetches use a 25s timeout and a 20MB cap — Wikimedia originals
/// can be 13MB+ and slow on this link, and they are downscaled to <=1200px
/// by the converter instead of being rejected. A failure never errors —
/// it becomes a placeholder line with the title and source URL, so one
/// dead photo cannot break the whole PDF.
async fn resolve_pdf_image(input: &PdfImageInput) -> PdfImageReady {
    let title = input.title.trim().to_string();
    let source = input.source.trim().to_string();
    let failed = || PdfImageReady {
        jpeg: None,
        width: 0,
        height: 0,
        title: title.clone(),
        source: source.clone(),
    };
    let url = input.url.trim();
    if url.is_empty() {
        return failed();
    }
    let bytes: Option<Vec<u8>> = if url.starts_with("http://") || url.starts_with("https://") {
        match http_client_ipv4(25) {
            Ok(client) => match client.get(url).send().await {
                Ok(resp) if resp.status().is_success() => resp
                    .bytes()
                    .await
                    .ok()
                    .map(|b| b.to_vec())
                    .filter(|b| !b.is_empty() && b.len() <= 20_000_000),
                _ => None,
            },
            Err(_) => None,
        }
    } else {
        // Local file: absolute path, or relative to the workspace.
        let path = if url.starts_with('/') {
            PathBuf::from(url)
        } else {
            PathBuf::from(WORKSPACE_DIR).join(url)
        };
        fs::read(path)
            .ok()
            .filter(|b| !b.is_empty() && b.len() <= 20_000_000)
    };
    let Some(bytes) = bytes else {
        return failed();
    };
    // Small RGB/grayscale JPEG: embed the bytes as-is.
    if bytes.starts_with(b"\xff\xd8\xff") {
        if let Some((w, h, comp)) = jpeg_dims(&bytes) {
            if (comp == 1 || comp == 3) && w <= 1200 && h <= 1200 && bytes.len() <= 2_000_000 {
                return PdfImageReady {
                    jpeg: Some(bytes),
                    width: w,
                    height: h,
                    title,
                    source,
                };
            }
        }
    }
    // Anything else convertible (PNG, GIF, CMYK or oversized JPEG):
    // convert with the helper. WebP and error pages fail here.
    match convert_image_bytes(&bytes) {
        Some(jpeg) => match jpeg_dims(&jpeg) {
            Some((w, h, _)) => PdfImageReady {
                jpeg: Some(jpeg),
                width: w,
                height: h,
                title,
                source,
            },
            None => failed(),
        },
        None => failed(),
    }
}

/// Pull Markdown images `![alt](url)` out of text. Returns the text with
/// each image replaced by `[Image: alt]` and the image inputs in order.
fn extract_md_images(text: &str) -> (String, Vec<PdfImageInput>) {
    let mut out = String::with_capacity(text.len());
    let mut images: Vec<PdfImageInput> = Vec::new();
    let mut rest = text;
    while let Some(bang) = rest.find("![") {
        let after_bang = &rest[bang + 2..];
        let Some(close) = after_bang.find("](") else {
            break;
        };
        let alt = &after_bang[..close];
        let after_paren = &after_bang[close + 2..];
        let Some(end) = after_paren.find(')') else {
            break;
        };
        let url = &after_paren[..end];
        if url.trim().is_empty() {
            break;
        }
        out.push_str(&rest[..bang]);
        out.push_str(&format!("[Image: {}]", alt.trim()));
        images.push(PdfImageInput {
            url: url.trim().to_string(),
            title: alt.trim().to_string(),
            source: String::new(),
        });
        rest = &after_paren[end + 1..];
    }
    out.push_str(rest);
    (out, images)
}

/// Build the ordered PDF blocks for a create/save request. Segments
/// (ordered text/image) win when present; otherwise the plain text is one
/// block with the images appended after it. At most 8 images are fetched;
/// any further images stay as placeholders with their title and source.
async fn pdf_request_blocks(payload: &PdfCreateRequest) -> Vec<PdfBlock> {
    let mut blocks: Vec<PdfBlock> = Vec::new();
    let mut pending: Vec<(usize, PdfImageInput)> = Vec::new();
    if !payload.segments.is_empty() {
        for seg in &payload.segments {
            if seg.kind == "image" {
                let input = PdfImageInput {
                    url: seg.url.clone(),
                    title: seg.title.clone(),
                    source: seg.source.clone(),
                };
                pending.push((blocks.len(), input));
                blocks.push(PdfBlock::Image(PdfImageReady {
                    jpeg: None,
                    width: 0,
                    height: 0,
                    title: seg.title.clone(),
                    source: seg.source.clone(),
                }));
            } else if !seg.text.trim().is_empty() {
                blocks.push(PdfBlock::Text(seg.text.clone()));
            }
        }
    } else {
        blocks.push(PdfBlock::Text(payload.text.clone()));
        for img in &payload.images {
            pending.push((blocks.len(), img.clone()));
            blocks.push(PdfBlock::Image(PdfImageReady {
                jpeg: None,
                width: 0,
                height: 0,
                title: img.title.clone(),
                source: img.source.clone(),
            }));
        }
    }
    for (n, (bi, input)) in pending.into_iter().enumerate() {
        if n >= 8 {
            break;
        }
        if let PdfBlock::Image(slot) = &mut blocks[bi] {
            *slot = resolve_pdf_image(&input).await;
        }
    }
    blocks
}

/// (embedded images, failed images) in a block list.
fn pdf_block_image_counts(blocks: &[PdfBlock]) -> (usize, usize) {
    let mut embedded = 0usize;
    let mut failed = 0usize;
    for b in blocks {
        if let PdfBlock::Image(img) = b {
            if img.jpeg.is_some() {
                embedded += 1;
            } else {
                failed += 1;
            }
        }
    }
    (embedded, failed)
}

/// Save a block-built PDF in the PDF library (pdfs/ + pdfs.json).
/// Mirrors save_pdf_to_library_at, but takes ordered blocks (text + images).
fn save_pdf_blocks_to_library(
    dir: &str,
    index: &str,
    title: &str,
    blocks: &[PdfBlock],
) -> Result<PdfEntry, String> {
    let has_content = blocks.iter().any(|b| match b {
        PdfBlock::Text(t) => !t.trim().is_empty(),
        PdfBlock::Image(_) => true,
    });
    if !has_content {
        return Err("nothing to put in the PDF".into());
    }
    let total_text: usize = blocks
        .iter()
        .map(|b| match b {
            PdfBlock::Text(t) => t.len(),
            PdfBlock::Image(_) => 0,
        })
        .sum();
    if total_text > 500_000 {
        return Err("text too long for a PDF".into());
    }
    let mut title = title.trim().to_string();
    if title.to_lowercase().ends_with(".pdf") {
        title.truncate(title.len() - 4);
        title = title.trim().to_string();
    }
    if title.is_empty() {
        title = "Mawaxi AI".to_string();
    }
    let bytes = make_pdf_blocks(&title, blocks);
    let _ = fs::create_dir_all(dir);
    let id = gen_id();
    let path = PathBuf::from(dir).join(format!("{id}.pdf"));
    fs::write(&path, &bytes).map_err(|e| e.to_string())?;
    let name = format!("{}.pdf", pdf_slug(&title));
    let mut entries: Vec<PdfEntry> = load_json(index);
    entries.push(PdfEntry {
        id: id.clone(),
        name: name.clone(),
        size: bytes.len(),
    });
    save_json(index, &entries)?;
    Ok(PdfEntry {
        id,
        name,
        size: bytes.len(),
    })
}

// POST /api/pdf/save - same body as /api/pdf/create, but the PDF is saved
// in the library instead of returned for immediate download.
async fn save_pdf(
    Json(payload): Json<PdfCreateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if payload.text.trim().is_empty()
        && payload.segments.is_empty()
        && payload.images.is_empty()
    {
        return Err((StatusCode::BAD_REQUEST, "nothing to put in the PDF".into()));
    }
    if payload.text.len() > 500_000 {
        return Err((StatusCode::BAD_REQUEST, "text too long for a PDF".into()));
    }
    let blocks = pdf_request_blocks(&payload).await;
    let (embedded, failed) = pdf_block_image_counts(&blocks);
    match save_pdf_blocks_to_library(PDFS_DIR, PDF_INDEX_FILE, &payload.title, &blocks) {
        Ok(e) => Ok(Json(serde_json::json!({
            "id": e.id,
            "name": e.name,
            "size": e.size,
            "url": format!("/api/pdf/file/{}", e.id),
            "images_embedded": embedded,
            "images_failed": failed,
        }))),
        Err(m) => Err((StatusCode::BAD_REQUEST, m)),
    }
}

async fn upload_pdf(
    Json(payload): Json<PdfUploadRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let name = payload.name.trim().to_string();
    if name.is_empty() || !name.to_lowercase().ends_with(".pdf") {
        return Err((StatusCode::BAD_REQUEST, "please upload a PDF file".into()));
    }
    let b64 = payload
        .data
        .split_once(',')
        .map(|(_, rest)| rest)
        .unwrap_or(&payload.data);
    let bytes =
        base64_decode(b64).map_err(|_| (StatusCode::BAD_REQUEST, "bad file data".into()))?;
    if bytes.len() > MAX_PDF_BYTES {
        return Err((
            StatusCode::BAD_REQUEST,
            "PDF is too large (max 20 MB)".into(),
        ));
    }
    if !bytes.starts_with(b"%PDF") {
        return Err((StatusCode::BAD_REQUEST, "that file is not a PDF".into()));
    }
    let _ = fs::create_dir_all(PDFS_DIR);
    let id = gen_id();
    let path = pdf_path(&id);
    fs::write(&path, &bytes).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    // Extract text (and word positions) with the PyMuPDF helper, so the
    // text shown for editing and the word positions agree. When the PDF
    // has no extractable text (e.g. a scanned image), the file is kept
    // for viewing but editing stays off.
    let (text, extract_error) = match run_pdf_helper(&["extract", &path.to_string_lossy()]) {
        Ok(v) => {
            let t = v
                .get("text")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if t.is_empty() {
                // No text layer: a scanned-image PDF. Read the pages with
                // Mirage OCR so the text is available in chat and editing.
                match run_pdf_ocr(&path.to_string_lossy()).await {
                    Ok(ocr) if !ocr.is_empty() => (
                        ocr,
                        Some(
                            "text was read from the scanned pages by OCR - check it for small errors"
                                .to_string(),
                        ),
                    ),
                    _ => (
                        String::new(),
                        Some(
                            "no text found in this PDF — it may be a scanned image"
                                .to_string(),
                        ),
                    ),
                }
            } else {
                (t, None)
            }
        }
        Err(e) => (
            String::new(),
            Some(format!("could not read this PDF for editing: {e}")),
        ),
    };
    let mut entries: Vec<PdfEntry> = load_json(PDF_INDEX_FILE);
    entries.push(PdfEntry {
        id: id.clone(),
        name: name.clone(),
        size: bytes.len(),
    });
    save_json(PDF_INDEX_FILE, &entries)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(serde_json::json!({
        "id": id,
        "name": name,
        "size": bytes.len(),
        "text": text,
        "extract_error": extract_error,
    })))
}

async fn list_pdfs() -> Json<Vec<PdfEntry>> {
    let mut entries: Vec<PdfEntry> = load_json(PDF_INDEX_FILE);
    entries.retain(|e| pdf_path(&e.id).exists());
    Json(entries)
}

async fn serve_pdf_file(
    Path(id): Path<String>,
) -> Result<(HeaderMap, Vec<u8>), (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad pdf id".into()));
    }
    let bytes =
        fs::read(pdf_path(&id)).map_err(|_| (StatusCode::NOT_FOUND, "PDF not found".into()))?;
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/pdf"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("inline"),
    );
    Ok((headers, bytes))
}

// Extract the text of a stored PDF (for editing), via the PyMuPDF helper.
async fn pdf_text(
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad pdf id".into()));
    }
    let path = pdf_path(&id);
    if !path.exists() {
        return Err((StatusCode::NOT_FOUND, "PDF not found".into()));
    }
    let v = run_pdf_helper(&["extract", &path.to_string_lossy()]).map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("could not read this PDF: {e}"),
        )
    })?;
    let text = v
        .get("text")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if text.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "no text found in this PDF — it may be a scanned image".into(),
        ));
    }
    Ok(Json(serde_json::json!({"id": id, "text": text})))
}

// Save edited text as a NEW PDF that keeps the ORIGINAL format. The
// helper redacts only the changed words in the original PDF and inserts
// the replacements at the same position and size. The original file is
// kept untouched. If the helper fails, report the error — never fall
// back to a plain-text PDF, which would lose the format.
async fn edit_pdf(
    Json(payload): Json<PdfEditRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !valid_id(&payload.id) {
        return Err((StatusCode::BAD_REQUEST, "bad pdf id".into()));
    }
    let mut entries: Vec<PdfEntry> = load_json(PDF_INDEX_FILE);
    let orig = match entries.iter().find(|e| e.id == payload.id).cloned() {
        Some(o) => o,
        None => return Err((StatusCode::NOT_FOUND, "PDF not found".into())),
    };
    let text = payload.text.trim_end().to_string();
    if text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "edited text is empty".into()));
    }
    if text.len() > 500_000 {
        return Err((StatusCode::BAD_REQUEST, "text too long for a PDF".into()));
    }
    let new_id = gen_id();
    let _ = fs::create_dir_all(PDFS_DIR);
    let out_path = pdf_path(&new_id);
    let tmp_txt = std::env::temp_dir().join(format!("pdf_edit_{}.txt", payload.id));
    fs::write(&tmp_txt, &text)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let res = run_pdf_helper(&[
        "edit",
        &pdf_path(&payload.id).to_string_lossy(),
        &tmp_txt.to_string_lossy(),
        &out_path.to_string_lossy(),
    ]);
    let _ = fs::remove_file(&tmp_txt);
    let v = res.map_err(|e| (StatusCode::BAD_GATEWAY, e))?;
    let n_changes = v
        .get("changes")
        .and_then(|c| c.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    if n_changes == 0 {
        // Nothing changed — the edited copy is just the original.
        let _ = fs::copy(pdf_path(&payload.id), &out_path);
    }
    let bytes = fs::read(&out_path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let base = orig
        .name
        .trim_end_matches(".pdf")
        .trim_end_matches(".PDF")
        .to_string();
    let new_name = format!("{base} (edited).pdf");
    entries.push(PdfEntry {
        id: new_id.clone(),
        name: new_name.clone(),
        size: bytes.len(),
    });
    save_json(PDF_INDEX_FILE, &entries)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(serde_json::json!({
        "id": new_id,
        "name": new_name,
        "size": bytes.len(),
    })))
}

async fn delete_pdf(
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad pdf id".into()));
    }
    let mut entries: Vec<PdfEntry> = load_json(PDF_INDEX_FILE);
    let before = entries.len();
    entries.retain(|e| e.id != id);
    if entries.len() == before {
        return Err((StatusCode::NOT_FOUND, "PDF not found".into()));
    }
    let _ = fs::remove_file(pdf_path(&id));
    save_json(PDF_INDEX_FILE, &entries)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(StatusCode::NO_CONTENT)
}

// Search saved chats by title or message content (case-insensitive).
async fn search_chats(Query(params): Query<std::collections::HashMap<String, String>>) -> Json<Vec<ChatSummary>> {
    let q = params.get("q").cloned().unwrap_or_default().to_lowercase();
    let mut out = Vec::new();
    if q.is_empty() {
        return Json(out);
    }
    if let Ok(entries) = fs::read_dir(CHATS_DIR) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            if let Ok(data) = fs::read_to_string(&p) {
                if let Ok(c) = serde_json::from_str::<Chat>(&data) {
                    let hay = format!(
                        "{}\n{}",
                        c.title,
                        c.messages
                            .iter()
                            .map(|m| m.content.as_str())
                            .collect::<Vec<_>>()
                            .join("\n")
                    )
                    .to_lowercase();
                    if hay.contains(&q) {
                        out.push(ChatSummary {
                            id: c.id,
                            title: c.title,
                            updated_at: c.updated_at,
                        });
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Json(out)
}

async fn list_endpoints() -> Json<Vec<EndpointPublic>> {
    let eps: Vec<Endpoint> = load_json(ENDPOINTS_FILE);
    Json(
        eps.iter()
            .map(|e| EndpointPublic {
                id: e.id.clone(),
                name: e.name.clone(),
                base_url: e.base_url.clone(),
                model: e.model.clone(),
                provider: e.provider.clone(),
                models: clean_model_list(e.models.clone()),
                models_fetched_at: e.models_fetched_at,
            })
            .collect(),
    )
}

async fn create_endpoint(
    Json(payload): Json<EndpointCreate>,
) -> Result<Json<EndpointPublic>, (StatusCode, String)> {
    let name = payload.name.trim().to_string();
    let base_url = payload.base_url.trim().trim_end_matches('/').to_string();
    let api_key = payload.api_key.trim().to_string();
    let model = payload.model.trim().to_string();
    let provider = payload.provider.trim().to_string();
    let provider = if preset_for(&provider).is_some() { provider } else { "custom".to_string() };
    let models = clean_model_list(payload.models);
    let models_fetched_at = if models.is_empty() { 0 } else { now_secs() };
    if name.is_empty() || base_url.is_empty() || model.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "name, base URL and model are required".into()));
    }
    if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
        return Err((StatusCode::BAD_REQUEST, "base URL must start with http:// or https://".into()));
    }
    let mut eps: Vec<Endpoint> = load_json(ENDPOINTS_FILE);
    let ep = Endpoint {
        id: gen_id(),
        name: name.clone(),
        base_url: base_url.clone(),
        api_key,
        model: model.clone(),
        provider: provider.clone(),
        models: models.clone(),
        models_fetched_at,
    };
    eps.push(ep.clone());
    save_json(ENDPOINTS_FILE, &eps).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(EndpointPublic {
        id: ep.id,
        name,
        base_url,
        model,
        provider,
        models,
        models_fetched_at,
    }))
}

async fn delete_endpoint(Path(id): Path<String>) -> Result<StatusCode, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let mut eps: Vec<Endpoint> = load_json(ENDPOINTS_FILE);
    let before = eps.len();
    eps.retain(|e| e.id != id);
    if eps.len() == before {
        return Err((StatusCode::NOT_FOUND, "endpoint not found".into()));
    }
    save_json(ENDPOINTS_FILE, &eps).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(StatusCode::NO_CONTENT)
}

/// Clean a model list: drop empties, dedupe, sort. Used for cached lists
/// sent by the browser and for parsed /models responses.
fn clean_model_list(models: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in models {
        let m = m.trim().to_string();
        // Google's /models route returns resource names like
        // "models/gemini-2.5-flash"; chat completions expects the bare name,
        // so strip the prefix. Other providers return bare ids already.
        let m = m.strip_prefix("models/").unwrap_or(&m).to_string();
        if !m.is_empty() && !out.contains(&m) {
            out.push(m);
        }
    }
    out.sort();
    out
}

/// Parse an OpenAI-style model list response into model ids.
/// Accepts {"data":[{"id":...}]}, {"models":[...]}, or a bare ["..."] array.
/// Unknown shapes return an empty list. Pure function, unit-tested.
fn parse_models_list(v: &serde_json::Value) -> Vec<String> {
    let mut raw: Vec<String> = Vec::new();
    if let Some(arr) = v.get("data").and_then(|d| d.as_array()) {
        for m in arr {
            if let Some(id) = m.get("id").and_then(|i| i.as_str()) {
                raw.push(id.to_string());
            }
        }
    } else if let Some(arr) = v.get("models").and_then(|m| m.as_array()) {
        for m in arr {
            if let Some(id) = m.as_str() {
                raw.push(id.to_string());
            }
        }
    } else if let Some(arr) = v.as_array() {
        for m in arr {
            if let Some(id) = m.as_str() {
                raw.push(id.to_string());
            }
        }
    }
    clean_model_list(raw)
}

fn models_list_url(base_url: &str) -> String {
    format!("{}/models", base_url.trim_end_matches('/'))
}

/// Fetch the model list from an OpenAI-compatible endpoint.
/// The API key is only ever sent to the endpoint itself, never logged or
/// returned. Errors are generic so no key or URL detail leaks out.
async fn fetch_endpoint_models(base_url: &str, api_key: &str) -> Result<Vec<String>, String> {
    let url = models_list_url(base_url);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|_| "could not start request".to_string())?;
    let mut req = client.get(&url).bearer_auth(api_key);
    if is_copilot_base(base_url) {
        req = with_copilot_headers(req, false, false);
    }
    let resp = req
        .send()
        .await
        .map_err(|_| "could not reach endpoint".to_string())?;
    if !resp.status().is_success() {
        return Err(format!("endpoint returned HTTP {}", resp.status().as_u16()));
    }
    let v: serde_json::Value = resp
        .json()
        .await
        .map_err(|_| "endpoint did not return a model list".to_string())?;
    let models = parse_models_list(&v);
    if models.is_empty() {
        return Err("endpoint did not return a model list".to_string());
    }
    Ok(models)
}

#[derive(Serialize)]
struct ModelsListResponse {
    models: Vec<String>,
}

#[derive(Deserialize)]
struct EndpointModelsRequest {
    base_url: String,
    api_key: String,
}

/// Fetch models for a NEW endpoint (key typed in the form, not saved yet).
async fn fetch_new_endpoint_models(
    Json(payload): Json<EndpointModelsRequest>,
) -> Result<Json<ModelsListResponse>, (StatusCode, String)> {
    let base_url = payload.base_url.trim().to_string();
    if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
        return Err((StatusCode::BAD_REQUEST, "base URL must start with http:// or https://".into()));
    }
    let models = fetch_endpoint_models(&base_url, payload.api_key.trim())
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(ModelsListResponse { models }))
}

/// Fetch models for a SAVED endpoint using its stored key, and cache them.
async fn fetch_saved_endpoint_models(
    Path(id): Path<String>,
) -> Result<Json<ModelsListResponse>, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let eps: Vec<Endpoint> = load_json(ENDPOINTS_FILE);
    let (base_url, api_key) = match eps.iter().find(|e| e.id == id) {
        Some(e) => (e.base_url.clone(), e.api_key.clone()),
        None => return Err((StatusCode::NOT_FOUND, "endpoint not found".into())),
    };
    let models = fetch_endpoint_models(&base_url, &api_key)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e))?;
    let mut eps = eps;
    if let Some(ep) = eps.iter_mut().find(|e| e.id == id) {
        ep.models = models.clone();
        ep.models_fetched_at = now_secs();
    }
    save_json(ENDPOINTS_FILE, &eps).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(ModelsListResponse { models }))
}

#[derive(Deserialize)]
struct EndpointUpdate {
    #[serde(default)]
    model: Option<String>,
}

/// Change a saved endpoint's selected model (used when Steven picks a
/// different model from that endpoint's list in the model picker).
async fn update_endpoint(
    Path(id): Path<String>,
    Json(payload): Json<EndpointUpdate>,
) -> Result<Json<EndpointPublic>, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let mut eps: Vec<Endpoint> = load_json(ENDPOINTS_FILE);
    let idx = match eps.iter().position(|e| e.id == id) {
        Some(i) => i,
        None => return Err((StatusCode::NOT_FOUND, "endpoint not found".into())),
    };
    if let Some(model) = payload.model {
        let model = model.trim().to_string();
        if model.is_empty() {
            return Err((StatusCode::BAD_REQUEST, "model is empty".into()));
        }
        eps[idx].model = model;
    }
    save_json(ENDPOINTS_FILE, &eps).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let ep = &eps[idx];
    Ok(Json(EndpointPublic {
        id: ep.id.clone(),
        name: ep.name.clone(),
        base_url: ep.base_url.clone(),
        model: ep.model.clone(),
        provider: ep.provider.clone(),
        models: ep.models.clone(),
        models_fetched_at: ep.models_fetched_at,
    }))
}

async fn list_memories() -> Json<Vec<Memory>> {
    Json(load_json(MEMORIES_FILE))
}

async fn create_memory(
    Json(payload): Json<MemoryCreate>,
) -> Result<Json<Memory>, (StatusCode, String)> {
    let text = payload.text.trim().to_string();
    if text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty memory".into()));
    }
    if text.len() > 500 {
        return Err((StatusCode::BAD_REQUEST, "memory too long".into()));
    }
    let mut mems: Vec<Memory> = load_json(MEMORIES_FILE);
    let mem = Memory {
        id: gen_id(),
        text,
    };
    mems.push(mem.clone());
    save_json(MEMORIES_FILE, &mems).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(mem))
}

async fn delete_memory(Path(id): Path<String>) -> Result<StatusCode, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let mut mems: Vec<Memory> = load_json(MEMORIES_FILE);
    let before = mems.len();
    mems.retain(|m| m.id != id);
    if mems.len() == before {
        return Err((StatusCode::NOT_FOUND, "memory not found".into()));
    }
    save_json(MEMORIES_FILE, &mems).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_skills() -> Json<Vec<Skill>> {
    Json(load_json(SKILLS_FILE))
}

async fn create_skill(
    Json(payload): Json<SkillCreate>,
) -> Result<Json<Skill>, (StatusCode, String)> {
    let name = payload.name.trim().to_string();
    let instruction = payload.instruction.trim().to_string();
    if name.is_empty() || instruction.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "name and instruction are required".into()));
    }
    if instruction.len() > 2000 {
        return Err((StatusCode::BAD_REQUEST, "instruction too long".into()));
    }
    let mut skills: Vec<Skill> = load_json(SKILLS_FILE);
    let skill = Skill {
        id: gen_id(),
        name,
        instruction,
    };
    skills.push(skill.clone());
    save_json(SKILLS_FILE, &skills).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(skill))
}

async fn delete_skill(Path(id): Path<String>) -> Result<StatusCode, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let mut skills: Vec<Skill> = load_json(SKILLS_FILE);
    let before = skills.len();
    skills.retain(|s| s.id != id);
    if skills.len() == before {
        return Err((StatusCode::NOT_FOUND, "skill not found".into()));
    }
    save_json(SKILLS_FILE, &skills).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(StatusCode::NO_CONTENT)
}

// ---------- Notes ----------

async fn list_notes() -> Json<Vec<Note>> {
    let mut notes: Vec<Note> = load_json(NOTES_FILE);
    notes.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Json(notes)
}

async fn create_note(
    Json(payload): Json<NoteCreate>,
) -> Result<Json<Note>, (StatusCode, String)> {
    let title = payload.title.trim().to_string();
    let body = payload.body.trim().to_string();
    if title.is_empty() && body.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty note".into()));
    }
    if title.len() > 200 || body.len() > 20000 {
        return Err((StatusCode::BAD_REQUEST, "note too long".into()));
    }
    let mut notes: Vec<Note> = load_json(NOTES_FILE);
    let note = Note {
        id: gen_id(),
        title: if title.is_empty() {
            "Untitled".to_string()
        } else {
            title
        },
        body,
        updated_at: now_unix(),
    };
    notes.push(note.clone());
    save_json(NOTES_FILE, &notes).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(note))
}

async fn update_note(
    Path(id): Path<String>,
    Json(payload): Json<NoteUpdate>,
) -> Result<Json<Note>, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let mut notes: Vec<Note> = load_json(NOTES_FILE);
    let note = notes
        .iter_mut()
        .find(|n| n.id == id)
        .ok_or((StatusCode::NOT_FOUND, "note not found".to_string()))?;
    if let Some(t) = &payload.title {
        let t = t.trim();
        if !t.is_empty() {
            if t.len() > 200 {
                return Err((StatusCode::BAD_REQUEST, "title too long".into()));
            }
            note.title = t.to_string();
        }
    }
    if let Some(b) = &payload.body {
        if b.len() > 20000 {
            return Err((StatusCode::BAD_REQUEST, "note too long".into()));
        }
        note.body = b.trim().to_string();
    }
    note.updated_at = now_unix();
    let out = note.clone();
    save_json(NOTES_FILE, &notes).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(out))
}

async fn delete_note(Path(id): Path<String>) -> Result<StatusCode, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let mut notes: Vec<Note> = load_json(NOTES_FILE);
    let before = notes.len();
    notes.retain(|n| n.id != id);
    if notes.len() == before {
        return Err((StatusCode::NOT_FOUND, "note not found".into()));
    }
    save_json(NOTES_FILE, &notes).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(StatusCode::NO_CONTENT)
}

// ---------- Calendar ----------
// Local-only events, stored in calendar.json on the phone. No Google
// Calendar / OAuth — the app works offline and Steven's data stays here.
const CALENDAR_FILE: &str = "calendar.json";

#[derive(Serialize, Deserialize, Clone)]
struct CalendarEvent {
    id: String,
    // Date as YYYY-MM-DD, time as HH:MM or "" for all-day.
    title: String,
    date: String,
    #[serde(default)]
    time: String,
    #[serde(default)]
    notes: String,
}

#[derive(Deserialize)]
struct CalendarEventCreate {
    title: String,
    date: String,
    #[serde(default)]
    time: String,
    #[serde(default)]
    notes: String,
}

fn valid_cal_date(d: &str) -> bool {
    // Strict YYYY-MM-DD with a real month/day range.
    let b: Vec<&str> = d.split('-').collect();
    if b.len() != 3 || b[0].len() != 4 || b[1].len() != 2 || b[2].len() != 2 {
        return false;
    }
    if !b.iter().all(|p| p.bytes().all(|c| c.is_ascii_digit())) {
        return false;
    }
    let m: u32 = b[1].parse().unwrap_or(0);
    let day: u32 = b[2].parse().unwrap_or(0);
    (1..=12).contains(&m) && (1..=31).contains(&day)
}

async fn list_calendar_events() -> Json<Vec<CalendarEvent>> {
    let mut evs: Vec<CalendarEvent> = load_json(CALENDAR_FILE);
    evs.sort_by(|a, b| (a.date.clone(), a.time.clone()).cmp(&(b.date.clone(), b.time.clone())));
    Json(evs)
}

async fn create_calendar_event(
    Json(payload): Json<CalendarEventCreate>,
) -> Result<Json<CalendarEvent>, (StatusCode, String)> {
    let title = payload.title.trim().to_string();
    if title.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "title is empty".into()));
    }
    if !valid_cal_date(payload.date.trim()) {
        return Err((StatusCode::BAD_REQUEST, "date must be YYYY-MM-DD".into()));
    }
    let ev = CalendarEvent {
        id: gen_id(),
        title,
        date: payload.date.trim().to_string(),
        time: payload.time.trim().to_string(),
        notes: payload.notes.trim().to_string(),
    };
    let mut evs: Vec<CalendarEvent> = load_json(CALENDAR_FILE);
    evs.push(ev.clone());
    save_json(CALENDAR_FILE, &evs).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(ev))
}

async fn delete_calendar_event(
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    if !valid_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let mut evs: Vec<CalendarEvent> = load_json(CALENDAR_FILE);
    let before = evs.len();
    evs.retain(|e| e.id != id);
    if evs.len() == before {
        return Err((StatusCode::NOT_FOUND, "event not found".into()));
    }
    save_json(CALENDAR_FILE, &evs).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(StatusCode::NO_CONTENT)
}

// ---------- Cookbook: model manager ----------
// Mirrors the Odysseus Cookbook (model serving): search HuggingFace, list
// cached models, download via Ollama or as a GGUF, list running models,
// stop a model. Phone-relevant subset only — no remote hosts, vLLM or GPU.

/// List the .gguf files in a HuggingFace repo (from its `siblings`), Q4_K_M
/// first, then smallest-first. Empty when the repo has no GGUF files.
fn hf_gguf_files(model: &serde_json::Value) -> Vec<GgufFileInfo> {
    let mut files: Vec<GgufFileInfo> = model
        .get("siblings")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| {
                    let filename = s.get("rfilename")?.as_str()?.to_string();
                    if !filename.to_lowercase().ends_with(".gguf") {
                        return None;
                    }
                    Some(GgufFileInfo {
                        filename,
                        size: s.get("size").and_then(|v| v.as_i64()),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort_by_key(|f| {
        let is_q4 = if f.filename.contains("Q4_K_M") { 0 } else { 1 };
        (is_q4, f.size.unwrap_or(i64::MAX))
    });
    files
}

fn hf_model_size_bytes(v: &serde_json::Value) -> Option<i64> {
    // Prefer the safetensors total (full=true returns it). For GGUF repos
    // there is no safetensors object: use one quant file, not the sum of
    // all of them — a repo holds many quant variants of the same model and
    // only one is downloaded. Prefer Q4_K_M, else the largest .gguf file.
    if let Some(total) = v
        .get("safetensors")
        .and_then(|s| s.get("total"))
        .and_then(|x| x.as_i64())
    {
        if total > 0 {
            return Some(total);
        }
    }
    let mut q4k: i64 = 0;
    let mut largest_gguf: i64 = 0;
    if let Some(sibs) = v.get("siblings").and_then(|s| s.as_array()) {
        for s in sibs {
            let name = s.get("rfilename").and_then(|x| x.as_str()).unwrap_or("");
            if name.ends_with(".gguf") {
                if let Some(sz) = s.get("size").and_then(|x| x.as_i64()) {
                    if name.contains("Q4_K_M") && sz > q4k {
                        q4k = sz;
                    }
                    if sz > largest_gguf {
                        largest_gguf = sz;
                    }
                }
            }
        }
    }
    if q4k > 0 {
        return Some(q4k);
    }
    if largest_gguf > 0 {
        return Some(largest_gguf);
    }
    // Fall back: sum the safetensors/bin shards (a model needs all of them).
    let mut total: i64 = 0;
    if let Some(sibs) = v.get("siblings").and_then(|s| s.as_array()) {
        for s in sibs {
            let name = s.get("rfilename").and_then(|x| x.as_str()).unwrap_or("");
            if name.ends_with(".safetensors") || name.ends_with(".bin") {
                if let Some(sz) = s.get("size").and_then(|x| x.as_i64()) {
                    total = total.saturating_add(sz);
                }
            }
        }
    }
    if total > 0 {
        return Some(total);
    }
    // GGUF repos: the API gives no per-file sizes. gguf.total is the
    // full model size in bytes — an upper bound for any single quant.
    if let Some(total) = v
        .get("gguf")
        .and_then(|g| g.get("total"))
        .and_then(|x| x.as_i64())
    {
        if total > 0 {
            return Some(total);
        }
    }
    None
}

fn hf_model_from(v: &serde_json::Value) -> HfModel {
    // The HF list API has no `author` field; the author is the part of the
    // id before `/` (e.g. "Qwen" in "Qwen/Qwen3-0.6B"). It also has no
    // `lastModified`; use `createdAt` when present.
    let id = v
        .get("id")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let author_field = v
        .get("author")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let author = if author_field.is_empty() {
        id.split('/').next().unwrap_or("").to_string()
    } else {
        author_field
    };
    let last_modified = v
        .get("lastModified")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            v.get("createdAt")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_default();
    HfModel {
        id,
        author,
        downloads: v.get("downloads").and_then(|x| x.as_i64()).unwrap_or(0),
        likes: v.get("likes").and_then(|x| x.as_i64()).unwrap_or(0),
        pipeline_tag: v
            .get("pipeline_tag")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        last_modified,
        size_bytes: hf_model_size_bytes(v),
        gguf_files: hf_gguf_files(v),
    }
}

async fn cookbook_search(
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<HfModel>>, (StatusCode, String)> {
    let q = params
        .get("q")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if q.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "search query required".into()));
    }
    let client = reqwest::Client::new();
    let resp = client
        .get(HF_API_URL)
        .query(&[
            ("search", q.as_str()),
            ("limit", "20"),
            ("full", "true"),
            ("sort", "downloads"),
            ("direction", "-1"),
        ])
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("HuggingFace unreachable: {e}")))?;
    if !resp.status().is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("HuggingFace HTTP {}", resp.status()),
        ));
    }
    let v: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("bad HF response: {e}")))?;
    let arr = v.as_array().cloned().unwrap_or_default();
    let mut models: Vec<HfModel> = arr.iter().map(hf_model_from).collect();
    // The list API returns no file sizes even with full=true, so fetch
    // each model's own endpoint for safetensors.total / gguf.total.
    // All fetches run concurrently, so the search stays fast.
    let futs: Vec<_> = models
        .iter()
        .map(|m| {
            let client = client.clone();
            let id = m.id.clone();
            async move {
                let url = format!("{HF_API_URL}/{id}");
                let (size, ggufs) = match client.get(&url).send().await {
                    Ok(resp) if resp.status().is_success() => match resp
                        .json::<serde_json::Value>()
                        .await
                    {
                        Ok(v) => (hf_model_size_bytes(&v), hf_gguf_files(&v)),
                        Err(_) => (None, Vec::new()),
                    },
                    _ => (None, Vec::new()),
                };
                (id, size, ggufs)
            }
        })
        .collect();
    let sizes = futures_util::future::join_all(futs).await;
    for (id, size, ggufs) in sizes {
        if let Some(m) = models.iter_mut().find(|m| m.id == id) {
            if size.is_some() {
                m.size_bytes = size;
            }
            if !ggufs.is_empty() {
                m.gguf_files = ggufs;
            }
        }
    }
    Ok(Json(models))
}

fn disk_space_bytes(path: &str) -> (Option<u64>, Option<u64>) {
    // `df -B1 <path>`: second line, columns are filesystem, total, used, available.
    let out = std::process::Command::new("df").args(["-B1", path]).output();
    if let Ok(o) = out {
        let s = String::from_utf8_lossy(&o.stdout);
        for line in s.lines().skip(1) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let total = parts[1].parse::<u64>().ok();
                let avail = parts[3].parse::<u64>().ok();
                return (avail, total);
            }
        }
    }
    (None, None)
}

async fn cookbook_cached() -> Json<CachedModels> {
    let client = reqwest::Client::new();
    let mut ollama: Vec<CachedOllamaModel> = Vec::new();
    if let Ok(resp) = client.get(OLLAMA_TAGS_URL).send().await {
        if let Ok(v) = resp.json::<serde_json::Value>().await {
            if let Some(arr) = v.get("models").and_then(|m| m.as_array()) {
                for m in arr {
                    ollama.push(CachedOllamaModel {
                        name: m
                            .get("name")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                        size: m.get("size").and_then(|x| x.as_i64()).unwrap_or(0),
                        modified_at: m
                            .get("modified_at")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                        parameter_size: m
                            .get("details")
                            .and_then(|d| d.get("parameter_size"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                        quantization: m
                            .get("details")
                            .and_then(|d| d.get("quantization_level"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                    });
                }
            }
        }
    }
    let mut gguf: Vec<CachedGguf> = Vec::new();
    if let Ok(rd) = fs::read_dir(MODELS_DIR) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.to_lowercase().ends_with(".gguf") {
                continue;
            }
            if let Ok(md) = e.metadata() {
                gguf.push(CachedGguf {
                    name,
                    size: md.len(),
                    modified: md
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                });
            }
        }
    }
    gguf.sort_by(|a, b| b.modified.cmp(&a.modified));
    let (free, total) = disk_space_bytes(MODELS_DIR);
    Json(CachedModels {
        ollama,
        gguf,
        disk_free_bytes: free,
        disk_total_bytes: total,
    })
}

fn find_llama_server() -> Option<serde_json::Value> {
    // Scan /proc for a llama-server process; report its pid and command.
    if let Ok(rd) = fs::read_dir("/proc") {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let cmdline_path = format!("/proc/{}/cmdline", name);
            if let Ok(bytes) = fs::read(&cmdline_path) {
                let cmd = String::from_utf8_lossy(&bytes).replace('\0', " ");
                if cmd.contains("llama-server") {
                    return Some(serde_json::json!({
                        "pid": name.parse::<u32>().unwrap_or(0),
                        "cmd": cmd.trim().to_string(),
                    }));
                }
            }
        }
    }
    None
}

async fn cookbook_running() -> Json<RunningModels> {
    let client = reqwest::Client::new();
    let mut ollama: Vec<serde_json::Value> = Vec::new();
    if let Ok(resp) = client.get(OLLAMA_PS_URL).send().await {
        if let Ok(v) = resp.json::<serde_json::Value>().await {
            if let Some(arr) = v.get("models").and_then(|m| m.as_array()) {
                ollama = arr.clone();
            }
        }
    }
    Json(RunningModels {
        ollama,
        llama_server: find_llama_server(),
    })
}

async fn cookbook_download(
    Json(payload): Json<DownloadRequest>,
) -> Result<Json<DownloadTask>, (StatusCode, String)> {
    let kind = payload.kind.trim().to_string();
    let id = gen_id();
    let task = match kind.as_str() {
        "ollama" => {
            let name = payload.name.unwrap_or_default().trim().to_string();
            if name.is_empty() {
                return Err((StatusCode::BAD_REQUEST, "model name required".into()));
            }
            // An Ollama registry name never contains '/'; a HuggingFace id
            // always does. Refuse to ollama-pull an HF id — it fails with a
            // confusing "pull model manifest: file does not exist" instead.
            if name.contains('/') {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "not an Ollama model name (it looks like a HuggingFace id). Download the GGUF file instead, or type an Ollama name like qwen3:0.6b".into(),
                ));
            }
            DownloadTask {
                id: id.clone(),
                kind: "ollama".into(),
                name,
                status: "downloading".into(),
                detail: String::new(),
            }
        }
        "gguf" => {
            let url = payload.url.unwrap_or_default().trim().to_string();
            let filename = payload.filename.unwrap_or_default().trim().to_string();
            if url.is_empty() || filename.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "url and filename required".into(),
                ));
            }
            if filename.contains('/')
                || filename.contains('\\')
                || filename.contains("..")
                || !filename.to_lowercase().ends_with(".gguf")
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "filename must be a plain .gguf name".into(),
                ));
            }
            if !url.starts_with("https://") && !url.starts_with("http://") {
                return Err((StatusCode::BAD_REQUEST, "url must be http(s)".into()));
            }
            DownloadTask {
                id: id.clone(),
                kind: "gguf".into(),
                name: filename,
                status: "downloading".into(),
                detail: url,
            }
        }
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "kind must be ollama or gguf".into(),
            ))
        }
    };
    {
        let mut map = download_tasks().lock().unwrap();
        map.insert(id.clone(), task.clone());
    }
    let tid = id.clone();
    // Clone the fields the background task needs; `task` itself is returned
    // to the caller, so it must not be moved into the spawned future.
    let task_kind = task.kind.clone();
    let task_name = task.name.clone();
    let task_detail = task.detail.clone();
    tokio::spawn(async move {
        let result: Result<(), String> = async {
            match task_kind.as_str() {
                "ollama" => {
                    let client = reqwest::Client::new();
                    let resp = client
                        .post(OLLAMA_PULL_URL)
                        .json(&serde_json::json!({"name": task_name, "stream": false}))
                        .send()
                        .await
                        .map_err(|e| e.to_string())?;
                    if !resp.status().is_success() {
                        let status = resp.status();
                        let t = resp.text().await.unwrap_or_default();
                        let short: String = t.chars().take(300).collect();
                        return Err(format!("Ollama pull HTTP {}: {}", status, short));
                    }
                    Ok(())
                }
                _ => {
                    // GGUF direct download, streamed to /root/models/<filename>.
                    let url = task_detail.clone();
                    let dest = format!("{}/{}", MODELS_DIR, task_name);
                    let client = reqwest::Client::new();
                    let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
                    if !resp.status().is_success() {
                        return Err(format!("download HTTP {}", resp.status()));
                    }
                    let _ = fs::create_dir_all(MODELS_DIR);
                    let mut file = tokio::fs::File::create(&dest)
                        .await
                        .map_err(|e| e.to_string())?;
                    let mut stream = resp.bytes_stream();
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk.map_err(|e| e.to_string())?;
                        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
                            .await
                            .map_err(|e| e.to_string())?;
                    }
                    tokio::io::AsyncWriteExt::flush(&mut file)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok(())
                }
            }
        }
        .await;
        let mut map = download_tasks().lock().unwrap();
        if let Some(t) = map.get_mut(&tid) {
            match result {
                Ok(()) => {
                    t.status = "done".into();
                }
                Err(e) => {
                    t.status = "failed".into();
                    t.detail = e;
                }
            }
        }
    });
    Ok(Json(task))
}

async fn cookbook_downloads() -> Json<Vec<DownloadTask>> {
    let map = download_tasks().lock().unwrap();
    let mut tasks: Vec<DownloadTask> = map.values().cloned().collect();
    tasks.sort_by(|a, b| b.id.cmp(&a.id));
    Json(tasks)
}

async fn cookbook_import(
    Json(body): Json<serde_json::Value>,
) -> Result<Json<DownloadTask>, (StatusCode, String)> {
    let filename = body
        .get("filename")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let name = body
        .get("name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| ollama_name_from_repo(&filename));
    let name = name.trim().to_lowercase();
    if filename.is_empty()
        || filename.contains('/')
        || filename.contains("..")
        || !filename.to_lowercase().ends_with(".gguf")
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "filename must be a plain .gguf file name in /root/models".into(),
        ));
    }
    if name.is_empty() || name.contains('/') {
        return Err((StatusCode::BAD_REQUEST, "bad model name".into()));
    }
    let path = format!("{}/{}", MODELS_DIR, filename);
    if !std::path::Path::new(&path).exists() {
        return Err((
            StatusCode::NOT_FOUND,
            "GGUF file not found — download it first".into(),
        ));
    }
    let id = gen_id();
    let task = DownloadTask {
        id: id.clone(),
        kind: "import".into(),
        name: name.clone(),
        status: "importing".into(),
        detail: filename.clone(),
    };
    {
        let mut map = download_tasks().lock().unwrap();
        map.insert(id.clone(), task.clone());
    }
    let tid = id.clone();
    let import_name = name.clone();
    let import_path = path.clone();
    tokio::spawn(async move {
        let result: Result<(), String> = async {
            // Use the `ollama` CLI, not the HTTP Create API: the CLI reads
            // the GGUF and uploads it to the daemon. The daemon itself
            // cannot see /root/models, so a Modelfile FROM path via the
            // HTTP API fails with "neither 'from' or 'files' was specified".
            let modelfile_path = format!("/tmp/Modelfile-{}", tid);
            tokio::fs::write(&modelfile_path, format!("FROM {}\n", import_path))
                .await
                .map_err(|e| e.to_string())?;
            let output = tokio::process::Command::new("ollama")
                .args(["create", &import_name, "-f", &modelfile_path])
                .output()
                .await
                .map_err(|e| format!("failed to run ollama: {}", e))?;
            let _ = tokio::fs::remove_file(&modelfile_path).await;
            if !output.status.success() {
                let err = String::from_utf8_lossy(&output.stderr).to_string();
                let out = String::from_utf8_lossy(&output.stdout).to_string();
                let msg = if !err.trim().is_empty() { err } else { out };
                let short: String = msg.chars().take(300).collect();
                return Err(format!("ollama create failed: {}", short));
            }
            Ok(())
        }
        .await;
        let mut map = download_tasks().lock().unwrap();
        if let Some(t) = map.get_mut(&tid) {
            match result {
                Ok(()) => {
                    t.status = "done".into();
                    t.detail = format!("imported as {}", import_name);
                }
                Err(e) => {
                    t.status = "failed".into();
                    t.detail = e;
                }
            }
        }
    });
    Ok(Json(task))
}

/// Derive a safe Ollama model name from a HuggingFace repo id or a GGUF
/// filename: lowercase, keep [a-z0-9._-], strip a trailing .gguf.
fn ollama_name_from_repo(id: &str) -> String {
    let base = id.rsplit('/').next().unwrap_or(id);
    let base = base.strip_suffix(".gguf").unwrap_or(base);
    let cleaned: String = base
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    cleaned
        .trim_matches(|c| c == '-' || c == '.' || c == '_')
        .to_string()
}

async fn cookbook_stop(
    Json(payload): Json<CookbookStopRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    match payload.kind.trim() {
        "ollama" => {
            let name = payload.name.unwrap_or_default().trim().to_string();
            if name.is_empty() {
                return Err((StatusCode::BAD_REQUEST, "model name required".into()));
            }
            // Unload the model: a generate call with keep_alive 0.
            let client = reqwest::Client::new();
            let resp = client
                .post(OLLAMA_GENERATE_URL)
                .json(&serde_json::json!({"model": name, "keep_alive": 0}))
                .send()
                .await
                .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Ollama unreachable: {e}")))?;
            if !resp.status().is_success() {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    format!("Ollama HTTP {}", resp.status()),
                ));
            }
            Ok(Json(serde_json::json!({"stopped": name})))
        }
        "llama" => {
            let pid = payload
                .pid
                .ok_or((StatusCode::BAD_REQUEST, "pid required".to_string()))?;
            if pid == 0 {
                return Err((StatusCode::BAD_REQUEST, "bad pid".into()));
            }
            // Only kill it if it really is a llama-server process.
            let cmdline = fs::read(format!("/proc/{}/cmdline", pid)).unwrap_or_default();
            let cmd = String::from_utf8_lossy(&cmdline).replace('\0', " ");
            if !cmd.contains("llama-server") {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "that pid is not llama-server".into(),
                ));
            }
            let out = std::process::Command::new("kill")
                .arg(pid.to_string())
                .output()
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            if !out.status.success() {
                return Err((StatusCode::INTERNAL_SERVER_ERROR, "kill failed".into()));
            }
            Ok(Json(serde_json::json!({"stopped_pid": pid})))
        }
        _ => Err((
            StatusCode::BAD_REQUEST,
            "kind must be ollama or llama".into(),
        )),
    }
}

// ---------- Deep Research ----------
// Iterative engine (see the prompts above). POST /api/research streams
// SSE progress events and a final report; POST /api/research/cancel stops
// a run. One model call is wrapped in a 150s timeout so a hung call
// degrades to a partial report instead of hanging the run. Overall run
// budget is 6 minutes; the loop checks it before every step.

fn research_event(tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>, v: serde_json::Value) {
    // Non-blocking: progress events are expendable if the browser is slow;
    // the receiver drains continuously, so in practice nothing is dropped.
    let _ = tx.try_send(Ok(Event::default().data(
        serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string()),
    )));
}

async fn research_call(
    endpoint: &Option<Endpoint>,
    model: &str,
    msgs: &[OllamaMessage],
) -> Result<String, String> {
    // Research calls get a long timeout: slow cloud models (NVIDIA) need
    // ~150-200s to write a synthesis, and the 120s default killed them.
    match tokio::time::timeout(
        std::time::Duration::from_secs(300),
        call_model_with_retry(endpoint, model, msgs, 300),
    )
    .await
    {
        Ok(r) => r,
        Err(_) => Err("model call timed out".to_string()),
    }
}

async fn research(
    Json(payload): Json<ResearchRequest>,
) -> Result<Sse<ReceiverStream<Result<Event, Infallible>>>, (StatusCode, String)> {
    let topic = payload.topic.trim().to_string();
    if topic.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty topic".into()));
    }
    if topic.len() > 500 {
        return Err((StatusCode::BAD_REQUEST, "topic too long".into()));
    }
    let endpoint: Option<Endpoint> = match &payload.endpoint_id {
        Some(id) if !id.is_empty() => {
            let eps: Vec<Endpoint> = load_json(ENDPOINTS_FILE);
            eps.into_iter().find(|e| &e.id == id)
        }
        _ => None,
    };
    // Deep Research needs a capable model. The on-phone models are too
    // small to drive a multi-round loop — block them with a clear message
    // instead of running a weak loop.
    if endpoint.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Deep Research needs a capable model. The local model is too small for this — pick a Gemini model from the model picker first.".into(),
        ));
    }
    let model = payload.model.clone().unwrap_or_default();
    let use_model = if model.trim().is_empty() {
        match &endpoint {
            Some(ep) => ep.model.clone(),
            None => DEFAULT_MODEL.to_string(),
        }
    } else {
        model
    };
    // Research options from the view; defaults copy Odysseus (see the
    // research_*_choice helpers). Brave with no API key falls back to Bing.
    let max_rounds = research_rounds_choice(payload.rounds);
    let format = payload.format.clone().unwrap_or_else(|| "full".to_string());
    let engine_picked = research_engine_choice(payload.engine.as_deref().unwrap_or("bing"));
    let engine_fell_back_from_brave = engine_picked == "brave";
    let engine: &'static str = if engine_fell_back_from_brave {
        "bing"
    } else {
        engine_picked
    };
    let engine_is_ddg = engine == "duckduckgo";
    let created_at = now_secs();
    let endpoint_name = endpoint
        .as_ref()
        .map(|e| e.name.clone())
        .unwrap_or_else(|| "Local Ollama".to_string());

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
    let run_id = research_new_id();
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    research_runs()
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "run registry locked".to_string()))?
        .insert(run_id.clone(), Arc::clone(&cancelled));

    tokio::spawn(async move {
        let send = |v: serde_json::Value| research_event(&tx, v);
        let run_started = Instant::now();
        // Save this run to History when it finishes (completed or failed).
        let save_entry = |report: String, sources: Vec<ResearchSource>, status: &str, error: Option<String>| {
            save_research_history_entry(ResearchHistoryEntry {
                id: run_id.clone(),
                topic: topic.clone(),
                report,
                sources,
                duration_secs: run_started.elapsed().as_secs(),
                created_at,
                model: use_model.clone(),
                endpoint: endpoint_name.clone(),
                rounds: max_rounds,
                format: format.clone(),
                engine: engine.to_string(),
                status: status.to_string(),
                error,
            });
        };
        send(serde_json::json!({"type": "started", "run_id": run_id}));
        if engine_fell_back_from_brave {
            send(serde_json::json!({"type": "note", "message": "Brave search needs an API key and none is configured — using Bing instead. (Odysseus defaults to SearXNG, self-hosted; no SearXNG server runs on this phone.)"}));
        }
        if engine_is_ddg {
            send(serde_json::json!({"type": "note", "message": "DuckDuckGo search is often blocked from this phone — falling back to Bing for any query it fails."}));
        }
        let deadline = Instant::now() + Duration::from_secs(900);
        let is_cancelled = || {
            cancelled.load(std::sync::atomic::Ordering::Relaxed)
                || Instant::now() > deadline
        };

        // 1. PLAN
        let plan_msgs = vec![OllamaMessage {
            role: "user".to_string(),
            content: RESEARCH_PLAN_PROMPT.replace("{topic}", &topic),
            images: None,
        }];
        let plan = match research_call(&endpoint, &use_model, &plan_msgs).await {
            Ok(text) => parse_plan_json(&text, &topic),
            Err(e) => {
                send(serde_json::json!({"type": "error", "message": format!("planning failed: {}", e)}));
                save_entry(String::new(), Vec::new(), "failed", Some(format!("planning failed: {}", e)));
                research_runs().lock().map(|mut m| { m.remove(&run_id); }).ok();
                return;
            }
        };
        send(serde_json::json!({"type": "plan", "plan": &plan}));

        let mut evolving = String::new();
        let mut sources: Vec<ResearchSource> = Vec::new();
        let mut seen_urls: Vec<String> = Vec::new();
        let mut stopped_early = false;

        // 2-4. ROUNDS: query-gen -> search -> fetch/extract -> synthesize -> stop
        for round in 1..=max_rounds {
            if is_cancelled() {
                break;
            }
            let report_so_far = if evolving.is_empty() {
                "Nothing yet — this is the first round.".to_string()
            } else {
                evolving.clone()
            };
            let plan_text = serde_json::to_string(&plan).unwrap_or_default();
            let q_msgs = vec![OllamaMessage {
                role: "user".to_string(),
                content: RESEARCH_QUERY_PROMPT
                    .replace("{topic}", &topic)
                    .replace("{plan}", &plan_text)
                    .replace("{report}", &report_so_far)
                    .replace("{round}", &round.to_string())
                    .replace("{n}", "2"),
                images: None,
            }];
            let queries = match research_call(&endpoint, &use_model, &q_msgs).await {
                Ok(text) => parse_queries_json(&text, &plan.sub_questions),
                Err(_) => plan.sub_questions.clone(),
            };
            let queries: Vec<String> = queries.into_iter().take(2).collect();
            if queries.is_empty() {
                break;
            }
            send(serde_json::json!({"type": "queries", "round": round, "queries": &queries}));

            let mut findings = String::new();
            let keywords = research_keywords(&topic, &queries);
            let mut fetched_this_round = 0;
            let mut first_query = true;
            for query in &queries {
                if is_cancelled() || fetched_this_round >= 6 {
                    break;
                }
                // Small pause between searches: rapid-fire queries trip
                // Bing's bot-check and come back empty.
                if !first_query {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
                first_query = false;
                let (results, fallback_note) = match web_search(query, engine, &keywords).await {
                    Ok((r, note)) => (r, note),
                    Err(e) => {
                        send(serde_json::json!({"type": "search", "query": query, "error": e}));
                        continue;
                    }
                };
                if let Some(note) = fallback_note {
                    send(serde_json::json!({"type": "note", "message": note}));
                }
                send(serde_json::json!({"type": "search", "query": query, "results": &results}));
                for r in results.into_iter().take(3) {
                    if is_cancelled() || fetched_this_round >= 6 {
                        break;
                    }
                    if seen_urls.iter().any(|u| u == &r.url) {
                        continue;
                    }
                    seen_urls.push(r.url.clone());
                    match fetch_page(&r.url, &keywords).await {
                        Ok((final_url, text)) => {
                            fetched_this_round += 1;
                            if !seen_urls.iter().any(|u| u == &final_url) {
                                seen_urls.push(final_url.clone());
                            }
                            sources.push(ResearchSource {
                                title: r.title.clone(),
                                url: final_url.clone(),
                            });
                            findings.push_str(&format!(
                                "\n\nSource: {} ({})\n{}",
                                r.title, final_url, text
                            ));
                            send(serde_json::json!({"type": "source", "title": r.title, "url": final_url, "chars": text.len()}));
                        }
                        Err(_) => continue,
                    }
                }
            }
            if findings.trim().is_empty() {
                send(serde_json::json!({"type": "synth", "round": round, "note": "no new sources this round"}));
            } else {
                // Cap the findings sent to the model: the NVIDIA model is
                // slow (~50 chars/s), and a full round of evidence (up to
                // 6 pages x 3000 chars) makes the synthesis call exceed the
                // 120s request timeout, which empties the whole report.
                // 4000 chars keeps a synthesis call under ~90s.
                let findings_capped: String = findings.chars().take(4000).collect();
                let s_msgs = vec![OllamaMessage {
                    role: "user".to_string(),
                    content: RESEARCH_SYNTH_PROMPT
                        .replace("{topic}", &topic)
                        .replace("{report}", &report_so_far)
                        .replace("{findings}", &findings_capped),
                    images: None,
                }];
                match research_call(&endpoint, &use_model, &s_msgs).await {
                    Ok(updated) => evolving = updated,
                    Err(e) => {
                        send(serde_json::json!({"type": "synth", "round": round, "note": format!("synthesis failed: {}", e)}));
                    }
                }
                send(serde_json::json!({"type": "synth", "round": round}));
            }

            if round < max_rounds && !is_cancelled() {
                let stop_msgs = vec![OllamaMessage {
                    role: "user".to_string(),
                    content: RESEARCH_STOP_PROMPT
                        .replace("{topic}", &topic)
                        .replace("{report}", &evolving)
                        .replace("{round}", &round.to_string())
                        .replace("{max_rounds}", &max_rounds.to_string()),
                    images: None,
                }];
                if let Ok(text) = research_call(&endpoint, &use_model, &stop_msgs).await {
                    let decision = text.trim().to_string();
                    send(serde_json::json!({"type": "stop", "decision": decision}));
                    if decision.to_lowercase().starts_with("yes") {
                        stopped_early = true;
                        break;
                    }
                }
            }
        }

        if is_cancelled() && cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            send(serde_json::json!({"type": "cancelled"}));
            save_entry(String::new(), Vec::new(), "failed", Some("Cancelled by user".to_string()));
            research_runs().lock().map(|mut m| { m.remove(&run_id); }).ok();
            return;
        }

        // 5. FINAL REPORT
        let base = if evolving.is_empty() {
            format!("No web evidence could be gathered for \"{}\".", topic)
        } else {
            evolving.clone()
        };
        let f_msgs = vec![
            OllamaMessage {
                role: "system".to_string(),
                content: RESEARCH_SYSTEM.to_string(),
                images: None,
            },
            OllamaMessage {
                role: "user".to_string(),
                content: research_format_prompt(&format)
                    .replace("{topic}", &topic)
                    .replace("{report}", &base),
                images: None,
            },
        ];
        let mut final_report = match research_call(&endpoint, &use_model, &f_msgs).await {
            Ok(r) => r,
            Err(_) => base.clone(),
        };
        if !sources.is_empty() {
            final_report.push_str("\n\n## Sources\n");
            for (i, s) in sources.iter().enumerate() {
                final_report.push_str(&format!("{}. [{}]({})\n", i + 1, s.title, s.url));
            }
        }
        let _ = stopped_early;
        let duration_secs = run_started.elapsed().as_secs();
        save_entry(final_report.clone(), sources.clone(), "completed", None);
        send(serde_json::json!({"type": "report", "report": final_report, "sources": sources, "duration_secs": duration_secs, "duration": research_format_duration(duration_secs)}));
        send(serde_json::json!({"type": "done"}));
        research_runs().lock().map(|mut m| { m.remove(&run_id); }).ok();
    });

    let sse_stream = ReceiverStream::new(rx);
    Ok(Sse::new(sse_stream))
}

async fn cancel_research(
    Json(payload): Json<ResearchCancelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let runs = research_runs()
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "run registry locked".to_string()))?;
    match runs.get(&payload.run_id) {
        Some(flag) => {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(Json(serde_json::json!({"cancelled": true})))
        }
        None => Err((StatusCode::NOT_FOUND, "run not found".into())),
    }
}

// ---------- Library: read one workspace text file ----------
// Read-only preview for the Library view. The path must resolve inside the
// workspace and the file must be small plain text.

async fn read_workspace_file(
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<WsFileContent>, (StatusCode, String)> {
    let rel = params.get("path").cloned().unwrap_or_default();
    let p = ws_path(&rel).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let root = ws_root();
    let canon_root = fs::canonicalize(&root).unwrap_or(root);
    let canon = fs::canonicalize(&p).map_err(|_| (StatusCode::NOT_FOUND, "file not found".to_string()))?;
    if !canon.starts_with(&canon_root) {
        return Err((StatusCode::FORBIDDEN, "outside workspace".into()));
    }
    if !canon.is_file() {
        return Err((StatusCode::BAD_REQUEST, "not a file".into()));
    }
    let meta = fs::metadata(&canon).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    if meta.len() > 20000 {
        return Err((StatusCode::BAD_REQUEST, "file too large to preview".into()));
    }
    let content = fs::read_to_string(&canon)
        .map_err(|_| (StatusCode::BAD_REQUEST, "not a text file".to_string()))?;
    Ok(Json(WsFileContent { path: rel, content }))
}

fn extract_stream_content(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let json_part = trimmed
        .strip_prefix("data:")
        .map(|s| s.trim())
        .unwrap_or(trimmed);
    if json_part == "[DONE]" {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(json_part).ok()?;
    // Ollama format: {"message":{"content":"..."}}
    if let Some(c) = v
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
    {
        if !c.is_empty() {
            return Some(c.to_string());
        }
    }
    // OpenAI format: {"choices":[{"delta":{"content":"..."}}]}
    if let Some(c) = v
        .get("choices")
        .and_then(|ch| ch.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
    {
        if !c.is_empty() {
            return Some(c.to_string());
        }
    }
    // OpenAI non-streaming message format
    if let Some(c) = v
        .get("choices")
        .and_then(|ch| ch.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
    {
        if !c.is_empty() {
            return Some(c.to_string());
        }
    }
    None
}

fn stream_is_done(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    let json_part = trimmed
        .strip_prefix("data:")
        .map(|s| s.trim())
        .unwrap_or(trimmed);
    if json_part == "[DONE]" {
        return true;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json_part) else {
        return false;
    };
    if v.get("done") == Some(&serde_json::Value::Bool(true)) {
        return true;
    }
    if v
        .get("choices")
        .and_then(|ch| ch.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
        .is_some()
    {
        return true;
    }
    false
}

// Describe one attached photo with Mirage (on-device vision) and return
// a short plain-text description for the text-only local model.
// `image` is a base64 data URL ("data:image/jpeg;base64,...") as sent by
// the client, or raw base64. The bytes are written to a temp file, the
// mirage_describe.py helper runs on it, and the file is removed after.
async fn describe_image_with_mirage(image: &str) -> Result<String, String> {
    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
    let b64part = match image.find(',') {
        Some(idx) if image[..idx].contains("base64") => &image[idx + 1..],
        _ => image,
    };
    let bytes = B64
        .decode(b64part.trim())
        .map_err(|e| format!("bad image data: {e}"))?;
    let fname = format!(
        "/tmp/mirage-{}-{}.jpg",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    std::fs::write(&fname, &bytes).map_err(|e| format!("cannot write temp image: {e}"))?;
    let run = tokio::process::Command::new(MIRAGE_PYTHON)
        .arg(MIRAGE_SCRIPT)
        .arg(&fname)
        .output();
    let out = tokio::time::timeout(std::time::Duration::from_secs(90), run)
        .await
        .map_err(|_| "mirage timed out".to_string())
        .and_then(|r| r.map_err(|e| format!("cannot run mirage: {e}")));
    let _ = std::fs::remove_file(&fname);
    let out = out?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(if err.is_empty() {
            "mirage failed".to_string()
        } else {
            err
        });
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        return Err("mirage returned nothing".to_string());
    }
    Ok(text)
}

async fn chat(Json(payload): Json<ChatRequest>) -> Result<impl IntoResponse, (StatusCode, String)> {
    // System prompt: selected skill instruction + stored memories.
    let skill: Option<Skill> = match &payload.skill_id {
        Some(id) if !id.is_empty() => {
            let skills: Vec<Skill> = load_json(SKILLS_FILE);
            skills.into_iter().find(|s| &s.id == id)
        }
        _ => None,
    };
    let memories: Vec<Memory> = load_json(MEMORIES_FILE);
    // Custom endpoint (OpenAI-compatible /chat/completions) if one is selected.
    // Determined here (before the system prompt) so the prompt can be kept
    // simple for the tiny local model, which cannot follow the [[WEBIMAGE:]]
    // marker instruction.
    let endpoint: Option<Endpoint> = match &payload.endpoint_id {
        Some(id) if !id.is_empty() => {
            let eps: Vec<Endpoint> = load_json(ENDPOINTS_FILE);
            eps.into_iter().find(|e| &e.id == id)
        }
        _ => None,
    };
    let system_prompt = build_system_prompt(skill.as_ref(), &memories, payload.persona.as_deref(), endpoint.is_none());
    if let Some(sp) = &system_prompt {
        eprintln!(
            "[chat] system prompt: skill={} memories={} chars={}",
            skill.as_ref().map(|s| s.name.as_str()).unwrap_or("-"),
            memories.len(),
            sp.len()
        );
    }

    let mut msgs: Vec<OllamaMessage> = Vec::new();
    if let Some(sp) = system_prompt {
        msgs.push(OllamaMessage {
            role: "system".to_string(),
            content: sp,
            images: None,
        });
    }
    for m in &payload.messages {
        msgs.push(OllamaMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            images: m.images.clone(),
        });
    }

    let client = reqwest::Client::new();
    // Local Ollama has no vision model that can run on this phone (the only
    // local vision model, gemma4-e2b, cannot load - out of memory). So an
    // attached photo is described on-device with Mirage (Cactus Compute,
    // 22MB VLM) and the text description is given to the local model
    // instead, which reads it as plain text. The image bytes are dropped
    // (the local model cannot use them).
    if endpoint.is_none() {
        let mut described = 0usize;
        let mut attempted = 0usize;
        for m in msgs.iter_mut() {
            if let Some(imgs) = m.images.take() {
                if imgs.is_empty() {
                    continue;
                }
                let mut descs: Vec<String> = Vec::new();
                for img in &imgs {
                    attempted += 1;
                    match describe_image_with_mirage(img).await {
                        Ok(d) => {
                            described += 1;
                            descs.push(d);
                        }
                        Err(e) => {
                            eprintln!("[mirage] describe failed: {e}");
                        }
                    }
                }
                if !descs.is_empty() {
                    let joined = descs.join("\n\n");
                    m.content = format!("{}\n\n[Attached photo: {}]", m.content, joined);
                }
            }
        }
        if attempted > 0 && described == 0 {
            return Err((
                StatusCode::BAD_GATEWAY,
                "Mirage could not describe the photo. Switch to a Gemini model (in the model picker) to describe a photo.".to_string(),
            ));
        }
    }
    let resp = match &endpoint {
        Some(ep) => {
            let url = if ep.base_url.ends_with("/chat/completions") {
                ep.base_url.clone()
            } else {
                format!("{}/chat/completions", ep.base_url)
            };
            // Prefer the model the picker selected (passed in); fall back to
            // the endpoint's stored model when the caller sent none.
            let use_model = if payload.model.trim().is_empty() {
                ep.model.clone()
            } else {
                payload.model.clone()
            };
            let body = serde_json::json!({
                "model": use_model,
                "messages": openai_messages(&msgs),
                "stream": true,
            });
            let mut req = client.post(url).bearer_auth(&ep.api_key);
            if is_copilot_base(&ep.base_url) {
                req = with_copilot_headers(req, false, messages_have_images(&msgs));
            }
            req.json(&body)
                .send()
                .await
                .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Endpoint unreachable: {e}")))?
        }
        None => {
            let model = if payload.model.trim().is_empty() {
                DEFAULT_MODEL.to_string()
            } else {
                payload.model.clone()
            };
            let req_body = OllamaRequest {
                model,
                messages: msgs,
                stream: true,
                think: false,
            };
            client
                .post(OLLAMA_URL)
                .json(&req_body)
                .send()
                .await
                .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Ollama unreachable: {e}")))?
        }
    };
    if !resp.status().is_success() {
        let label = if endpoint.is_some() { "Endpoint" } else { "Ollama" };
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("{} error: {}", label, resp.status()),
        ));
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(bytes) => {
                    buf.extend_from_slice(&bytes);
                    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                        let line: Vec<u8> = buf.drain(..=pos).collect();
                        let text = String::from_utf8_lossy(&line);
                        if let Some(content) = extract_stream_content(&text) {
                            if tx.send(Ok(Event::default().data(content))).await.is_err() {
                                return;
                            }
                        }
                        if stream_is_done(&text) {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });
    let sse_stream = ReceiverStream::new(rx);
    Ok(Sse::new(sse_stream))
}

// ---------- Agent mode ----------

static AGENT_RUNS: OnceLock<Mutex<HashMap<String, AbortHandle>>> = OnceLock::new();

fn agent_runs() -> &'static Mutex<HashMap<String, AbortHandle>> {
    AGENT_RUNS.get_or_init(|| Mutex::new(HashMap::new()))
}

// ---------- Agent workspace setting ----------
// Steven can set the agent workspace from Settings. The path is stored in
// settings.json so it survives restarts. Empty/absent = the default
// "workspace" folder next to the server binary.

#[derive(Serialize, Deserialize, Default, Clone)]
struct Settings {
    #[serde(default)]
    workspace: String,
    // Magic Hour API key for video generation. Stored on disk only; it is
    // never returned to the browser (SettingsResponse carries just a bool).
    #[serde(default)]
    magic_hour_api_key: String,
}

#[derive(Serialize)]
struct SettingsResponse {
    workspace: String,
    magic_hour_configured: bool,
}

#[derive(Deserialize)]
struct SettingsUpdate {
    #[serde(default)]
    workspace: String,
}

fn load_settings() -> Settings {
    fs::read_to_string(SETTINGS_FILE)
        .ok()
        .and_then(|d| serde_json::from_str(&d).ok())
        .unwrap_or_default()
}

fn save_settings(s: &Settings) -> Result<(), String> {
    let data = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    fs::write(SETTINGS_FILE, data).map_err(|e| e.to_string())
}

// Resolve a path to its canonical (symlink-resolved) form as far as possible:
// if the path itself does not exist yet, canonicalize the nearest existing
// ancestor and re-append the rest. This stops a symlink from smuggling the
// workspace somewhere else (e.g. a link pointing outside the allowed roots).
fn canonicalize_best_effort(path: &std::path::Path) -> PathBuf {
    if let Ok(c) = fs::canonicalize(path) {
        return c;
    }
    let mut cur = path.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        match fs::canonicalize(&cur) {
            Ok(c) => {
                let mut out = c;
                for comp in tail.iter().rev() {
                    out.push(comp);
                }
                return out;
            }
            Err(_) => {
                if let Some(file) = cur.file_name() {
                    tail.push(file.to_os_string());
                    if !cur.pop() {
                        return path.to_path_buf();
                    }
                } else {
                    return path.to_path_buf();
                }
            }
        }
    }
}

// Validate a user-supplied workspace path. The workspace is only the
// default folder the agent starts in, not a boundary: any absolute path to
// a real folder is accepted. Only sanity checks remain: it must be absolute,
// must not contain '..', and must not be the filesystem root itself.
fn validate_workspace(raw: &str) -> Result<PathBuf, String> {
    let p = raw.trim();
    if p.is_empty() {
        return Err("workspace path is empty".into());
    }
    if !p.starts_with('/') {
        return Err("workspace must be an absolute path, e.g. /root/my-work or /sdcard/odysseus_rust".into());
    }
    let path = std::path::Path::new(p);
    let comps: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    if comps.iter().any(|c| c == "..") {
        return Err("workspace path must not contain '..'".into());
    }
    if comps.len() < 2 {
        return Err("workspace must be a folder, e.g. /root/my-work or /sdcard/odysseus_rust".into());
    }
    Ok(canonicalize_best_effort(path))
}

fn configured_workspace() -> Option<PathBuf> {
    let s = load_settings();
    let raw = s.workspace.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    validate_workspace(&raw).ok()
}

fn ws_root() -> PathBuf {
    if let Some(p) = configured_workspace() {
        return p;
    }
    PathBuf::from(WORKSPACE_DIR)
}

// The absolute path shown in Settings and the Workspace panel: the
// configured path when set, otherwise the default workspace resolved
// against the server's working directory.
fn effective_workspace() -> String {
    if let Some(p) = configured_workspace() {
        return p.to_string_lossy().to_string();
    }
    match fs::canonicalize(WORKSPACE_DIR) {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(_) => {
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/root/odysseus-rust"));
            cwd.join(WORKSPACE_DIR).to_string_lossy().to_string()
        }
    }
}

async fn get_settings() -> Json<SettingsResponse> {
    let s = load_settings();
    Json(SettingsResponse {
        workspace: effective_workspace(),
        magic_hour_configured: !s.magic_hour_api_key.trim().is_empty(),
    })
}

#[derive(Deserialize)]
struct MagicHourKeyRequest {
    #[serde(default)]
    key: String,
}

#[derive(Serialize)]
struct MagicHourKeyResponse {
    configured: bool,
}

// Save (or clear, with an empty key) the Magic Hour API key used for video
// generation. The key itself is never returned to the browser.
async fn set_magic_hour_key(
    Json(payload): Json<MagicHourKeyRequest>,
) -> Result<Json<MagicHourKeyResponse>, (StatusCode, String)> {
    let mut s = load_settings();
    s.magic_hour_api_key = payload.key.trim().to_string();
    save_settings(&s).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(MagicHourKeyResponse {
        configured: !s.magic_hour_api_key.is_empty(),
    }))
}

async fn put_settings(
    Json(payload): Json<SettingsUpdate>,
) -> Result<Json<SettingsResponse>, (StatusCode, String)> {
    let raw = payload.workspace.trim().to_string();
    // Empty input resets to the default workspace. Keep the stored
    // Magic Hour key: saving the workspace must not wipe it.
    if raw.is_empty() {
        let mut s = load_settings();
        s.workspace = String::new();
        save_settings(&s).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let _ = fs::create_dir_all(WORKSPACE_DIR);
        return Ok(Json(SettingsResponse {
            workspace: effective_workspace(),
            magic_hour_configured: !s.magic_hour_api_key.trim().is_empty(),
        }));
    }
    let path = validate_workspace(&raw).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    fs::create_dir_all(&path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("could not create folder: {e}")))?;
    let mut s = load_settings();
    s.workspace = path.to_string_lossy().to_string();
    save_settings(&s).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(SettingsResponse {
        workspace: effective_workspace(),
        magic_hour_configured: !s.magic_hour_api_key.trim().is_empty(),
    }))
}

// Resolve a path for the agent tools. Absolute paths are used as-is;
// relative paths are resolved against the workspace (the default folder).
// There is no confinement check: the agent runs as root and may reach any
// path. `..` is resolved, not rejected.
fn ws_path(rel: &str) -> Result<PathBuf, String> {
    let rel = rel.trim();
    if rel.is_empty() || rel == "." {
        return Ok(ws_root());
    }
    let p = if rel.starts_with('/') {
        PathBuf::from(rel)
    } else if rel == "~" {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        PathBuf::from(home)
    } else if let Some(stripped) = rel.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        PathBuf::from(home).join(stripped)
    } else {
        ws_root().join(rel)
    };
    Ok(canonicalize_best_effort(&p))
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...\n[output truncated, {} chars total]", &s[..end], s.len())
}

struct ToolResult {
    output: String,
    exit_code: i32,
    diff: Option<String>,
}

/// Line diff of old vs new file content, Codex-style: lines prefixed
/// with "+ " (added), "- " (removed) or "  " (unchanged context, 2 lines
/// around each change). Unchanged stretches longer than 4 lines are
/// collapsed to "  ...". Capped at 120 lines. A new file shows all lines
/// as added. Used to show file changes in the agent terminal bubble.
fn line_diff(old: &str, new: &str) -> String {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let (n, m) = (a.len(), b.len());
    if n.saturating_mul(m) > 400_000 {
        let mut out = String::new();
        for l in &a { out.push_str("- "); out.push_str(l); out.push('\n'); }
        for l in &b { out.push_str("+ "); out.push_str(l); out.push('\n'); }
        return out;
    }
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] { dp[i + 1][j + 1] + 1 } else { dp[i + 1][j].max(dp[i][j + 1]) };
        }
    }
    let mut ops: Vec<(char, &str)> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if a[i] == b[j] { ops.push((' ', a[i])); i += 1; j += 1; }
        else if dp[i + 1][j] >= dp[i][j + 1] { ops.push(('-', a[i])); i += 1; }
        else { ops.push(('+', b[j])); j += 1; }
    }
    while i < n { ops.push(('-', a[i])); i += 1; }
    while j < m { ops.push(('+', b[j])); j += 1; }
    let mut out = String::new();
    let mut shown = 0usize;
    let total = ops.len();
    let mut k = 0usize;
    while k < total {
        if ops[k].0 == ' ' {
            let start = k;
            while k < total && ops[k].0 == ' ' { k += 1; }
            let run = k - start;
            if run > 4 && start != 0 && k != total {
                for t in start..start + 2 { out.push_str("  "); out.push_str(ops[t].1); out.push('\n'); shown += 1; }
                out.push_str("  ...\n");
                for t in k - 2..k { out.push_str("  "); out.push_str(ops[t].1); out.push('\n'); shown += 1; }
            } else {
                for t in start..k { out.push_str("  "); out.push_str(ops[t].1); out.push('\n'); shown += 1; }
            }
        } else {
            out.push(ops[k].0); out.push(' '); out.push_str(ops[k].1); out.push('\n'); shown += 1; k += 1;
        }
        if shown > 120 { out.push_str("  ... (diff truncated)\n"); break; }
    }
    out
}

async fn run_shell(cmd: &str) -> ToolResult {
    let cmd = cmd.trim();
    if cmd.is_empty() {
        return ToolResult { diff: None,  output: "Empty command".into(), exit_code: 1 };
    }
    let ws = ws_root();
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(&ws)
            .output(),
    )
    .await;
    match out {
        Ok(Ok(o)) => {
            let mut s = String::new();
            if !o.stdout.is_empty() {
                s.push_str(&String::from_utf8_lossy(&o.stdout));
            }
            if !o.stderr.is_empty() {
                if !s.is_empty() {
                    s.push('\n');
                }
                s.push_str(&String::from_utf8_lossy(&o.stderr));
            }
            if s.trim().is_empty() {
                s = format!("(command finished, exit {})", o.status.code().unwrap_or(-1));
            }
            ToolResult { diff: None,  output: truncate(&s, 8000), exit_code: o.status.code().unwrap_or(-1) }
        }
        Ok(Err(e)) => ToolResult { diff: None,  output: format!("Failed to run command: {e}"), exit_code: 1 },
        Err(_) => ToolResult { diff: None,  output: "Command timed out after 60 seconds".into(), exit_code: 124 },
    }
}

async fn execute_tool(tool: &str, args: &str) -> ToolResult {
    let (arg1, arg2) = normalise_tool_args(tool, args);
    match tool {
        "run_shell" => run_shell(&arg1).await,
        "read_file" => match ws_path(&arg1) {
            Ok(p) => match fs::read_to_string(&p) {
                Ok(c) => ToolResult { diff: None,  output: truncate(&c, 20000), exit_code: 0 },
                Err(e) => ToolResult { diff: None,  output: format!("Cannot read file: {e}"), exit_code: 1 },
            },
            Err(e) => ToolResult { diff: None,  output: format!("Path error: {e}"), exit_code: 1 },
        },
        "write_file" => {
            let path = arg1.trim();
            let content = arg2;
            if path.is_empty() {
                return ToolResult { diff: None, 
                    output: "write_file needs a file path on the first line".into(),
                    exit_code: 1,
                };
            }
            match ws_path(path) {
                Ok(p) => {
                    if let Some(parent) = p.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    let old = fs::read_to_string(&p).ok();
                    match fs::write(&p, &content) {
                        Ok(()) => ToolResult { diff: Some(line_diff(old.as_deref().unwrap_or(""), &content)), 
                            output: format!("Wrote {} ({} bytes)", path, content.len()),
                            exit_code: 0,
                        },
                        Err(e) => ToolResult { diff: None,  output: format!("Cannot write file: {e}"), exit_code: 1 },
                    }
                }
                Err(e) => ToolResult { diff: None,  output: format!("Path error: {e}"), exit_code: 1 },
            }
        }
        "list_files" => {
            let dir = if arg1.trim().is_empty() { "." } else { arg1.trim() };
            match ws_path(dir) {
                Ok(p) => match fs::read_dir(&p) {
                    Ok(entries) => {
                        let mut names: Vec<String> = Vec::new();
                        for e in entries.flatten() {
                            let name = e.file_name().to_string_lossy().to_string();
                            let is_dir = e.path().is_dir();
                            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                            names.push(if is_dir {
                                format!("{}/ (dir)", name)
                            } else {
                                format!("{} ({} bytes)", name, size)
                            });
                        }
                        names.sort();
                        ToolResult { diff: None, 
                            output: if names.is_empty() {
                                "(empty directory)".into()
                            } else {
                                names.join("\n")
                            },
                            exit_code: 0,
                        }
                    }
                    Err(e) => ToolResult { diff: None,  output: format!("Cannot list directory: {e}"), exit_code: 1 },
                },
                Err(e) => ToolResult { diff: None,  output: format!("Path error: {e}"), exit_code: 1 },
            }
        }
        "search_images" => {
            let q = arg1.trim();
            if q.is_empty() {
                return ToolResult { diff: None, 
                    output: "search_images needs a search query".into(),
                    exit_code: 1,
                };
            }
            match search_web_images(q).await {
                Ok(imgs) => {
                    let mut s = String::new();
                    for (i, im) in imgs.iter().enumerate() {
                        s.push_str(&format!(
                            "{}. {}\n   Image: {}\n   Thumbnail: {}\n   Source: {}\n",
                            i + 1,
                            im.title,
                            im.image_url,
                            im.thumbnail_url,
                            im.source_url
                        ));
                    }
                    ToolResult { diff: None,  output: truncate(&s, 4000), exit_code: 0 }
                }
                Err(e) => ToolResult { diff: None,  output: format!("Image search failed: {e}"), exit_code: 1 },
            }
        }
        "create_pdf" => {
            let filename = arg1.trim();
            let content = arg2.trim();
            if filename.is_empty() || content.is_empty() {
                return ToolResult { diff: None, 
                    output: "create_pdf needs a filename on the first line and the PDF text after it".into(),
                    exit_code: 1,
                };
            }
            // Photos referenced as Markdown images in the text are
            // embedded in the PDF (under the text), the rest is plain text.
            let (clean_text, md_images) = extract_md_images(content);
            let req = PdfCreateRequest {
                title: filename.to_string(),
                text: clean_text,
                images: md_images,
                segments: Vec::new(),
            };
            let blocks = pdf_request_blocks(&req).await;
            let (embedded, failed) = pdf_block_image_counts(&blocks);
            match save_pdf_blocks_to_library(PDFS_DIR, PDF_INDEX_FILE, filename, &blocks) {
                Ok(e) => {
                    let mut note = String::new();
                    if embedded > 0 {
                        note.push_str(&format!(" ({} image{} embedded", embedded, if embedded == 1 { "" } else { "s" }));
                        if failed > 0 {
                            note.push_str(&format!(", {} could not be loaded", failed));
                        }
                        note.push(')');
                    }
                    ToolResult { diff: None, 
                        output: format!(
                            "Created PDF {} ({} bytes{}). Download: [{}](/api/pdf/file/{})",
                            e.name, e.size, note, e.name, e.id
                        ),
                        exit_code: 0,
                    }
                }
                Err(m) => ToolResult { diff: None, 
                    output: format!("Could not create PDF: {m}"),
                    exit_code: 1,
                },
            }
        }
        _ => ToolResult { diff: None,  output: "Unknown tool".into(), exit_code: 1 },
    }
}

fn is_tool_name(s: &str) -> bool {
    matches!(s, "run_shell" | "read_file" | "write_file" | "list_files" | "search_images" | "create_pdf")
}

// Strip one surrounding pair of matching quotes ("..." or '...').
// Small models often quote their arguments; a quoted path is never intended.
fn strip_quotes(s: &str) -> (&str, bool) {
    let t = s.trim();
    if t.len() >= 2 {
        let b = t.as_bytes();
        if (b[0] == b'"' && b[t.len() - 1] == b'"')
            || (b[0] == b'\'' && b[t.len() - 1] == b'\'')
        {
            return (&t[1..t.len() - 1], true);
        }
    }
    (t, false)
}

// Normalise tool arguments: strip the quotes small models wrap around
// paths and whole commands, and split write_file into path + content.
fn normalise_tool_args(tool: &str, args: &str) -> (String, String) {
    match tool {
        "write_file" | "create_pdf" => {
            let mut lines = args.lines();
            let raw_path = lines.next().unwrap_or("");
            let (path, path_quoted) = strip_quotes(raw_path);
            let content = lines.collect::<Vec<_>>().join("\n");
            let content = if path_quoted {
                strip_quotes(&content).0.to_string()
            } else {
                content
            };
            (path.to_string(), content)
        }
        _ => {
            let (a, _) = strip_quotes(args);
            (a.to_string(), String::new())
        }
    }
}

// Parse a tool call from the model reply. Accepts a fenced block whose
// first line is a tool name (with or without a ```tool tag), or a
// "TOOL: <name>" line. Lenient on purpose — the local model is small.
// DSML parameter values for a tool call, mapped to the tool's plain
// argument form. Only the shapes seen in the wild are mapped; anything
// else falls back to the first parameter's value.
fn dsml_args(tool: &str, params: &[(String, String)]) -> String {
    if params.is_empty() {
        return String::new();
    }
    let get = |names: &[&str]| -> Option<String> {
        for n in names {
            for (k, v) in params {
                if k.eq_ignore_ascii_case(n) {
                    return Some(v.clone());
                }
            }
        }
        None
    };
    match tool {
        "write_file" | "create_pdf" => {
            let path = get(&["path", "file", "filename", "name"]);
            let content = get(&["content", "text", "code", "body"]);
            match (path, content) {
                (Some(p), Some(c)) => format!("{}\n{}", p, c),
                (Some(p), None) => p,
                (None, Some(c)) => c,
                (None, None) => params[0].1.clone(),
            }
        }
        "read_file" => get(&["path", "file", "filename", "name"])
            .unwrap_or_else(|| params[0].1.clone()),
        "list_files" => get(&["path", "directory", "dir"]).unwrap_or_default(),
        "run_shell" => get(&["command", "cmd", "command_line"])
            .unwrap_or_else(|| params[0].1.clone()),
        "search_images" => get(&["query", "q", "text", "search"])
            .unwrap_or_else(|| params[0].1.clone()),
        _ => params[0].1.clone(),
    }
}

// Remove any bare <tool>...</tool> block from text shown to the user.
fn strip_tool_tag(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let mut tstart_opt = None;
        let mut from = 0;
        while let Some(tpos) = rest[from..].find("<tool") {
            let ts = from + tpos;
            if !rest[ts + "<tool".len()..].starts_with('_') {
                tstart_opt = Some(ts);
                break;
            }
            from = ts + "<tool".len();
        }
        let Some(tstart) = tstart_opt else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..tstart]);
        let after = &rest[tstart..];
        match after.find("</tool>") {
            Some(end) => {
                rest = &after[end + "</tool>".len()..];
            }
            None => {
                rest = "";
            }
        }
    }
    out
}

// Remove any DSML tool-call block from text shown to the user, so a
// model that emits DSML never leaves raw markup in the chat.
fn strip_dsml(text: &str) -> String {
    let norm: String = text.chars().filter(|c| *c != '\u{200b}').collect();
    let norm = norm.replace('\u{ff5c}', "|");
    let mut out = String::with_capacity(norm.len());
    let mut rest = norm.as_str();
    loop {
        let Some(start) = rest.find("<|DSML|") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        match after.find("</|DSML| calls>") {
            Some(end) => {
                rest = &after[end + "</|DSML| calls>".len()..];
            }
            None => {
                rest = "";
            }
        }
    }
    out
}

fn parse_tool_call(reply: &str) -> Option<(String, String)> {
    let mut from = 0;
    while from < reply.len() {
        let Some(fence) = reply[from..].find("```") else {
            break;
        };
        let abs = from + fence;
        let rest = &reply[abs + 3..];
        // Skip the tag line (e.g. "tool" right after the opening fence).
        let body_start = match rest.find('\n') {
            Some(nl) => nl + 1,
            None => rest.len(),
        };
        let body_rest = &rest[body_start..];
        let (body, next_from) = match body_rest.find("```") {
            Some(end) => (
                body_rest[..end].trim().to_string(),
                abs + 3 + body_start + end + 3,
            ),
            None => (body_rest.trim().to_string(), reply.len()),
        };
        if !body.is_empty() {
            let mut lines = body.lines();
            let first = lines.next().unwrap_or("").trim().to_lowercase();
            if is_tool_name(&first) {
                let args = lines.collect::<Vec<_>>().join("\n");
                return Some((first, args.trim().to_string()));
            }
        }
        from = next_from;
    }
    // XML form some models emit instead of a fenced block (Agnes):
    //   <tool_call>
    //   <function=tool>
    //   run_shell
    //   <arguments>
    //   </function>
    //   </tool_call>
    // The tag may also carry the name (<function=run_shell>); otherwise the
    // first content line is the tool name and the rest is the arguments.
    if let Some(tc) = reply.find("<tool_call>") {
        let after_tc = &reply[tc + "<tool_call>".len()..];
        if let Some(fn_start) = after_tc.find("<function") {
            let tag = &after_tc[fn_start..];
            if let Some(tag_end) = tag.find('>') {
                let tag_inner = tag["<function".len()..tag_end].trim();
                let tag_name = tag_inner
                    .strip_prefix('=')
                    .map(|s| {
                        let s = s.trim().trim_matches('"').to_lowercase();
                        // Agnes sometimes writes <function=Tool: write_file> -
                        // drop the "tool:" prefix so the name matches.
                        s.strip_prefix("tool:")
                            .map(|x| x.trim().to_string())
                            .unwrap_or(s)
                    })
                    .filter(|n| is_tool_name(n));
                let content = &tag[tag_end + 1..];
                let content = match content.find("</function>") {
                    Some(end) => &content[..end],
                    None => content,
                };
                let content = content.trim();
                if !content.is_empty() {
                    let mut lines = content.lines();
                    let first_line = lines.next().unwrap_or("").trim().to_lowercase();
                    if let Some(name) = tag_name {
                        // Name came from the tag; the content is all
                        // arguments, unless its first line repeats the name.
                        let args = if first_line == name {
                            lines.collect::<Vec<_>>().join("\n")
                        } else {
                            content.to_string()
                        };
                        return Some((name, args.trim().to_string()));
                    }
                    if is_tool_name(&first_line) {
                        let args = lines.collect::<Vec<_>>().join("\n");
                        return Some((first_line, args.trim().to_string()));
                    }
                }
            }
        }
    }
    // DSML form some models emit (free-pi DeepSeek):
    //   <|DSML| calls>
    //   <|DSML| invoke name="run_shell">
    //   <|DSML| parameter name="command" string="true">...</|DSML| parameter>
    //   </|DSML| invoke>
    //   </|DSML| calls>
    // The delimiters may carry a zero-width space and/or a fullwidth bar,
    // so normalize them before matching.
    {
        let norm: String = reply.chars().filter(|c| *c != '\u{200b}').collect();
        let norm = norm.replace('\u{ff5c}', "|");
        if let Some(inv) = norm.find("<|DSML| invoke") {
            let after = &norm[inv..];
            let name = after.find("name=\"").and_then(|i| {
                let rest = &after[i + 6..];
                rest.find('"').map(|e| rest[..e].to_lowercase())
            });
            if let Some(tool) = name.filter(|n| is_tool_name(n)) {
                let end = after.find("</|DSML| invoke>").unwrap_or(after.len());
                let body = &after[..end];
                let mut params: Vec<(String, String)> = Vec::new();
                let mut pfrom = 0;
                while let Some(ps) = body[pfrom..].find("<|DSML| parameter") {
                    let pabs = pfrom + ps;
                    let prest = &body[pabs..];
                    let pname = prest.find("name=\"").and_then(|i| {
                        let r = &prest[i + 6..];
                        r.find('"').map(|e| r[..e].to_string())
                    });
                    let vstart = prest.find('>').map(|g| g + 1);
                    let vend = prest.find("</|DSML| parameter>");
                    match (pname, vstart, vend) {
                        (Some(pn), Some(vs), Some(ve)) if vs <= ve => {
                            params.push((pn, prest[vs..ve].trim().to_string()));
                            pfrom = pabs + ve + "</|DSML| parameter>".len();
                        }
                        _ => break,
                    }
                }
                let args = dsml_args(&tool, &params);
                return Some((tool, args));
            }
        }
    }
    // Bare <tool>...</tool> form some models emit (free-pi particle):
    //   <tool
    //   run_shell
    //   <args>
    //   </tool>
    // The opening tag carries no ">" — the tool name is the next line.
    // Skip <tool_call> (handled by the XML branch above).
    {
        let mut tstart_opt = None;
        let mut from = 0;
        while let Some(tpos) = reply[from..].find("<tool") {
            let ts = from + tpos;
            if !reply[ts + "<tool".len()..].starts_with('_') {
                tstart_opt = Some(ts);
                break;
            }
            from = ts + "<tool".len();
        }
        if let Some(tstart) = tstart_opt {
            let after_tag = &reply[tstart + "<tool".len()..];
            let content_start = match after_tag.find('>') {
                Some(gt) if gt < after_tag.find('\n').unwrap_or(usize::MAX) => gt + 1,
                _ => after_tag.find('\n').map(|nl| nl + 1).unwrap_or(after_tag.len()),
            };
            let content = &after_tag[content_start..];
            let content = match content.find("</tool>") {
                Some(end) => &content[..end],
                None => content,
            };
            let content = content.trim();
            if !content.is_empty() {
                let mut lines = content.lines();
                let first = lines.next().unwrap_or("").trim().to_lowercase();
                if is_tool_name(&first) {
                    let args = lines.collect::<Vec<_>>().join("\n");
                    return Some((first, args.trim().to_string()));
                }
            }
        }
    }
    for line in reply.lines() {
        let t = line.trim();
        if t.to_lowercase().starts_with("tool:") {
            let tool = t[5..].trim().to_lowercase();
            if is_tool_name(&tool) {
                let idx = reply.find(line).unwrap_or(0);
                let rest = reply[idx + line.len()..].trim().to_string();
                let args = rest
                    .strip_prefix("ARGS:")
                    .or_else(|| rest.strip_prefix("args:"))
                    .map(|s| s.trim().to_string())
                    .unwrap_or(rest);
                return Some((tool, args));
            }
        }
    }
    None
}

// Does the user's request ask the agent to create or run something?
fn looks_like_create_request(text: &str) -> bool {
    let t = text.to_lowercase();
    ["create", "build", "make", "write", "run", "develop", "generate", "implement", "program", "system", "app", "script"]
        .iter()
        .any(|w| t.contains(w))
}

// Did the model paste code in its reply instead of using a tool?
fn reply_pasted_code(reply: &str) -> bool {
    if reply.contains("```") {
        return true;
    }
    let t = reply.to_lowercase();
    (t.contains("def ") && t.contains("import "))
        || (t.contains("class ") && t.contains("print("))
}

// Does the user's request ask the agent to CREATE and RUN something?
// Both halves are needed: a create word AND a run word. "Create a file
// called notes.txt" is create-only (no auto-run); "Run the tests" is
// run-only (nothing was created to run).
fn is_create_and_run_request(text: &str) -> bool {
    let t = text.to_lowercase();
    let create = ["create", "build", "make", "write", "develop", "generate", "implement"]
        .iter()
        .any(|w| t.contains(w));
    let run = t.contains("run") || t.contains("execute");
    create && run
}

// Quote a file path for safe use inside a shell command (single quotes,
// with embedded single quotes escaped). The path comes from the model,
// so never interpolate it into a command unquoted.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

// The server-side auto-run commands for a runnable file the agent wrote.
// Returns (compile command, run command). Python only: first verify it
// compiles, then run it with piped input so an interactive menu program
// exits instead of hanging (the worked example's menu uses 5 for Exit).
// The run is capped by `timeout 10` on top of run_shell's own 60s cap.
fn auto_run_commands(path: &str) -> Option<(String, String)> {
    if !path.to_lowercase().ends_with(".py") {
        return None;
    }
    let q = shell_quote(path);
    let compile = format!("python3 -m py_compile {}", q);
    let run = format!("printf \"5\\n\" | timeout 10 python3 {}", q);
    Some((compile, run))
}

// Is this model-call error transient (worth retrying)? The Gemini free
// tier returns 503/429 when it is overloaded or throttled; a single such
// failure must not kill the whole agent run. 4xx errors like 400/401/404
// (bad request, bad key, unknown model) are permanent and are not retried.
fn is_transient_model_error(e: &str) -> bool {
    let t = e.to_lowercase();
    for needle in [
        "503", "429", "500", "502", "504",
        "service unavailable", "too many requests",
        "internal server error", "bad gateway", "gateway timeout",
        "timeout", "timed out", "connection reset", "connection refused",
        "connection aborted", "broken pipe", "temporarily unavailable",
        "error sending request", "error decoding response", "dns", "network",
    ] {
        if t.contains(needle) {
            return true;
        }
    }
    false
}

// One non-streaming model call with retries on transient errors.
// Tries up to 3 times total (5s, then 10s backoff) so a transient 503/429
// from a free-tier endpoint does not kill the agent run.

// Build an OpenAI-compatible messages array. When a message carries images
// (base64 JPEG, already downscaled to max 1024px by the browser), each image
// is sent as an image_url content part so a vision-capable model (Gemini)
// can describe it. Messages without images keep a plain string content.
fn openai_messages(messages: &[OllamaMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| {
            if let Some(imgs) = &m.images {
                if !imgs.is_empty() {
                    let mut parts: Vec<serde_json::Value> = vec![serde_json::json!({
                        "type": "text",
                        "text": m.content
                    })];
                    for b64 in imgs {
                        let url = if b64.starts_with("data:") {
                            b64.clone()
                        } else {
                            format!("data:image/jpeg;base64,{}", b64)
                        };
                        parts.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": { "url": url }
                        }));
                    }
                    return serde_json::json!({ "role": m.role, "content": parts });
                }
            }
            serde_json::json!({ "role": m.role, "content": m.content })
        })
        .collect()
}

async fn call_model_with_retry(
    endpoint: &Option<Endpoint>,
    model: &str,
    messages: &[OllamaMessage],
    timeout_secs: u64,
) -> Result<String, String> {
    let mut last_err: Option<String> = None;
    for attempt in 0..3 {
        match call_model_once(endpoint, model, messages, timeout_secs).await {
            Ok(r) => return Ok(r),
            Err(e) => {
                if !is_transient_model_error(&e) || attempt == 2 {
                    return Err(e);
                }
                last_err = Some(e);
                let wait = if attempt == 0 { 5 } else { 10 };
                tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| "model call failed".to_string()))
}

// One non-streaming model call (Ollama or a custom OpenAI-compatible endpoint).
async fn call_model_once(
    endpoint: &Option<Endpoint>,
    model: &str,
    messages: &[OllamaMessage],
    timeout_secs: u64,
) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| e.to_string())?;
    match endpoint {
        Some(ep) => {
            let url = if ep.base_url.ends_with("/chat/completions") {
                ep.base_url.clone()
            } else {
                format!("{}/chat/completions", ep.base_url)
            };
            // Prefer the model the picker selected (passed in); fall back to
            // the endpoint's stored model when the caller sent none.
            let use_model = if model.trim().is_empty() {
                ep.model.clone()
            } else {
                model.to_string()
            };
            let body = serde_json::json!({
                "model": use_model,
                "messages": openai_messages(messages),
                "stream": false,
            });
            let mut req = client.post(url).bearer_auth(&ep.api_key);
            if is_copilot_base(&ep.base_url) {
                req = with_copilot_headers(req, true, messages_have_images(messages));
            }
            let resp = req
                .json(&body)
                .send()
                .await
                .map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                return Err(format!("Endpoint error: {}", resp.status()));
            }
            let v: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            v.get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| "empty reply from endpoint".to_string())
        }
        None => {
            let m = if model.trim().is_empty() { DEFAULT_MODEL } else { model };
            let req_body = OllamaRequest {
                model: m.to_string(),
                messages: messages.to_vec(),
                stream: false,
                think: false,
            };
            let resp = client
                .post(OLLAMA_URL)
                .json(&req_body)
                .send()
                .await
                .map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                return Err(format!("Ollama error: {}", resp.status()));
            }
            let v: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            v.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| "empty reply from Ollama".to_string())
        }
    }
}

#[derive(Deserialize)]
struct AgentRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    endpoint_id: Option<String>,
    #[serde(default)]
    skill_id: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    persona: Option<String>,
}

#[derive(Deserialize)]
struct StopRequest {
    run_id: String,
}

async fn agent(Json(payload): Json<AgentRequest>) -> Result<impl IntoResponse, (StatusCode, String)> {
    let run_id = payload.run_id.clone().unwrap_or_else(gen_id);
    let skill: Option<Skill> = match &payload.skill_id {
        Some(id) if !id.is_empty() => {
            let skills: Vec<Skill> = load_json(SKILLS_FILE);
            skills.into_iter().find(|s| &s.id == id)
        }
        _ => None,
    };
    let memories: Vec<Memory> = load_json(MEMORIES_FILE);
    let mut sys_parts = vec![AGENT_SYSTEM.to_string()];
    if let Some(s) = &skill {
        sys_parts.push(s.instruction.clone());
    }
    if let Some(p) = payload.persona.as_deref().and_then(persona_prompt) {
        sys_parts.push(p.to_string());
    }
    if !memories.is_empty() {
        let mut m = String::from("Remember these facts about the user:");
        for mem in &memories {
            m.push_str(&format!("\n- {}", mem.text));
        }
        sys_parts.push(m);
    }
    let system_prompt = sys_parts.join("\n\n");
    let mut loop_msgs: Vec<OllamaMessage> = vec![OllamaMessage {
        role: "system".to_string(),
        content: system_prompt,
        images: None,
    }];
    for m in &payload.messages {
        loop_msgs.push(OllamaMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            images: None,
        });
    }
    let endpoint: Option<Endpoint> = match &payload.endpoint_id {
        Some(id) if !id.is_empty() => {
            let eps: Vec<Endpoint> = load_json(ENDPOINTS_FILE);
            eps.into_iter().find(|e| &e.id == id)
        }
        _ => None,
    };
    let model = payload.model.clone();
    // Does this request ask the agent to create/run something? Used by the
    // paste-code safety net below.
    let wants_create = payload
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| looks_like_create_request(&m.content))
        .unwrap_or(false);
    let wants_run = payload
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| {
            let t = m.content.to_lowercase();
            t.contains("run") || t.contains("execute")
        })
        .unwrap_or(false);
    let wants_create_and_run = payload
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| is_create_and_run_request(&m.content))
        .unwrap_or(false);
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
    let run_id_inner = run_id.clone();
    let handle = tokio::spawn(async move {
        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::json!({"type":"run","run_id":run_id_inner}).to_string(),
            )))
            .await;
        let mut steps = 0;
        let mut nudges = 0;
        let mut did_write = false;
        let mut did_run = false;
        let mut wrote_empty = false;
        let mut last_py: Option<String> = None;
        let mut auto_ran = false;
        loop {
            if steps >= 10 {
                let _ = tx
                    .send(Ok(Event::default().data(
                        serde_json::json!({"type":"text","content":"\n\n[I stopped after 10 tool steps. Ask me to continue if needed.]"}).to_string(),
                    )))
                    .await;
                break;
            }
            steps += 1;
            let reply = match call_model_with_retry(&endpoint, &model, &loop_msgs, 120).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx
                        .send(Ok(Event::default().data(
                            serde_json::json!({"type":"error","message":format!("Model error: {e}")}).to_string(),
                        )))
                        .await;
                    break;
                }
            };
            if reply.trim().is_empty() {
                let _ = tx
                    .send(Ok(Event::default().data(
                        serde_json::json!({"type":"error","message":"Model returned an empty reply"}).to_string(),
                    )))
                    .await;
                break;
            }
            match parse_tool_call(&reply) {
                Some((tool, args)) => {
                    let _ = tx
                        .send(Ok(Event::default().data(
                            serde_json::json!({"type":"tool","tool":tool,"args":args}).to_string(),
                        )))
                        .await;
                    let result = execute_tool(&tool, &args).await;
                    if tool == "write_file" {
                        did_write = true;
                        // An empty write (0 bytes) is a failure, not a file.
                        wrote_empty = result.output.contains("(0 bytes)");
                        // Remember the last .py file written successfully so
                        // the server-side auto-run knows what to run.
                        if result.exit_code == 0 {
                            let (wpath, _) = normalise_tool_args(&tool, &args);
                            let wp = wpath.trim().to_string();
                            if wp.to_lowercase().ends_with(".py") {
                                last_py = Some(wp);
                            }
                        }
                    }
                    if tool == "run_shell" {
                        // py_compile only checks the file — it is not running
                        // the program, so it must not count as a run.
                        let a = args.to_lowercase();
                        if a.contains("python") && !a.contains("py_compile") {
                            did_run = true;
                        }
                    }
                    let _ = tx
                        .send(Ok(Event::default().data(
                            serde_json::json!({"type":"result","tool":tool,"output":result.output,"exit_code":result.exit_code,"diff":result.diff}).to_string(),
                        )))
                        .await;
                    let ctx_out = format!(
                        "Tool {} output (exit {}):\n{}",
                        tool,
                        result.exit_code,
                        truncate(&result.output, 4000)
                    );
                    loop_msgs.push(OllamaMessage {
                        role: "assistant".to_string(),
                        content: reply,
                        images: None,
                    });
                    loop_msgs.push(OllamaMessage {
                        role: "user".to_string(),
                        content: ctx_out,
                        images: None,
                    });
                }
                None => {
                    // Safety net 0: the file the model wrote is empty
                    // (0 bytes). Nudge it to write the real code.
                    if wrote_empty && nudges < 2 {
                        nudges += 1;
                        loop_msgs.push(OllamaMessage {
                            role: "assistant".to_string(),
                            content: reply,
                            images: None,
                        });
                        loop_msgs.push(OllamaMessage {
                            role: "user".to_string(),
                            content: "The file you wrote is empty (0 bytes). Use write_file again and include the complete program code this time — the file path on the first line, then the full code below it. Do not send an empty write_file.".to_string(),
                            images: None,
                        });
                        continue;
                    }
                    // Safety net 1: the model pasted code instead of using a
                    // tool for a create/run task. Nudge it (max 2) instead
                    // of ending the run with no file created.
                    if wants_create
                        && reply_pasted_code(&reply)
                        && nudges < 2
                        && (!did_write || !did_run)
                    {
                        nudges += 1;
                        loop_msgs.push(OllamaMessage {
                            role: "assistant".to_string(),
                            content: reply,
                            images: None,
                        });
                        loop_msgs.push(OllamaMessage {
                            role: "user".to_string(),
                            content: "You pasted code in your reply, but no file was created. Use write_file to create the file now, then run_shell to run it. Do not paste code in your reply.".to_string(),
                            images: None,
                        });
                        continue;
                    }
                    // Safety net 2: the user asked to create AND run, the
                    // file was written, but the model is finishing without
                    // ever running it (and may claim it ran). Nudge it to
                    // actually run the file before summarising.
                    if wants_run && did_write && !did_run && nudges < 2 {
                        nudges += 1;
                        loop_msgs.push(OllamaMessage {
                            role: "assistant".to_string(),
                            content: reply,
                            images: None,
                        });
                        loop_msgs.push(OllamaMessage {
                            role: "user".to_string(),
                            content: "You wrote the file but you have not run it yet. Use run_shell to run it now — pipe input so it exits (for example: printf \"5\\n\" | python hospital.py). Then report only the output it actually printed. Do not say it ran if you did not run it.".to_string(),
                            images: None,
                        });
                        continue;
                    }
                    // Safety net 3 (server-side auto-run): the user asked to
                    // create AND run, a runnable file was written, the model
                    // still has not run it, and the nudges are exhausted
                    // (none of the nets above fired). Run it ourselves and
                    // feed the REAL output back so the final answer reports
                    // what actually happened instead of an invented run.
                    if wants_create_and_run && did_write && !did_run && !auto_ran {
                        if let Some(path) = last_py.clone() {
                            if let Some((compile_cmd, run_cmd)) = auto_run_commands(&path) {
                                auto_ran = true;
                                // Compile step.
                                let _ = tx
                                    .send(Ok(Event::default().data(
                                        serde_json::json!({"type":"tool","tool":"run_shell","args":format!("{} (auto-run)", compile_cmd)}).to_string(),
                                    )))
                                    .await;
                                let compile_res = run_shell(&compile_cmd).await;
                                let _ = tx
                                    .send(Ok(Event::default().data(
                                        serde_json::json!({"type":"result","tool":"run_shell","output":compile_res.output.clone(),"exit_code":compile_res.exit_code}).to_string(),
                                    )))
                                    .await;
                                let auto_out = if compile_res.exit_code != 0 {
                                    format!(
                                        "Server auto-run: the program did NOT compile.\nCompile output (exit {}):\n{}",
                                        compile_res.exit_code, compile_res.output
                                    )
                                } else {
                                    // Run step.
                                    let _ = tx
                                        .send(Ok(Event::default().data(
                                            serde_json::json!({"type":"tool","tool":"run_shell","args":format!("{} (auto-run)", run_cmd)}).to_string(),
                                        )))
                                        .await;
                                    let run_res = run_shell(&run_cmd).await;
                                    let _ = tx
                                        .send(Ok(Event::default().data(
                                            serde_json::json!({"type":"result","tool":"run_shell","output":run_res.output.clone(),"exit_code":run_res.exit_code}).to_string(),
                                        )))
                                        .await;
                                    did_run = true;
                                    format!(
                                        "Server auto-run: I ran the program myself because you did not run it.\nCommand: {}\nReal output (exit {}):\n{}",
                                        run_cmd, run_res.exit_code, run_res.output
                                    )
                                };
                                loop_msgs.push(OllamaMessage {
                                    role: "assistant".to_string(),
                                    content: reply.clone(),
                                    images: None,
                                });
                                loop_msgs.push(OllamaMessage {
                                    role: "user".to_string(),
                                    content: format!(
                                        "{}\n\nNow reply with your final answer as plain text, reporting only the real output above. Do not invent output.",
                                        auto_out
                                    ),
                                    images: None,
                                });
                                continue;
                            }
                        }
                    }
                    let _ = tx
                        .send(Ok(Event::default().data(
                            serde_json::json!({"type":"text","content":strip_dsml(&strip_tool_tag(&reply))}).to_string(),
                        )))
                        .await;
                    break;
                }
            }
        }
        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::json!({"type":"done","steps":steps}).to_string(),
            )))
            .await;
        if let Ok(mut m) = agent_runs().lock() {
            m.remove(&run_id_inner);
        }
    });
    if let Ok(mut m) = agent_runs().lock() {
        m.insert(run_id, handle.abort_handle());
    }
    let sse_stream = ReceiverStream::new(rx);
    Ok(Sse::new(sse_stream))
}

async fn stop_agent(Json(payload): Json<StopRequest>) -> Result<StatusCode, (StatusCode, String)> {
    let mut map = agent_runs()
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "lock error".to_string()))?;
    match map.remove(&payload.run_id) {
        Some(h) => {
            h.abort();
            Ok(StatusCode::NO_CONTENT)
        }
        None => Err((StatusCode::NOT_FOUND, "run not found".into())),
    }
}

#[derive(Serialize)]
struct WsEntry {
    path: String,
    is_dir: bool,
    size: u64,
}

async fn list_workspace() -> Json<Vec<WsEntry>> {
    let mut out: Vec<WsEntry> = Vec::new();
    fn walk(dir: &std::path::Path, base: &std::path::Path, depth: usize, out: &mut Vec<WsEntry>) {
        if depth > 3 || out.len() >= 200 {
            return;
        }
        if let Ok(entries) = fs::read_dir(dir) {
            let mut items: Vec<_> = entries.flatten().collect();
            items.sort_by_key(|e| e.file_name());
            for e in items {
                let p = e.path();
                let rel = p
                    .strip_prefix(base)
                    .unwrap_or(&p)
                    .to_string_lossy()
                    .to_string();
                let is_dir = p.is_dir();
                let size = if is_dir { 0 } else { e.metadata().map(|m| m.len()).unwrap_or(0) };
                out.push(WsEntry { path: rel, is_dir, size });
                if is_dir {
                    walk(&p, base, depth + 1, out);
                }
            }
        }
    }
    walk(&ws_root(), &ws_root(), 0, &mut out);
    Json(out)
}

#[tokio::main]
async fn main() {
    let _ = fs::create_dir_all(CHATS_DIR);
    let _ = fs::create_dir_all(ws_root());
    let readme = ws_root().join("README.txt");
    if !readme.exists() {
        let _ = fs::write(
            &readme,
            "Mawaxi AI agent workspace.\nFiles the agent creates live here.\n",
        );
    }
    // Seed one example skill so the Skills section is not empty.
    if !std::path::Path::new(SKILLS_FILE).exists() {
        let seed = vec![Skill {
            id: gen_id(),
            name: "Summarise".to_string(),
            instruction: "Summarise the user's text clearly and briefly, keeping only the key points."
                .to_string(),
        }];
        let _ = save_json(SKILLS_FILE, &seed);
    }
    let app = Router::new()
        .route("/", get(index))
        .route("/api/chat", post(chat))
        .route("/api/agent", post(agent))
        .route("/api/agent/stop", post(stop_agent))
        .route("/api/workspace", get(list_workspace))
        .route("/api/settings", get(get_settings).put(put_settings))
        .route("/api/settings/magic-hour-key", put(set_magic_hour_key))
        .route("/api/image", post(generate_image))
        .route("/api/web-image", post(web_image))
        .route("/api/video", post(generate_video))
        .route("/api/video/file/:name", get(serve_video_file))
        .route("/api/models", get(list_models))
        .route("/api/chats", get(list_chats))
        .route("/api/chats/search", get(search_chats))
        .route(
            "/api/chats/:id",
            get(get_chat).put(put_chat).delete(delete_chat),
        )
        .route("/api/endpoint-presets", get(list_endpoint_presets))
        .route("/api/endpoint-models", post(fetch_new_endpoint_models))
        .route("/api/endpoints", get(list_endpoints).post(create_endpoint))
        .route("/api/endpoints/:id", put(update_endpoint).delete(delete_endpoint))
        .route("/api/endpoints/:id/models", get(fetch_saved_endpoint_models))
        .route("/api/copilot/device/start", post(copilot_device_start))
        .route("/api/copilot/device/poll", post(copilot_device_poll))
        .route("/api/memories", get(list_memories).post(create_memory))
        .route("/api/memories/:id", delete(delete_memory))
        .route("/api/skills", get(list_skills).post(create_skill))
        .route("/api/skills/:id", delete(delete_skill))
        .route("/api/notes", get(list_notes).post(create_note))
        .route("/api/notes/:id", put(update_note).delete(delete_note))
        .route(
            "/api/calendar",
            get(list_calendar_events).post(create_calendar_event),
        )
        .route("/api/calendar/:id", delete(delete_calendar_event))
        .route("/api/browser/fetch", get(browser_fetch))
        .route("/api/pdf/create", post(create_pdf))
        .route("/api/pdf/save", post(save_pdf))
        .route("/api/pdf/upload", post(upload_pdf))
        .route("/api/pdf/list", get(list_pdfs))
        .route("/api/pdf/file/:id", get(serve_pdf_file))
        .route("/api/pdf/text/:id", get(pdf_text))
        .route("/api/pdf/edit", post(edit_pdf))
        .route("/api/pdf/:id", delete(delete_pdf))
        .route("/api/cookbook/search", get(cookbook_search))
        .route("/api/cookbook/cached", get(cookbook_cached))
        .route("/api/cookbook/running", get(cookbook_running))
        .route("/api/cookbook/download", post(cookbook_download))
        .route("/api/cookbook/import", post(cookbook_import))
        .route("/api/cookbook/downloads", get(cookbook_downloads))
        .route("/api/cookbook/stop", post(cookbook_stop))
        .route("/api/research", post(research))
        .route("/api/research/cancel", post(cancel_research))
        .route("/api/research/history", get(list_research_history))
        .route(
            "/api/research/history/delete",
            post(delete_research_history_bulk),
        )
        .route(
            "/api/research/history/:id",
            get(get_research_history_entry).delete(delete_research_history_entry),
        )
        .route(
            "/api/research/history/:id/download",
            get(download_research_history_entry),
        )
        .route("/api/workspace/file", get(read_workspace_file))
        // PDF upload sends the file as base64 inside JSON, so the
        // body limit must cover a 20 MB PDF (MAX_PDF_BYTES) plus
        // base64 overhead (~33%). Axum defaults JSON bodies to
        // 2 MB, which rejected any PDF over ~1.5 MB with 413.
        .layer(DefaultBodyLimit::max(30 * 1024 * 1024));
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("mawaxi-ai listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_gguf_files_prefers_q4_k_m() {
        let m = serde_json::json!({"siblings": [
            {"rfilename": "model-Q8_0.gguf", "size": 800},
            {"rfilename": "model-Q4_K_M.gguf", "size": 500},
            {"rfilename": "model-Q2_K.gguf", "size": 300},
            {"rfilename": "config.json", "size": 10},
            {"rfilename": "model.safetensors", "size": 900}
        ]});
        let files = hf_gguf_files(&m);
        assert_eq!(files.len(), 3);
        assert_eq!(files[0].filename, "model-Q4_K_M.gguf");
        assert_eq!(files[1].filename, "model-Q2_K.gguf");
        assert_eq!(files[2].filename, "model-Q8_0.gguf");
    }

    #[test]
    fn chat_message_keeps_pdf_text() {
        let m = ChatMessage {
            role: "user".into(),
            content: "read this".into(),
            image: None,
            images: None,
            video: None,
            source: None,
            pdf_text: Some("hello pdf".into()),
            pdf_name: Some("korr.pdf".into()),
        };
        let j = serde_json::to_value(&m).unwrap();
        assert_eq!(j["pdfText"], "hello pdf");
        assert_eq!(j["pdfName"], "korr.pdf");
        let back: ChatMessage = serde_json::from_value(j).unwrap();
        assert_eq!(back.pdf_text.as_deref(), Some("hello pdf"));
        assert_eq!(back.pdf_name.as_deref(), Some("korr.pdf"));
    }

    #[test]
    fn base64_round_trips() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"hello video"), "aGVsbG8gdmlkZW8=");
        assert_eq!(base64_decode("Zm9v").unwrap(), b"foo");
        assert_eq!(base64_decode("aGVsbG8gdmlkZW8=").unwrap(), b"hello video");
        assert_eq!(
            base64_decode(&base64_encode(b"\x00\xff\x10\x80")).unwrap(),
            b"\x00\xff\x10\x80"
        );
    }

    #[test]
    fn hf_video_ref_and_data_uri_decode() {
        // The Space's SSE "complete" event data, as seen live.
        let data = r#"[{"video": "data:video/mp4;base64,AAAAIGZ0eXBpc29t"}]"#;
        let vid = hf_video_ref(data).unwrap();
        assert!(vid.starts_with("data:video/mp4;base64,"));
        let bytes = decode_data_uri_video(&vid).unwrap();
        assert_eq!(&bytes[0..4], b"\x00\x00\x00\x20"); // MP4 ftyp box size (32) for "ftypisom"
        assert!(hf_video_ref("[]").is_err());
        assert!(hf_video_ref(r#"[{"video": ""}]"#).is_err());
        assert!(decode_data_uri_video("https://example.com/v.mp4").is_err());
    }
    #[test]
    fn hf_gguf_files_empty_when_no_gguf() {
        let m = serde_json::json!({"siblings": [
            {"rfilename": "model.safetensors", "size": 900},
            {"rfilename": "config.json", "size": 10}
        ]});
        assert!(hf_gguf_files(&m).is_empty());
    }
    #[test]
    fn ollama_name_from_repo_is_lowercase_and_safe() {
        assert_eq!(ollama_name_from_repo("Comfy-Org/Qwen-Image-2.1"), "qwen-image-2.1");
        assert_eq!(ollama_name_from_repo("mlabonne_Qwen3-0.6B-abliterated-Q4_K_M.gguf"), "mlabonne_qwen3-0.6b-abliterated-q4_k_m");
    }

    #[test]
    fn ws_path_allows_workspace_relative() {
        let p = ws_path("hello.txt").unwrap();
        assert!(p.ends_with("hello.txt"));
        let p2 = ws_path("a/b/c.txt").unwrap();
        assert!(p2.ends_with("a/b/c.txt"));
    }

    #[test]
    fn ws_path_allows_any_path() {
        // No confinement: absolute paths and `..` are accepted now.
        assert!(ws_path("/etc/passwd").is_ok());
        assert!(ws_path("/root/odysseus/data/app.db").is_ok());
        assert!(ws_path("../main.rs").is_ok());
        assert!(ws_path("a/../../etc/passwd").is_ok());
        assert!(ws_path("~/secret").is_ok());
        assert!(ws_path("/sdcard/odysseus_rust").is_ok());
        assert!(ws_path("/root/odysseus-rust").is_ok());
    }

    #[test]
    fn validate_workspace_accepts_any_absolute_folder() {
        assert!(validate_workspace("/root/my-work").is_ok());
        assert!(validate_workspace("/root/odysseus-rust/workspace-test").is_ok());
        assert!(validate_workspace("/root/odysseus-rust/workspace").is_ok());
        assert!(validate_workspace("/sdcard/odysseus_rust").is_ok());
        assert!(validate_workspace("/storage/emulated/0/odysseus_rust").is_ok());
        // Previously protected paths are now accepted: the workspace is
        // only a default folder, not a boundary.
        assert!(validate_workspace("/root/odysseus").is_ok());
        assert!(validate_workspace("/root/.ssh").is_ok());
        assert!(validate_workspace("/etc").is_ok());
        assert!(validate_workspace("/root/odysseus-rust/src").is_ok());
    }

    #[test]
    fn validate_workspace_rejects_bad_paths() {
        // Only sanity checks remain: empty, relative, the filesystem root,
        // and paths containing '..'.
        assert!(validate_workspace("").is_err());
        assert!(validate_workspace("workspace").is_err());
        assert!(validate_workspace("relative/path").is_err());
        assert!(validate_workspace("/").is_err());
        assert!(validate_workspace("/root/../etc/passwd").is_err());
        assert!(validate_workspace("/sdcard/../etc/passwd").is_err());
    }

    #[test]
    fn chat_deserializes_without_updated_at() {
        // The browser's new-chat object never sends updated_at; saving must
        // not fail to deserialize. put_chat overwrites it with now_secs().
        let c: Chat = serde_json::from_str(
            r#"{"id":"abc","title":"Hi","model":"m","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(c.id, "abc");
        assert_eq!(c.updated_at, 0);
    }


    #[test]
    fn parse_tool_call_fenced() {
        let reply = "I will write the file.\n```tool\nwrite_file\nhello.txt\nHello Mawaxi\n```";
        let (tool, args) = parse_tool_call(reply).unwrap();
        assert_eq!(tool, "write_file");
        let (path, content) = normalise_tool_args(&tool, &args);
        assert_eq!(path, "hello.txt");
        assert_eq!(content, "Hello Mawaxi");
    }

    #[test]
    fn parse_tool_call_search_images() {
        let reply = "```tool\nsearch_images\ngiraffe\n```";
        let (tool, args) = parse_tool_call(reply).unwrap();
        assert_eq!(tool, "search_images");
        assert_eq!(args, "giraffe");
    }

    #[test]
    fn parse_tool_call_quoted_args() {
        let reply = "```tool\nwrite_file\n\"hello.txt\"\n\"Hello Mawaxi\"\n```";
        let (tool, args) = parse_tool_call(reply).unwrap();
        assert_eq!(tool, "write_file");
        let (path, content) = normalise_tool_args(&tool, &args);
        assert_eq!(path, "hello.txt");
        assert_eq!(content, "Hello Mawaxi");
    }

    #[test]
    fn parse_tool_call_xml_agness() {
        // Agnes emits <tool_call> XML instead of a fenced block; the tool
        // name is the first line after <function=tool>.
        let reply = "<tool_call>\n<function=tool>\nrun_shell\ncd /root && echo hi\n</function>\n</tool_call>";
        let (tool, args) = parse_tool_call(reply).unwrap();
        assert_eq!(tool, "run_shell");
        assert_eq!(args, "cd /root && echo hi");
    }

    #[test]
    fn parse_tool_call_xml_named_tag() {
        // The tool name may also ride in the tag itself.
        let reply = "<tool_call>\n<function=run_shell>\nls -la\n</function>\n</tool_call>";
        let (tool, args) = parse_tool_call(reply).unwrap();
        assert_eq!(tool, "run_shell");
        assert_eq!(args, "ls -la");
    }

    #[test]
    fn parse_tool_call_bare_tool_tag() {
        let reply = "Checking.\n<tool\nrun_shell\nls -la\n</tool>";
        let (tool, args) = parse_tool_call(reply).unwrap();
        assert_eq!(tool, "run_shell");
        assert_eq!(args, "ls -la");
    }

    #[test]
    fn parse_tool_call_dsml() {
        let reply = "Sure.\n<\u{200b}\u{ff5c}DSML\u{ff5c} calls>\n<\u{200b}\u{ff5c}DSML\u{ff5c} invoke name=\"run_shell\">\n<\u{200b}\u{ff5c}DSML\u{ff5c} parameter name=\"command\" string=\"true\">ls -la<\u{200b}/\u{ff5c}DSML\u{ff5c} parameter>\n<\u{200b}/\u{ff5c}DSML\u{ff5c} invoke>\n<\u{200b}/\u{ff5c}DSML\u{ff5c} calls>";
        let (tool, args) = parse_tool_call(reply).unwrap();
        assert_eq!(tool, "run_shell");
        assert_eq!(args, "ls -la");
    }

    #[test]
    fn parse_tool_call_plain_answer_is_none() {
        assert!(parse_tool_call("Hello! How can I help?").is_none());
    }

    #[test]
    fn create_request_detection() {
        assert!(looks_like_create_request(
            "Create and run a hospital management system in python"
        ));
        assert!(looks_like_create_request("Build me a todo app"));
        assert!(!looks_like_create_request("What is the capital of France?"));
        assert!(!looks_like_create_request("Hello, how are you?"));
    }

    #[test]
    fn transient_error_detection() {
        assert!(is_transient_model_error("Endpoint error: 503 Service Unavailable"));
        assert!(is_transient_model_error("Endpoint error: 429 Too Many Requests"));
        assert!(is_transient_model_error("Endpoint error: 500 Internal Server Error"));
        assert!(is_transient_model_error("Endpoint error: 502 Bad Gateway"));
        assert!(is_transient_model_error("request timed out"));
        assert!(is_transient_model_error("error sending request for url (https://example.com/v1/chat/completions)"));
        assert!(!is_transient_model_error("Endpoint error: 401 Unauthorized"));
        assert!(!is_transient_model_error("Endpoint error: 404 Not Found"));
        assert!(!is_transient_model_error("Endpoint error: 400 Bad Request"));
        assert!(!is_transient_model_error("empty reply from endpoint"));
    }

    fn create_and_run_detection() {
        assert!(is_create_and_run_request(
            "Create and run a hospital management system in python"
        ));
        assert!(is_create_and_run_request("Build and run the app"));
        assert!(is_create_and_run_request("Make a script and execute it"));
        assert!(!is_create_and_run_request(
            "Create a file called notes.txt with a shopping list"
        ));
        assert!(!is_create_and_run_request("Run the tests"));
        assert!(!is_create_and_run_request("Hello, how are you?"));
    }

    #[test]
    fn auto_run_commands_python() {
        let (compile, run) = auto_run_commands("hospital.py").unwrap();
        assert_eq!(compile, "python3 -m py_compile 'hospital.py'");
        assert!(run.contains("printf"));
        assert!(run.contains("timeout 10 python3 'hospital.py'"));
        assert!(auto_run_commands("notes.txt").is_none());
        assert!(auto_run_commands("app.sh").is_none());
    }

    #[test]
    fn shell_quote_cases() {
        assert_eq!(shell_quote("hospital.py"), "'hospital.py'");
        assert_eq!(shell_quote("my file.py"), "'my file.py'");
        assert_eq!(shell_quote("a'b.py"), "'a'\\''b.py'");
    }

    #[test]
    fn pasted_code_detection() {
        assert!(reply_pasted_code(
            "Here is the code:\n```python\nprint('hi')\n```"
        ));
        assert!(reply_pasted_code(
            "import os\ndef main():\n    print('hi')"
        ));
        assert!(!reply_pasted_code(
            "I created hospital.py and ran it. It printed a menu."
        ));
    }

    #[test]
    fn preset_gemini() {
        let p = preset_for("gemini").unwrap();
        assert_eq!(p.base_url, "https://generativelanguage.googleapis.com/v1beta/openai/");
        assert_eq!(p.model, "gemini-2.5-flash");
    }

    #[test]
    fn preset_nvidia() {
        let p = preset_for("nvidia").unwrap();
        assert_eq!(p.base_url, "https://integrate.api.nvidia.com/v1");
        assert_eq!(p.model, "deepseek-v4.1-flash");
    }

    #[test]
    fn preset_openai() {
        let p = preset_for("openai").unwrap();
        assert_eq!(p.base_url, "https://api.openai.com/v1");
        assert_eq!(p.model, "gpt-4o");
    }

    #[test]
    fn preset_custom_is_blank() {
        let p = preset_for("custom").unwrap();
        assert_eq!(p.base_url, "");
        assert_eq!(p.model, "");
    }

    #[test]
    fn preset_unknown_is_none() {
        assert!(preset_for("shannon").is_none());
        assert!(preset_for("").is_none());
    }

    #[test]
    fn parse_models_list_openai() {
        let v = serde_json::json!({"data": [{"id": "gemini-2.5-flash"}, {"id": "gemini-2.0-flash"}, {"id": "gemini-2.5-flash"}]});
        assert_eq!(parse_models_list(&v), vec!["gemini-2.0-flash", "gemini-2.5-flash"]);
    }

    #[test]
    fn parse_models_list_models_key() {
        let v = serde_json::json!({"models": ["b", "a"]});
        assert_eq!(parse_models_list(&v), vec!["a", "b"]);
    }

    #[test]
    fn parse_models_list_bare_array() {
        let v = serde_json::json!(["x", "y"]);
        assert_eq!(parse_models_list(&v), vec!["x", "y"]);
    }

    #[test]
    fn parse_models_list_ignores_bad_entries() {
        let v = serde_json::json!({"data": [{"id": ""}, {"id": "a"}, {"nope": 1}, {"id": "a"}]});
        assert_eq!(parse_models_list(&v), vec!["a"]);
    }

    #[test]
    fn parse_models_list_unknown_shape_is_empty() {
        let v = serde_json::json!({"foo": 1});
        assert!(parse_models_list(&v).is_empty());
    }

    #[test]
    fn parse_models_list_strips_google_prefix() {
        let v = serde_json::json!({"data": [{"id": "models/gemini-2.5-flash"}, {"id": "models/gemini-2.5-pro"}]});
        assert_eq!(parse_models_list(&v), vec!["gemini-2.5-flash", "gemini-2.5-pro"]);
    }

    #[test]
    fn models_list_url_trims_slash() {
        assert_eq!(models_list_url("https://example.com/v1/"), "https://example.com/v1/models");
        assert_eq!(models_list_url("https://example.com/v1"), "https://example.com/v1/models");
    }

    #[test]
    fn index_html_includes_puter() {
        // Puter (puter.js) free higher-end models must be served in the page:
        // the CDN script, the picker group, and the browser-side chat wiring.
        assert!(INDEX_HTML.contains("https://js.puter.com/v2/"));
        assert!(INDEX_HTML.contains("Puter (free)"));
        assert!(INDEX_HTML.contains("openai/gpt-5.5"));
        assert!(INDEX_HTML.contains("anthropic/claude-opus-5-5"));
        assert!(INDEX_HTML.contains("runPuterChat"));
    }

    #[test]
    fn copilot_base_detection() {
        assert!(is_copilot_base("https://api.githubcopilot.com"));
        assert!(is_copilot_base("https://api.githubcopilot.com/v1"));
        assert!(is_copilot_base("https://api.enterprise.githubcopilot.com"));
        assert!(!is_copilot_base("https://api.openai.com/v1"));
        assert!(!is_copilot_base("https://example.com"));
    }

    #[test]
    fn copilot_preset_exists() {
        let p = preset_for("copilot").expect("copilot preset");
        assert_eq!(p.base_url, "https://api.githubcopilot.com");
        assert_eq!(p.label, "GitHub Copilot");
    }

    #[test]
    fn messages_have_images_detects() {
        let plain = OllamaMessage { role: "user".into(), content: "hi".into(), images: None };
        assert!(!messages_have_images(&[plain]));
        let with_img = OllamaMessage {
            role: "user".into(),
            content: "hi".into(),
            images: Some(vec!["abc".into()]),
        };
        assert!(messages_have_images(&[with_img]));
    }

    #[test]
    fn index_html_includes_copilot_signin() {
        // The Endpoints tab must offer GitHub Copilot with a device-flow
        // sign-in (no API key), matching Odysseus.
        assert!(INDEX_HTML.contains("GitHub Copilot"));
        assert!(INDEX_HTML.contains("id=\"copilotSignin\""));
        assert!(INDEX_HTML.contains("id=\"copilotStatus\""));
        assert!(INDEX_HTML.contains("/api/copilot/device/start"));
        assert!(INDEX_HTML.contains("/api/copilot/device/poll"));
        assert!(INDEX_HTML.contains("github.com/login/device"));
    }

    #[test]
    fn index_html_includes_workspace_setting_and_chat_dates() {
        // Settings must offer a Workspace tab with a path field and Save, and
        // the chat list must show a date under each title.
        assert!(INDEX_HTML.contains("data-tab=\"workspace\""));
        assert!(INDEX_HTML.contains("id=\"tab-workspace\""));
        assert!(INDEX_HTML.contains("id=\"wsPath\""));
        assert!(INDEX_HTML.contains("id=\"wsSave\""));
        assert!(INDEX_HTML.contains("loadWorkspaceSetting"));
        assert!(INDEX_HTML.contains("fmtChatDate"));
        assert!(INDEX_HTML.contains(".chatitem .d"));
        assert!(INDEX_HTML.contains("id=\"wsPathLabel\""));
    }

    #[test]
    fn index_html_includes_video_generation() {
        // Chat must auto-detect video requests, generate inline, and offer a
        // Video tab in Settings for the Magic Hour key.
        assert!(INDEX_HTML.contains("videoPromptFromText"));
        assert!(INDEX_HTML.contains("generateVideoFromInput"));
        assert!(INDEX_HTML.contains("id=\"vidBtn\""));
        assert!(INDEX_HTML.contains("data-tab=\"video\""));
        assert!(INDEX_HTML.contains("id=\"tab-video\""));
        assert!(INDEX_HTML.contains("id=\"mhKey\""));
        assert!(INDEX_HTML.contains("id=\"mhSave\""));
        assert!(INDEX_HTML.contains("loadVideoSetting"));
        assert!(INDEX_HTML.contains("/api/video"));
        assert!(INDEX_HTML.contains("magic_hour_configured"));
    }

    #[test]
    fn index_html_includes_sidebar_feature_buttons() {
        // The sidebar button area must hold all ten feature buttons, and
        // the footer must keep only Settings (Memory/Skills moved up).
        for id in [
            "id=\"openWorkspace\"",
            "id=\"openMemory\"",
            "id=\"openSkills\"",
            "id=\"openNotes\"",
            "id=\"openResearch\"",
            "id=\"openCookbook\"",
            "id=\"openLibrary\"",
            "id=\"openBrowser\"",
            "id=\"openCalendar\"",
            "id=\"openPdf\"",
        ] {
            assert!(INDEX_HTML.contains(id), "missing sidebar button {id}");
        }
        assert!(INDEX_HTML.contains("termBubble"), "terminal bubble missing");
        assert!(INDEX_HTML.contains("renderA2UI"), "A2UI renderer missing");
        assert!(INDEX_HTML.contains("a2ui-surface"), "A2UI surface CSS missing");
        assert!(INDEX_HTML.contains("a2uiAction"), "A2UI action handler missing");
        assert!(AGENT_SYSTEM.contains("A2UI"), "agent system prompt missing A2UI hint");
        assert!(AGENT_SYSTEM.contains("Do not use asterisks"), "agent system prompt missing no-asterisk rule");
        assert!(INDEX_HTML.contains("id=\"stopAgentBtn\""), "stop button missing");
        assert!(INDEX_HTML.contains("id=\"sideBtns\""));
        assert!(INDEX_HTML.contains("id=\"featureView\""));
        assert!(INDEX_HTML.contains("id=\"fvTitle\""));
        assert!(INDEX_HTML.contains("id=\"fvBody\""));
        assert!(INDEX_HTML.contains("openNotesView"));
        assert!(INDEX_HTML.contains("openResearchView"));
        assert!(INDEX_HTML.contains("openCookbookView"));
        assert!(INDEX_HTML.contains("openLibraryView"));
        // Notes, Cookbook and Research must call their server endpoints.
        assert!(INDEX_HTML.contains("/api/notes"));
        assert!(INDEX_HTML.contains("/api/cookbook/search"));
        assert!(INDEX_HTML.contains("/api/cookbook/cached"));
        assert!(INDEX_HTML.contains("/api/cookbook/running"));
        assert!(INDEX_HTML.contains("/api/cookbook/download"));
        assert!(INDEX_HTML.contains("/api/cookbook/import"));
        assert!(INDEX_HTML.contains("downloadCbGguf"));
        assert!(INDEX_HTML.contains("No GGUF file"));
        assert!(INDEX_HTML.contains("/api/research"));
        assert!(INDEX_HTML.contains("/api/research/cancel"));
        assert!(INDEX_HTML.contains("buildResearchModelSelect"));
        assert!(INDEX_HTML.contains("id=\"rsModel\""));
        assert!(INDEX_HTML.contains("/api/workspace/file"));
    }

    #[test]
    fn index_html_includes_browser_themes_datasaver_calendar_pdf() {
        // Browser: sidebar button, view, address bar, server fetch, reader
        // fallback, and chat links opening in it.
        assert!(INDEX_HTML.contains("id=\"openBrowser\""));
        assert!(INDEX_HTML.contains("openBrowserView"));
        assert!(INDEX_HTML.contains("/api/browser/fetch"));
        assert!(INDEX_HTML.contains("browseTo"));
        assert!(INDEX_HTML.contains("openLinkInBrowser"));
        // Themes: picker in Settings, theme definitions, apply + persist.
        assert!(INDEX_HTML.contains("data-tab=\"theme\""));
        assert!(INDEX_HTML.contains("id=\"tab-theme\""));
        assert!(INDEX_HTML.contains("THEMES"));
        assert!(INDEX_HTML.contains("applyTheme"));
        assert!(INDEX_HTML.contains("mawaxi-theme"));
        assert!(INDEX_HTML.contains("Snapchat"));
        assert!(INDEX_HTML.contains("Spotify"));
        assert!(INDEX_HTML.contains("TikTok"));
        assert!(INDEX_HTML.contains("WhatsApp"));
        assert!(INDEX_HTML.contains("Instagram"));
        assert!(INDEX_HTML.contains("Sunset"));
        assert!(INDEX_HTML.contains("Lavender"));
        // Chat text size + zoom: size picker in Settings, zoom buttons in header.
        assert!(INDEX_HTML.contains("mawaxi-chat-font-size"));
        assert!(INDEX_HTML.contains("id=\"fontSizeRow\""));
        assert!(INDEX_HTML.contains("applyChatFontSize"));
        assert!(INDEX_HTML.contains("CHAT_FONT_SIZES"));
        assert!(INDEX_HTML.contains("mawaxi-chat-zoom"));
        assert!(INDEX_HTML.contains("id=\"zoomIn\""));
        assert!(INDEX_HTML.contains("id=\"zoomOut\""));
        assert!(INDEX_HTML.contains("id=\"zoomLabel\""));
        assert!(INDEX_HTML.contains("applyChatZoom"));
        assert!(INDEX_HTML.contains("--chat-font-size"));
        assert!(INDEX_HTML.contains("--chat-zoom"));
        // Chat background image: picker in Settings > Theme, behind the bubbles.
        assert!(INDEX_HTML.contains("mawaxi-chat-bg"));
        assert!(INDEX_HTML.contains("id=\"bgUploadBtn\""));
        assert!(INDEX_HTML.contains("id=\"bgRemoveBtn\""));
        assert!(INDEX_HTML.contains("id=\"bgFile\""));
        assert!(INDEX_HTML.contains("applyChatBg"));
        assert!(INDEX_HTML.contains("clearChatBg"));
        assert!(INDEX_HTML.contains("handleBgFile"));
        // Data Saver: toggle in Settings, persisted, wired into image/video.
        assert!(INDEX_HTML.contains("data-tab=\"data\""));
        assert!(INDEX_HTML.contains("id=\"tab-data\""));
        assert!(INDEX_HTML.contains("mawaxi-data-saver"));
        assert!(INDEX_HTML.contains("dataSaver"));
        assert!(INDEX_HTML.contains("tapToLoadBox"));
        // Calendar: sidebar button, month view, server endpoints.
        assert!(INDEX_HTML.contains("id=\"openCalendar\""));
        assert!(INDEX_HTML.contains("openCalendarView"));
        assert!(INDEX_HTML.contains("/api/calendar"));
        // PDF: sidebar button, upload/view/edit, chat + research download.
        assert!(INDEX_HTML.contains("id=\"openPdf\""));
        assert!(INDEX_HTML.contains("openPdfView"));
        assert!(INDEX_HTML.contains("/api/pdf/create"));
        assert!(INDEX_HTML.contains("/api/pdf/upload"));
        assert!(INDEX_HTML.contains("accept=\"image/*,application/pdf,.pdf\""));
        assert!(INDEX_HTML.contains("pendingPdf"));
        assert!(INDEX_HTML.contains("attachPdfFile"));
        assert!(INDEX_HTML.contains("/api/pdf/list"));
        assert!(INDEX_HTML.contains("/api/pdf/text"));
        assert!(INDEX_HTML.contains("/api/pdf/edit"));
        assert!(INDEX_HTML.contains("downloadChatPdf"));
        assert!(INDEX_HTML.contains("/api/pdf/save"));
        assert!(INDEX_HTML.contains("pdfRequestFromText"));
        assert!(INDEX_HTML.contains("generatePdfFromInput"));
        // Edited PDF: after saving, the Edit screen shows a Download link for
        // the new file, and every row in the PDF list has a Download control.
        assert!(INDEX_HTML.contains("Download it here"));
        assert!(INDEX_HTML.contains("dl.textContent='Download'"));
        // Format-preserving edit: the edit screen states the original
        // format is kept (the server calls the PyMuPDF helper).
        assert!(INDEX_HTML.contains("The original format is kept"));
        // Coloured research PDF: the Download PDF buttons open a print view
        // that keeps the colours (print-color-adjust: exact) and the
        // VISUAL view has a Save as PDF (colour) button.
        assert!(INDEX_HTML.contains("researchPrintDoc"));
        assert!(INDEX_HTML.contains("printResearchReport"));
        assert!(INDEX_HTML.contains("print-color-adjust:exact"));
        assert!(INDEX_HTML.contains("Save as PDF (colour)"));
        assert!(INDEX_HTML.contains("Save as PDF"));
    }

    #[test]
    fn now_unix_is_plausible() {
        assert!(now_unix() > 1_700_000_000);
    }

    #[test]
    fn note_round_trip_json() {
        let note = Note {
            id: "abc".to_string(),
            title: "Test".to_string(),
            body: "body".to_string(),
            updated_at: 1_700_000_000,
        };
        let s = serde_json::to_string(&note).unwrap();
        let back: Note = serde_json::from_str(&s).unwrap();
        assert_eq!(back.title, "Test");
    }

    #[test]
    fn calendar_event_round_trip_json() {
        let ev = CalendarEvent {
            id: "abc".to_string(),
            title: "Dentist".to_string(),
            date: "2026-11-25".to_string(),
            time: "10:30".to_string(),
            notes: "Bring card".to_string(),
        };
        let s = serde_json::to_string(&ev).unwrap();
        let back: CalendarEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back.date, "2026-11-25");
        assert_eq!(back.time, "10:30");
        assert!(valid_cal_date("2026-11-25"));
        assert!(!valid_cal_date("2026-13-01"));
        assert!(!valid_cal_date("25-11-2026"));
        assert!(!valid_cal_date("2026-11-5"));
    }

    #[test]
    fn normalize_browser_url_cases() {
        assert_eq!(normalize_browser_url("example.com"), "https://example.com");
        assert_eq!(
            normalize_browser_url("https://example.com/x"),
            "https://example.com/x"
        );
        assert_eq!(
            normalize_browser_url("cats and dogs"),
            "https://www.bing.com/search?q=cats%20and%20dogs"
        );
        assert_eq!(normalize_browser_url(""), "");
    }

    #[test]
    fn frameable_from_headers_cases() {
        use reqwest::header::HeaderMap;
        let empty = HeaderMap::new();
        assert!(frameable_from_headers(&empty));
        let mut deny = HeaderMap::new();
        deny.insert("x-frame-options", "DENY".parse().unwrap());
        assert!(!frameable_from_headers(&deny));
        let mut same = HeaderMap::new();
        same.insert("x-frame-options", "SAMEORIGIN".parse().unwrap());
        assert!(!frameable_from_headers(&same));
        let mut csp = HeaderMap::new();
        csp.insert(
            "content-security-policy",
            "default-src 'self'; frame-ancestors 'none'".parse().unwrap(),
        );
        assert!(!frameable_from_headers(&csp));
    }

    #[test]
    fn html_title_extracted() {
        assert_eq!(
            html_title("<html><head><title>  Hello &amp; World </title></head>"),
            "Hello & World"
        );
        assert_eq!(html_title("<html><head></head>"), "");
    }

    #[test]
    fn wrap_pdf_lines_wraps_long_text() {
        let lines = wrap_pdf_lines("one two three four five six seven eight", 10);
        assert!(lines.len() > 1);
        for l in &lines {
            assert!(l.chars().count() <= 10);
        }
        assert_eq!(wrap_pdf_lines("hello", 90), vec!["hello".to_string()]);
        assert_eq!(wrap_pdf_lines("", 90), vec![String::new()]);
    }

    #[test]
    fn pdf_slug_cases() {
        assert_eq!(pdf_slug("My Chat 2026"), "my-chat-2026");
        assert_eq!(pdf_slug("!!!"), "document");
        assert_eq!(pdf_slug("Kirya Emmanuel"), "kirya-emmanuel");
    }

    #[test]
    fn agent_can_create_pdfs() {
        assert!(AGENT_SYSTEM.contains("create_pdf"));
        assert!(AGENT_SYSTEM.contains("Never say you cannot create a PDF"));
        assert!(is_tool_name("create_pdf"));
        let sp = build_system_prompt(None, &[], None, false).expect("prompt exists");
        assert!(sp.contains("Never say you cannot create PDFs"));
    }

    #[test]
    fn save_pdf_to_library_writes_valid_pdf() {
        let dir = std::env::temp_dir().join(format!("mawaxi-pdf-test-{}", now_secs()));
        let _ = std::fs::create_dir_all(&dir);
        let index = dir.join("pdfs.json");
        let entry = save_pdf_to_library_at(
            dir.to_str().unwrap(),
            index.to_str().unwrap(),
            "Test Note",
            "Hello PDF.\nSecond line.",
        )
        .expect("save works");
        assert_eq!(entry.name, "test-note.pdf");
        let bytes =
            std::fs::read(dir.join(format!("{}.pdf", entry.id))).expect("pdf file exists");
        assert!(bytes.starts_with(b"%PDF-1.4\n"));
        let entries: Vec<PdfEntry> = load_json(index.to_str().unwrap());
        assert!(entries.iter().any(|e| e.id == entry.id));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn make_pdf_is_valid() {
        let title = "Test Report (draft)";
        let text = "Hello world.\nSecond line with (parens) and a \\ backslash.\n\n".to_string()
            + &"A long paragraph that keeps going so the writer must wrap it across several lines and several pages. ".repeat(40);
        let pdf = make_pdf(title, &text);
        assert!(pdf.starts_with(b"%PDF-1.4\n"));
        assert!(pdf.ends_with(b"%%EOF\n"));
        let s = String::from_utf8_lossy(&pdf);
        // Title text (escaped) is in the first page's content stream.
        assert!(s.contains("Test Report \\(draft\\)"));
        assert!(s.contains("Hello world."));
        assert!(s.contains("/Type /Catalog"));
        assert!(s.contains("/BaseFont /Helvetica-Bold"));
        // The xref table must point at each object's header.
        let xref = s.find("xref\n").expect("xref missing");
        let trailer = s[xref..].find("trailer").expect("trailer missing");
        let table = &s[xref + 5..xref + trailer];
        let mut lines = table.lines();
        let n_objs = s.matches(" 0 obj\n").count();
        let first = lines.next().unwrap();
        assert_eq!(first.trim(), format!("0 {}", n_objs + 1));
        let _ = lines.next().unwrap(); // free entry for object 0
        for (i, line) in lines.enumerate() {
            let off: usize = line[..10].trim().parse().unwrap();
            let expect = format!("{} 0 obj", i + 1);
            assert!(
                s[off..].starts_with(&expect),
                "xref entry {} points at {:?}, expected {expect}",
                i + 1,
                &s[off..off + 12.min(s.len() - off)]
            );
        }
    }

    /// A tiny real 8x6 JPEG (red), generated with PyMuPDF, for PDF tests.
    fn test_jpeg_8x6() -> Vec<u8> {
        base64_decode("/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wgARCAAGAAgDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAAB//EABUBAQEAAAAAAAAAAAAAAAAAAAcI/9oADAMBAAIQAxAAAAEXKb+//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABAf/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPxB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxB//9k=").unwrap()
    }

    #[test]
    fn jpeg_dims_parses_sof() {
        let jpg = test_jpeg_8x6();
        assert_eq!(jpeg_dims(&jpg), Some((8, 6, 3)));
        assert_eq!(jpeg_dims(b"not a jpeg"), None);
        assert_eq!(jpeg_dims(b"\xff\xd8\xff\xe0\x00\x10JFIF"), None);
    }

    #[test]
    fn make_pdf_embeds_jpeg_and_placeholders() {
        let jpg = test_jpeg_8x6();
        let blocks = vec![
            PdfBlock::Text("A smear photo follows.".to_string()),
            PdfBlock::Image(PdfImageReady {
                jpeg: Some(jpg.clone()),
                width: 8,
                height: 6,
                title: "Giraffe photo".to_string(),
                source: "https://example.com/giraffe".to_string(),
            }),
            PdfBlock::Image(PdfImageReady {
                jpeg: None,
                width: 0,
                height: 0,
                title: "Blocked photo".to_string(),
                source: "https://example.com/blocked".to_string(),
            }),
        ];
        let pdf = make_pdf_blocks("Photo Report", &blocks);
        assert!(pdf.starts_with(b"%PDF-1.4\n"));
        assert!(pdf.ends_with(b"%%EOF\n"));
        let s = String::from_utf8_lossy(&pdf);
        assert!(s.contains("/Subtype /Image"));
        assert!(s.contains("/DCTDecode"));
        assert!(s.contains("/Width 8"));
        assert!(s.contains("/Height 6"));
        assert!(s.contains("/Im0 Do"));
        assert!(s.contains("Giraffe photo"));
        assert!(s.contains("https://example.com/giraffe"));
        // The failed photo is a placeholder line, not a broken image.
        assert!(s.contains("[Image not available: Blocked photo]"));
        assert!(s.contains("https://example.com/blocked"));
        // The JPEG bytes themselves are in the file.
        assert!(pdf.windows(jpg.len()).any(|w| w == jpg.as_slice()));
    }

    #[test]
    fn extract_md_images_splits() {
        let text = "Here is a smear ![P. falciparum rings](https://example.com/rings.jpg) and another ![Giraffe](https://example.com/g.jpg) done.";
        let (clean, imgs) = extract_md_images(text);
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[0].url, "https://example.com/rings.jpg");
        assert_eq!(imgs[0].title, "P. falciparum rings");
        assert_eq!(imgs[1].url, "https://example.com/g.jpg");
        assert!(clean.contains("[Image: P. falciparum rings]"));
        assert!(clean.contains("[Image: Giraffe]"));
        assert!(!clean.contains("!["));
        let (clean2, imgs2) = extract_md_images("plain text, no images");
        assert!(imgs2.is_empty());
        assert_eq!(clean2, "plain text, no images");
    }

    #[test]
    fn cookbook_model_round_trip_json() {
        let m = HfModel {
            id: "Qwen/Qwen3-0.6B".to_string(),
            author: "Qwen".to_string(),
            downloads: 123,
            likes: 45,
            pipeline_tag: "text-generation".to_string(),
            last_modified: "2025-01-01".to_string(),
            size_bytes: Some(123456789),
            gguf_files: vec![],
        };
        let s = serde_json::to_string(&m).unwrap();
        let back: HfModel = serde_json::from_str(&s).unwrap();
        assert_eq!(back.id, "Qwen/Qwen3-0.6B");
        assert_eq!(back.downloads, 123);
        assert_eq!(back.size_bytes, Some(123456789));
    }

    #[test]
    fn research_system_prompt_is_structured() {
        assert!(RESEARCH_SYSTEM.contains("Background"));
        assert!(RESEARCH_SYSTEM.contains("Key points"));
        assert!(RESEARCH_SYSTEM.contains("Open questions"));
        assert!(RESEARCH_SYSTEM.contains("Conclusion"));
    }

    #[test]
    fn research_html_to_text_strips_tags_and_scripts() {
        let html = "<html><head><style>.x{color:red}</style><script>var a=1;</script></head><body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>";
        let t = html_to_text(html);
        assert!(t.contains("Title"));
        assert!(t.contains("Hello world"));
        assert!(!t.contains("color:red"));
        assert!(!t.contains("var a"));
        assert!(!t.contains('<'));
    }

    #[test]
    fn research_extract_relevant_keeps_keyword_sentences() {
        let text = "Cats are small animals. The sky is blue because of Rayleigh scattering. Dogs bark loudly. Rayleigh scattering makes the sky appear blue to our eyes.";
        let kws = research_keywords("why is the sky blue", &[]);
        let out = extract_relevant(text, &kws, 300);
        assert!(out.contains("Rayleigh"));
        assert!(out.contains("sky"));
    }

    #[test]
    fn research_bing_block_parses_title_and_url() {
        let block = "<li class=\"b_algo\"><h2><a href=\"https://en.wikipedia.org/wiki/Rayleigh_scattering\">Rayleigh scattering - Wikipedia</a></h2><p>Some snippet.</p></li>";
        let (title, url) = bing_parse_block(block).expect("should parse");
        assert!(title.contains("Rayleigh scattering"));
        assert_eq!(url, "https://en.wikipedia.org/wiki/Rayleigh_scattering");
    }

    #[test]
    fn research_bing_block_parses_tilk_anchor() {
        // Real Bing layout (Nov 2026): the anchor carries class="tilk"
        // with href NOT first, and the <h2> sits inside the anchor.
        let block = "<li class=\"b_algo\"><div class=\"b_tpcn\"><a class=\"tilk\" aria-label=\"en.wikipedia.org\" href=\"https://en.wikipedia.org/wiki/Rayleigh_scattering\" h=\"ID=SERP,5103.1\"><div class=\"tpic\"></div><h2><strong>Rayleigh scattering</strong> - Wikipedia</h2></a></div><div class=\"b_caption\"><p>Snippet.</p></div></li>";
        let (title, url) = bing_parse_block(block).expect("should parse");
        assert!(title.contains("Rayleigh scattering"));
        assert_eq!(url, "https://en.wikipedia.org/wiki/Rayleigh_scattering");
    }

    #[test]
    fn research_filter_relevant_drops_junk_results() {
        // Bing serves login pages and dictionary entries for real queries
        // from this phone (Nov 2026) — only keyword-matching results stay.
        let kws = research_keywords("why is the sky blue", &["Rayleigh scattering wavelength".to_string()]);
        let results = vec![
            SearchResult { title: "Rayleigh scattering - Wikipedia".to_string(), url: "https://en.wikipedia.org/wiki/Rayleigh_scattering".to_string() },
            SearchResult { title: "tiktok.comhttps://www.tiktok.com".to_string(), url: "https://www.tiktok.com/".to_string() },
            SearchResult { title: "WHY | English meaning - Cambridge Dictionary".to_string(), url: "https://dictionary.cambridge.org/dictionary/english/why".to_string() },
            SearchResult { title: "Sign in - Google Accounts".to_string(), url: "https://accounts.google.com/Login".to_string() },
        ];
        let kept = research_filter_relevant(results, &kws);
        assert_eq!(kept.len(), 1);
        assert!(kept[0].url.contains("Rayleigh"));
    }

    #[test]
    #[test]
    fn research_google_block_parses_url_q_links() {
        let html = "<div class=\"g\"><a href=\"/url?q=https://en.wikipedia.org/wiki/Rayleigh_scattering&sa=U&ved=1\"><h3>Rayleigh scattering - Wikipedia</h3></a></div><div class=\"g\"><a href=\"https://example.com/mie\"><h3>Mie scattering</h3></a></div>";
        let out = google_parse_results(html);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].title, "Rayleigh scattering - Wikipedia");
        assert_eq!(out[0].url, "https://en.wikipedia.org/wiki/Rayleigh_scattering");
        assert_eq!(out[1].url, "https://example.com/mie");
    }

    #[test]
    fn research_google_redirect_page_parses_nothing() {
        // The page Google serves this phone: a JS redirect notice, no results.
        let html = "<title>Google Search</title><p>Please click here if you are not redirected within a few seconds.</p>";
        assert!(google_parse_results(html).is_empty());
    }

    #[test]
    fn research_marginalia_block_parses_direct_links() {
        let html = "<a href=\"https://en.wikipedia.org/wiki/Rayleigh_scattering\">Rayleigh scattering</a><a href=\"https://en.wikipedia.org/wiki/Rayleigh_scattering\">https://en.wikipedia.org/wiki/Rayleigh_scattering</a><a href=\"https://github.com/MarginaliaSearch/MarginaliaSearch/issues\">GitHub issue tracker</a>";
        let out = marginalia_parse_results(html);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].title, "Rayleigh scattering");
        assert_eq!(out[0].url, "https://en.wikipedia.org/wiki/Rayleigh_scattering");
    }

    #[test]
    fn research_searxng_parse_maps_json_to_results() {
        let json = r#"{"results":[{"title":"Rayleigh scattering - Wikipedia","url":"https://en.wikipedia.org/wiki/Rayleigh_scattering","engines":["duckduckgo"]},{"title":"Mie scattering","url":"https://en.wikipedia.org/wiki/Mie_scattering","engines":["bing"]}]}"#;
        let out = parse_searxng_results(json);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].title, "Rayleigh scattering - Wikipedia");
        assert_eq!(out[0].url, "https://en.wikipedia.org/wiki/Rayleigh_scattering");
    }

    #[test]
    fn research_searxng_image_parse_maps_json_to_results() {
        let json = r#"{"results":[{"title":"Tall giraffe","img_src":"https://example.com/giraffe.jpg","thumbnail_src":"https://example.com/thumb.jpg","url":"https://example.com/photo/1"},{"title":"Giraffe In Wild","img_src":"https://example.com/giraffe2.jpg","thumbnail_src":"","url":"https://example.com/pin/2"}]}"#;
        let out = parse_searxng_image_results(json);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].image_url, "https://example.com/giraffe.jpg");
        assert_eq!(out[0].thumbnail_url, "https://example.com/thumb.jpg");
        assert_eq!(out[0].title, "Tall giraffe");
        assert_eq!(out[0].source_url, "https://example.com/photo/1");
        assert_eq!(out[1].image_url, "https://example.com/giraffe2.jpg");
    }

    #[test]
    fn agent_system_mentions_search_images_and_index_renders_md_images() {
        assert!(AGENT_SYSTEM.contains("search_images"));
        assert!(AGENT_SYSTEM.contains("![short description](direct image URL)"));
        assert!(INDEX_HTML.contains("loading=\"lazy\""));
    }

    #[test]
    fn research_topic_slug_makes_filename() {
        assert_eq!(research_topic_slug("THE HOUSE FLY"), "the-house-fly");
        assert_eq!(research_topic_slug("Why is the sky blue?"), "why-is-the-sky-blue");
        assert_eq!(research_topic_slug("  "), "research");
    }

    #[test]
    fn research_format_date_formats_unix_time() {
        // 2026-11-20 00:00 UTC.
        assert_eq!(research_format_date(1795132800), "2026-11-20 00:00 UTC");
    }

    #[test]
    fn research_entry_to_markdown_has_header_and_sources() {
        let e = ResearchHistoryEntry {
            id: "x".into(),
            topic: "THE HOUSE FLY".into(),
            report: "## Background\nFlies are insects.".into(),
            sources: vec![ResearchSource { title: "Housefly".into(), url: "https://en.wikipedia.org/wiki/Housefly".into() }],
            duration_secs: 134,
            created_at: 1795132800,
            model: "gemini-3.5-flash-lite".into(),
            endpoint: "Gemini".into(),
            rounds: 3,
            format: "full".into(),
            engine: "searxng".into(),
            status: "completed".into(),
            error: None,
        };
        let md = research_entry_to_markdown(&e);
        assert!(md.contains("# THE HOUSE FLY"));
        assert!(md.contains("2026-11-20 00:00 UTC"));
        assert!(md.contains("gemini-3.5-flash-lite"));
        assert!(md.contains("## Sources"));
        assert!(md.contains("[Housefly](https://en.wikipedia.org/wiki/Housefly)"));
    }

    #[test]
    fn research_wikipedia_parse_maps_titles_to_urls() {
        let json = r#"{"query":{"search":[{"title":"Rayleigh scattering","pageid":26176},{"title":"Mie scattering","pageid":123}]}}"#;
        let out = parse_wikipedia_results(json);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].url, "https://en.wikipedia.org/wiki/Rayleigh_scattering");
        assert_eq!(out[1].title, "Mie scattering");
    }

    #[test]
    fn research_plan_parse_falls_back_on_bad_json() {
        let p = parse_plan_json("not json at all", "why is the sky blue");
        assert_eq!(p.sub_questions.len(), 3);
        assert!(p.sub_questions[0].contains("sky blue"));
    }

    #[test]
    fn research_queries_parse_keeps_valid_json() {
        let q = parse_queries_json("[\"why is the sky blue\", \"Rayleigh scattering explained\"]", &[]);
        assert_eq!(q.len(), 2);
        assert!(q[0].contains("sky blue"));
    }

    #[test]
    fn research_rounds_default_matches_odysseus_eight() {
        // Odysseus DeepResearcher default max_rounds=8.
        assert_eq!(research_rounds_choice(None), 8);
        assert_eq!(research_rounds_choice(Some(1)), 1);
        assert_eq!(research_rounds_choice(Some(3)), 3);
        assert_eq!(research_rounds_choice(Some(5)), 5);
        assert_eq!(research_rounds_choice(Some(99)), 8);
        assert_eq!(research_rounds_choice(Some(0)), 1);
    }

    #[test]
    fn research_engine_defaults_to_searxng() {
        // Default is the dedicated local SearXNG instance (port 8888).
        assert_eq!(research_engine_choice(""), "searxng");
        assert_eq!(research_engine_choice("searxng"), "searxng");
        assert_eq!(research_engine_choice("bing"), "bing");
        assert_eq!(research_engine_choice("duckduckgo"), "duckduckgo");
        assert_eq!(research_engine_choice("brave"), "brave");
        assert_eq!(research_engine_choice("google"), "google");
        assert_eq!(research_engine_choice("marginalia"), "marginalia");
        assert_eq!(research_engine_choice("unknown-engine"), "searxng");
    }

    #[test]
    fn research_format_prompt_selects_template() {
        // Odysseus always writes the full structured report: "full" is
        // the default and unknown values fall back to it.
        assert!(research_format_prompt("full").contains("comprehensive"));
        assert!(research_format_prompt("brief").contains("brief summary"));
        assert!(research_format_prompt("bullets").contains("bullet list"));
        assert_eq!(research_format_prompt("weird"), RESEARCH_FINAL_PROMPT);
    }

    #[test]
    fn research_format_duration_is_m_ss() {
        assert_eq!(research_format_duration(0), "0:00");
        assert_eq!(research_format_duration(65), "1:05");
        assert_eq!(research_format_duration(187), "3:07");
        assert_eq!(research_format_duration(3600), "60:00");
    }

    #[test]
    fn research_history_entry_round_trips_json() {
        let e = ResearchHistoryEntry {
            id: "rs-1".to_string(),
            topic: "Why is the sky blue".to_string(),
            report: "Blue because of Rayleigh scattering.".to_string(),
            sources: vec![ResearchSource {
                title: "Rayleigh scattering".to_string(),
                url: "https://en.wikipedia.org/wiki/Rayleigh_scattering".to_string(),
            }],
            duration_secs: 187,
            created_at: 1795175000,
            model: "gemini-2.5-flash-lite".to_string(),
            endpoint: "Gemini".to_string(),
            rounds: 3,
            format: "full".to_string(),
            engine: "bing".to_string(),
            status: "completed".to_string(),
            error: None,
        };
        let s = serde_json::to_string(&e).unwrap();
        let back: ResearchHistoryEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(back.topic, "Why is the sky blue");
        assert_eq!(back.sources.len(), 1);
        assert_eq!(back.duration_secs, 187);
        assert_eq!(back.status, "completed");
        assert!(back.error.is_none());
    }

    #[test]
    fn research_history_ui_present() {
        assert!(INDEX_HTML.contains("/api/research/history"));
        assert!(INDEX_HTML.contains("rsHistTab"));
        assert!(INDEX_HTML.contains("rsHistSearch"));
        assert!(INDEX_HTML.contains("rsHistSelect"));
    }

    #[test]
    fn chat_bubbles_fit_screen() {
        // Bubbles must be relative to the chat area, not a fixed pixel
        // width, and must be able to shrink below their content width.
        assert!(INDEX_HTML.contains(".msg{max-width:min(720px,75%);min-width:0;font-size:var(--chat-font-size,16px)}"));
        // Long words/URLs must break inside the bubble.
        assert!(INDEX_HTML.contains("overflow-wrap:break-word;word-break:break-word"));
        // Media inside a bubble must scale to the bubble.
        assert!(INDEX_HTML.contains(".msg img,.msg video,.msg iframe{max-width:100%;height:auto}"));
        // Tables must scroll inside the bubble, not push it wider.
        assert!(INDEX_HTML.contains(".msg table{display:block;max-width:100%;overflow-x:auto}"));
        // Small screens get a wider bubble cap and tighter padding.
        assert!(INDEX_HTML.contains("@media (max-width:600px)"));
        assert!(INDEX_HTML.contains(".msg,.msg.term{max-width:94%}"));
    }

    #[test]
    fn chat_system_prompt_asks_for_real_photos_not_ai_images() {
        // Normal chat must not ask for AI-generated images (Steven's
        // word, 2026-11-21), but real web photos are welcome: the
        // prompt always carries the [[WEBIMAGE: ...]] hint and never
        // the old [[IMAGE: ...]] AI-image instruction.
        // Steven's word, 2026-11-23: replies must not include asterisks.
        let sp = build_system_prompt(None, &[], None, false).expect("prompt exists");
        assert!(sp.contains("[[WEBIMAGE:"));
        assert!(sp.contains("never AI-generated"));
        assert!(!sp.contains("[[IMAGE:"));
        assert!(!INDEX_HTML.contains("CHAT_IMAGE_HINT"));
        assert!(sp.contains("Do not use asterisks"));
    }

    #[test]
    fn local_model_prompt_has_no_webimage_marker() {
        // Regression (2026-11-24): the tiny local model
        // (qwen3-abliterated-0.6b) cannot follow the [[WEBIMAGE: ...]]
        // marker instruction -- it latches onto the marker syntax and
        // repeats it instead of answering ("Give total of 5 and 6" came
        // back as a stream of [[WEBIMAGE: ...]] markers, showing as "]"
        // in chat). So the local prompt omits the hint entirely, while
        // the no-asterisk rule and the PDF hint stay.
        // The local model gets NO system prompt at all.
        assert!(build_system_prompt(None, &[], None, true).is_none());
    }

    #[test]
    fn chat_personas() {
        // Personas (Steven's word, 2026-11-21): four built-in reply
        // styles. General adds nothing; the other three inject a short
        // preset into the plain-chat system prompt, and the picker sends
        // the chosen id with every chat/agent request.
        assert!(persona_prompt("general").is_none());
        assert!(persona_prompt("bogus").is_none());
        let med = persona_prompt("medical").expect("medical preset");
        assert!(med.contains("medical laboratory scientist"));
        assert!(med.contains("ASCP"));
        let teach = persona_prompt("teacher").expect("teacher preset");
        assert!(teach.contains("step by step"));
        let code = persona_prompt("coder").expect("coder preset");
        assert!(code.contains("code"));
        let sp = build_system_prompt(None, &[], Some("medical"), false).expect("prompt exists");
        assert!(sp.contains("medical laboratory scientist"));
        let plain = build_system_prompt(None, &[], None, false).expect("prompt exists");
        assert!(!plain.contains("medical laboratory scientist"));
        assert!(INDEX_HTML.contains("id=\"personaBtn\""));
        assert!(INDEX_HTML.contains("mawaxi-persona"));
        assert!(INDEX_HTML.contains("persona: currentPersona"));
    }

    #[test]
    fn chat_web_images_are_real_photos() {
        // Real web photos in plain chat (Steven's word, 2026-11-21):
        // the client strips [[WEBIMAGE: ...]] markers, fetches real
        // photos from POST /api/web-image, and renders them under the
        // reply. The old [[IMAGE: ...]] marker is still stripped and
        // never turned into an image.
        assert!(INDEX_HTML.contains("function extractWebImageMarker"));
        assert!(INDEX_HTML.contains("function stripWebImageMarkerLive"));
        assert!(INDEX_HTML.contains("function attachWebImages"));
        assert!(INDEX_HTML.contains("/api/web-image"));
        assert!(INDEX_HTML.contains("function extractImageMarker"));
        assert!(!INDEX_HTML.contains("function attachIllustration"));
    }

    #[test]
    fn chat_replies_do_not_auto_illustrate() {
        // Auto-illustration removed at Steven's word (2026-11-21): no
        // image is generated from a chat reply. A stray [[IMAGE: ...]]
        // marker is still stripped client-side (safety net), never
        // shown raw and never turned into an image.
        assert!(!INDEX_HTML.contains("function attachIllustration"));
        assert!(!INDEX_HTML.contains("attachIllustration("));
        assert!(INDEX_HTML.contains("function extractImageMarker"));
        assert!(INDEX_HTML.contains("function stripMarkerLive"));
    }

    #[test]
    fn user_messages_have_resend_button() {
        // Steven's word, 2026-11-21: every user message gets a Resend
        // button that re-sends that exact message for a fresh reply,
        // replacing the assistant reply directly after it.
        assert!(INDEX_HTML.contains("resendbtn"));
        assert!(INDEX_HTML.contains("function resendMessage"));
        assert!(INDEX_HTML.contains("function runPlainChat"));
        assert!(INDEX_HTML.contains("function updateResendButtons"));
    }

    #[test]
    fn skills_auto_activate_when_needed() {
        // Steven's word, 2026-11-23: a skill should activate
        // automatically when the chat message clearly needs it,
        // instead of only when he picks it by hand.
        assert!(INDEX_HTML.contains("function autoSkillForMessage"));
        assert!(INDEX_HTML.contains("function effectiveSkillId"));
        assert!(INDEX_HTML.contains("AUTO_TRIGGERS"));
        assert!(INDEX_HTML.contains("Auto-activated skill: "));
    }

    #[test]
    fn topic_pdf_writes_content_first() {
        // Steven, 2026-11-23: "make a PDF of African countries"
        // made an empty PDF holding only the topic words, and
        // "make a PDF of Uganda anthem" PDF'd the previous
        // download message instead. A topic PDF must use an
        // earlier reply only when it is about that topic;
        // otherwise it asks the model to write about the topic
        // first, then PDFs that reply.
        assert!(INDEX_HTML.contains("Writing about "));
        assert!(INDEX_HTML.contains("Write about: "));
        assert!(INDEX_HTML.contains("prevAbout"));
    }
}

const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Mawaxi AI</title>
<script src="https://js.puter.com/v2/"></script>
<style>
:root{--bg:#282c34;--panel:#111;--panel2:#1e2228;--border:#355a66;--green:#50fa7b;--green-hover:#3ddc6a;--text:#9cdef2;--muted:#828997;--user:#1DB954;--userfg:#fff;--ai:#111;--code:#1e2228;--side:#111;--hover:#343a45}
*{box-sizing:border-box}
html,body{height:100%}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);margin:0}
#app{display:flex;height:100dvh}
#side{width:260px;min-width:220px;background:var(--side);border-right:1px solid var(--border);display:flex;flex-direction:column;z-index:60}
#side .sidehead{display:flex;align-items:center;justify-content:space-between}
#side h2{font-size:14px;margin:14px 14px 8px;color:var(--muted);font-weight:600}
#newchat{margin:0 14px 10px;padding:10px;border:1px solid var(--border);background:var(--panel2);color:var(--text);border-radius:10px;cursor:pointer;font-size:14px}
#chatSearch{margin:0 14px 8px;padding:9px 12px;border:1px solid var(--border);background:var(--panel2);color:var(--text);border-radius:10px;font-size:14px;outline:none}
#chatSearch::placeholder{color:var(--muted)}
#newchat:hover{background:var(--hover)}
#chatlist{flex:1;overflow-y:auto;padding:0 8px 12px}
.chatitem{display:flex;align-items:center;gap:6px;padding:10px;border-radius:10px;cursor:pointer}
.chatitem:hover{background:var(--panel2)}
.chatitem.active{background:var(--hover)}
.chatitem .tx{flex:1;min-width:0;display:flex;flex-direction:column;gap:2px}
.chatitem .t{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:14px}
.chatitem .d{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chatitem .del{background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;padding:0 4px}
#overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:50;display:none}
#overlay.show{display:block}
#closeSide{display:none}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
header{display:flex;align-items:center;gap:8px;padding:10px 14px}
header .spacer{width:40px}
header .title{flex:1;min-width:0;text-align:center;background:none;border:none;color:var(--text);font-size:16px;font-weight:600;cursor:pointer;padding:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
header .title .chev{color:var(--green);font-size:14px}
#hamburger{background:none;border:none;cursor:pointer;padding:6px;display:flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:8px}
#hamburger:hover{background:var(--panel2)}
#logWrap{flex:1;position:relative;overflow:hidden;display:flex;flex-direction:column}
#log{flex:1;overflow-y:auto;overflow-x:hidden;padding:8px 16px 16px;display:flex;flex-direction:column;gap:14px;transform:scale(var(--chat-zoom,1));transform-origin:top left;width:calc(100%/var(--chat-zoom,1))}
.msg{max-width:min(720px,75%);min-width:0;font-size:var(--chat-font-size,16px)}
.msg .body{border-radius:18px;padding:10px 14px;word-wrap:break-word;overflow-wrap:break-word;word-break:break-word;line-height:1.55;min-width:0}
.msg.user{align-self:flex-end}
.msg.user .body{background:var(--user);color:var(--userfg);border-radius:18px 18px 0 18px;border:1px solid var(--border)}
.msg.ai{align-self:flex-start}
.msg.ai .body{background:var(--ai);border:1px solid var(--border);border-radius:18px 18px 18px 0}
.msg p{margin:6px 0}
.msg pre{position:relative;background:var(--code);border:1px solid var(--border);border-radius:10px;padding:26px 10px 10px;overflow-x:auto;max-width:100%}
.msg code{background:var(--code);border:1px solid var(--border);border-radius:4px;padding:1px 4px;font-size:13px}
.msg pre code{background:none;border:none;padding:0}
.msg img,.msg video,.msg iframe{max-width:100%;height:auto}
.msg table{display:block;max-width:100%;overflow-x:auto}
.copybtn{position:absolute;top:6px;right:6px;background:var(--panel2);color:var(--muted);border:1px solid var(--border);border-radius:6px;padding:2px 8px;font-size:11px;line-height:1.4;cursor:pointer;z-index:1}
.copybtn:hover{color:var(--text);border-color:var(--green)}
.resendbtn{background:rgba(0,0,0,.22);color:#fff;border:1px solid rgba(255,255,255,.4);border-radius:6px;padding:2px 8px;font-size:11px;line-height:1.4;cursor:pointer;margin-top:6px}
.resendbtn:hover{border-color:#fff}
.resendbtn:disabled{opacity:.45;cursor:default}
.msg a{color:var(--green)}
.msg ul{margin:6px 0;padding-left:20px}
.msg h1,.msg h2,.msg h3{margin:10px 0 6px;color:var(--text)}
#empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;text-align:center;padding:20px;pointer-events:none}
#empty .logo{margin-bottom:2px}
#empty .logo img{width:96px;height:96px;border-radius:22px;display:block;margin:0 auto}
#empty .wordmark{font-size:44px;font-weight:800;font-style:italic;color:var(--green);letter-spacing:.5px;line-height:1.1}
#empty .tagline{color:var(--muted);font-size:17px;margin-top:2px}
#personaBtn{pointer-events:auto;display:inline-flex;align-items:center;gap:6px;margin-top:14px;padding:7px 16px;border:1px solid var(--border);border-radius:999px;background:transparent;color:var(--muted);font-size:14px;cursor:pointer}
#personaBtn:hover{border-color:var(--green);color:var(--text)}
#composer{margin:0 12px 14px;background:var(--panel2);border:1px solid var(--border);border-radius:22px;padding:10px 12px 8px}
.input-row{display:flex;align-items:center;gap:8px}
#q{flex:1;min-width:0;background:transparent;border:none;outline:none;color:var(--text);font-size:16px;padding:6px 4px;font-family:inherit}
#q::placeholder{color:var(--muted)}
#model{background:transparent;border:none;color:var(--muted);font-size:13px;max-width:150px;text-overflow:ellipsis;outline:none;cursor:pointer}
#model option{background:var(--panel2);color:var(--text)}
#persona{background:transparent;border:none;color:var(--muted);font-size:13px;max-width:130px;text-overflow:ellipsis;outline:none;cursor:pointer}
#persona option{background:var(--panel2);color:var(--text)}
.action-row{display:flex;align-items:center;gap:6px;margin-top:8px;flex-wrap:wrap}
.iconbtn{background:none;border:none;cursor:pointer;padding:8px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:var(--muted)}
.iconbtn:hover{background:var(--hover);color:var(--text)}
#modeToggle{margin:0 auto;display:flex;background:var(--panel);border:1px solid var(--border);border-radius:999px;padding:3px}
#modeToggle button{background:none;border:none;color:var(--muted);font-size:14px;padding:6px 16px;border-radius:999px;cursor:pointer}
#modeToggle button.active{background:var(--hover);color:var(--text)}
#micBtn{background:var(--green);border:none;width:44px;height:44px;border-radius:50%;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0}
#micBtn:hover{background:var(--green-hover)}
#sendBtn{background:var(--panel);border:1px solid var(--border);width:40px;height:40px;border-radius:50%;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0}
#sendBtn:hover{border-color:var(--green)}
#toast{position:fixed;left:50%;bottom:110px;transform:translateX(-50%);background:var(--panel2);color:var(--text);padding:10px 18px;border-radius:999px;font-size:14px;z-index:100;display:none;box-shadow:0 4px 16px rgba(0,0,0,.5);white-space:nowrap;max-width:92vw;overflow:hidden;text-overflow:ellipsis}
.sidebrand{font-size:20px;font-weight:800;color:var(--green);padding:16px 16px 2px;letter-spacing:.3px}
#sideFoot{display:flex;flex-direction:column;gap:2px;padding:10px 8px;border-top:1px solid var(--border)}
.sidebtn{flex:1 1 40%;padding:9px 4px;border:1px solid var(--border);background:var(--panel2);color:var(--text);border-radius:10px;cursor:pointer;font-size:13px}
#sideBtns{display:flex;flex-wrap:wrap;gap:6px;padding:0 14px 10px}
#featureView{position:fixed;inset:0;background:var(--bg);z-index:70;display:none;flex-direction:column}
#featureView.show{display:flex}
.fv-head{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.fv-title{font-size:18px;font-weight:700}
.fv-body{flex:1;min-height:0;overflow-y:auto;-webkit-overflow-scrolling:touch;padding:14px 16px 24px;display:flex;flex-direction:column;gap:8px}
.rsActCard{position:relative;background:var(--panel);border:1px solid var(--border);border-radius:16px;overflow:hidden;margin-bottom:10px}
.rsActHead{display:flex;align-items:center;gap:8px;padding:12px 14px;cursor:pointer}
.rsActTitle{flex:1;font-weight:600;font-size:15px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rsActTime{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap}
.rsActMenuBtn{background:transparent;border:none;color:var(--muted);font-size:18px;cursor:pointer;padding:2px 6px;line-height:1}
.rsActMenu{position:absolute;right:10px;top:44px;background:var(--panel);border:1px solid var(--border);border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.4);z-index:5;min-width:170px;overflow:hidden}
.rsActMenu button{display:block;width:100%;text-align:left;background:transparent;border:none;color:var(--text);padding:10px 14px;font-size:14px;cursor:pointer}
.rsActMenu button:hover{background:var(--surface)}
.rsActMenu button.danger{color:#ff7b7b}
.rsActPhase{padding:0 14px 8px;color:var(--muted);font-size:13px}
.rsActPhase b{color:var(--text);font-weight:600}
.rsActVis{position:relative;margin:0 14px;border-radius:12px;overflow:hidden;background:#0b1220;border:1px solid var(--border);min-height:150px}
.rsActVisGrid{position:absolute;inset:0;background-image:linear-gradient(rgba(120,180,255,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(120,180,255,.07) 1px,transparent 1px);background-size:22px 22px}
.rsActScan{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#4ade80,transparent);opacity:.7;animation:rsActScan 3.2s ease-in-out infinite alternate}
@keyframes rsActScan{0%{top:8%}100%{top:88%}}
.rsActVisInner{position:relative;padding:14px;text-align:center;color:#dbeafe}
.rsActRound{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#93c5fd}
.rsActBig{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:15px;color:#4ade80;margin-top:10px;letter-spacing:.1em;min-height:20px;word-break:break-all}
.rsActMeta{display:flex;justify-content:center;gap:18px;margin-top:8px;font-size:13px;color:#93c5fd;font-variant-numeric:tabular-nums}
.rsActBar{height:4px;background:var(--surface);margin:10px 14px 0;border-radius:99px;overflow:hidden}
.rsActFill{height:100%;background:var(--green);border-radius:99px;transition:width .4s ease}
.rsActLog{padding:10px 14px 12px;font-size:13px;line-height:1.5}
.rsActRow{display:flex;gap:8px;padding:3px 0;align-items:baseline}
.rsActRowTag{color:#4ade80;font-size:11px;white-space:nowrap;font-weight:600}
.rsActRowText{flex:1;color:var(--text);word-break:break-word}
.rsActRowText .sub{color:var(--muted);font-size:12px}
.rsActRowTime{color:var(--muted);font-size:11px;white-space:nowrap;font-variant-numeric:tabular-nums}
.rsActBtns{display:flex;gap:8px;padding:0 14px 14px}
.rsActReport{padding:0 14px 14px}
.rsActEmpty{color:var(--muted);font-size:13px;padding:8px 2px}
.fv-sec{font-weight:700;font-size:15px;margin-top:10px}
.fv-sec:first-child{margin-top:0}
.sidebtn.active{border-color:var(--green);color:var(--green)}
.sidebtn:hover{background:var(--hover)}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:80;display:none;align-items:center;justify-content:center;padding:16px}
#modal.show{display:flex}
.modal-panel{background:var(--panel);border:1px solid var(--border);border-radius:16px;width:100%;max-width:520px;max-height:85dvh;display:flex;flex-direction:column;overflow:hidden}
.modal-head{display:flex;align-items:center;justify-content:space-between;padding:12px 12px 0}
.modal-tabs{display:flex;gap:6px}
.mtab{background:none;border:none;color:var(--muted);font-size:15px;font-weight:600;padding:8px 12px;border-radius:999px;cursor:pointer}
.mtab.active{background:var(--hover);color:var(--text)}
.modal-body{padding:12px 16px 16px;overflow-y:auto}
.tabpage{display:flex;flex-direction:column;gap:8px}
.hint{color:var(--muted);font-size:13px;margin:0 0 4px;line-height:1.5}
.fld{background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:10px;padding:10px 12px;font-size:15px;outline:none;font-family:inherit;width:100%}
.fld:focus{border-color:var(--green)}
textarea.fld{resize:vertical}
.btn{background:var(--green);border:none;color:#fff;border-radius:10px;padding:11px;font-size:15px;font-weight:600;cursor:pointer}
.btn:hover{background:var(--green-hover)}
.rowflex{display:flex;gap:8px}
.rowflex .fld{flex:1}
.listrow{display:flex;align-items:center;gap:8px;background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin-top:8px}
.listrow .info{flex:1;min-width:0}
.listrow .nm{font-weight:600;font-size:14px}
.listrow .sub{color:var(--muted);font-size:13px;word-break:break-word}
.listrow .sub2{color:var(--muted);font-size:13px;line-height:1.45}
.mini{background:none;border:1px solid var(--border);color:var(--text);border-radius:999px;padding:6px 12px;font-size:13px;cursor:pointer;flex-shrink:0}
.zoomctl{padding:6px 9px;min-width:30px;text-align:center}
#zoomLabel{min-width:52px}
.mini:hover{border-color:var(--green)}
.mini.on{background:var(--green);border-color:var(--green);color:#fff}
.mini.del{border:none;color:var(--muted);padding:6px 8px;font-size:16px}
#skillChip{display:flex;align-items:center;gap:6px;margin-bottom:8px}
#skillChip span{background:var(--panel2);border:1px solid var(--green);color:var(--green);border-radius:999px;padding:5px 12px;font-size:13px}
#skillChip button{background:none;border:none;color:var(--muted);cursor:pointer;font-size:14px;padding:4px}
#attachPreview{display:flex;align-items:center;gap:8px;margin-bottom:8px;background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:6px 10px}
#attachPreview img{width:44px;height:44px;object-fit:cover;border-radius:8px}
#attachPreview span{flex:1;font-size:13px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#attachPreview button{background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;padding:4px}
#micBtn.listening{background:#e5484d;animation:pulse 1s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(229,72,77,.5)}70%{box-shadow:0 0 0 10px rgba(229,72,77,0)}100%{box-shadow:0 0 0 0 rgba(229,72,77,0)}}
.msg .imgthumb{max-width:100%;border-radius:12px;display:block;margin-bottom:8px}
#stopAgentBtn{background:#e5484d;border:none;color:#fff;border-radius:999px;padding:7px 14px;font-size:13px;font-weight:600;cursor:pointer;flex-shrink:0}
.msg.term{align-self:flex-start;max-width:min(720px,75%)}
.msg.term .body{background:#0d0d0d;color:#e8e8e8;border:1px solid var(--border);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-word;box-shadow:0 2px 12px rgba(0,0,0,.45)}
.msg.term .cmd{color:var(--green);font-weight:600}
.msg.term .err{color:#ff7b7b}
.msg.term .note{color:var(--muted)}
.term .add{color:#50fa7b}
.term .del{color:#ff7b7b}
.term .ctx{color:#8a8a8a}
.asteps{margin:8px 0;max-width:min(720px,94%)}
.aprog{display:flex;align-items:center;gap:8px;margin:2px 0 8px;font-size:12px;color:var(--muted)}
.aprog .lbl{white-space:nowrap}
.aprog .bar{flex:1;height:6px;background:var(--panel2);border-radius:3px;overflow:hidden;display:block}
.aprog .fill{display:block;height:100%;background:var(--green);width:0%;transition:width .3s}
.astep{position:relative;padding-left:24px;margin:3px 0}
.astep::before{content:'';position:absolute;left:7px;top:22px;bottom:-4px;width:1px;background:var(--border)}
.astep:last-child::before{display:none}
.astep-dot{position:absolute;left:1px;top:7px;width:13px;height:13px;border-radius:50%;border:2px solid var(--muted);background:var(--panel);box-sizing:border-box}
.astep.done .astep-dot{border-color:var(--green)}
.astep.err .astep-dot{border-color:#ff7b7b}
.astep-head{display:flex;align-items:center;gap:7px;cursor:pointer;padding:2px 0;user-select:none}
.astep-ico{color:var(--muted);font-weight:700;width:14px;text-align:center}
.astep.done .astep-ico{color:var(--green)}
.astep.err .astep-ico{color:#ff7b7b}
.astep-tico{color:var(--muted)}
.astep-tool{color:var(--green);font-weight:700;letter-spacing:.02em}
.astep.err .astep-tool{color:#ff7b7b}
.astep-status{color:var(--muted);font-size:13px}
.astep-chev{margin-left:auto;color:var(--muted);transition:transform .15s}
.astep.open .astep-chev{transform:rotate(90deg)}
.astep-body{display:none;margin:5px 0 8px 0;padding:9px 11px;background:var(--panel);border:1px solid var(--border);border-radius:10px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;line-height:1.55;white-space:pre-wrap;word-break:break-word;color:var(--text)}
.astep.open .astep-body{display:block}
.astep-cmd{background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:8px 12px;margin:4px 0}
.astep-body .err{color:#ff7b7b}
.astep-body .add{color:#50fa7b}
.astep-body .del{color:#ff7b7b}
.astep-body .ctx{color:#8a8a8a}
.astep-out{margin-top:8px;background:color-mix(in srgb, var(--green) 5%, transparent);border:1px solid color-mix(in srgb, var(--green) 20%, transparent);border-radius:8px;overflow:hidden}
.astep-out summary{color:var(--green);background:color-mix(in srgb, var(--green) 10%, transparent);border-bottom:1px solid color-mix(in srgb, var(--green) 15%, transparent);cursor:pointer;font-size:12.5px;user-select:none;padding:6px 10px;font-weight:600;display:flex;align-items:center;justify-content:space-between;gap:8px;list-style:none}
.astep-out summary::-webkit-details-marker{display:none}
.astep-out summary::after{content:'\25BC';color:var(--green);font-size:.9em;transition:transform .3s ease}
.astep-out[open]>summary::after{transform:rotate(180deg)}
.astep-out summary:hover{background:color-mix(in srgb, var(--green) 15%, transparent)}
.astep-out pre{margin:0;padding:8px 10px;overflow-x:auto;font-size:12px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
.astep-diff{background:color-mix(in srgb, var(--text) 4%, transparent);border-color:color-mix(in srgb, var(--text) 18%, transparent)}
.astep-diff summary{color:var(--text);background:color-mix(in srgb, var(--text) 7%, transparent);border-bottom-color:color-mix(in srgb, var(--text) 12%, transparent);justify-content:flex-start}
.astep-diff summary::after{color:var(--text)}
.astep-diff summary:hover{background:color-mix(in srgb, var(--text) 12%, transparent)}
.astep-diff .diff-file{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.astep-diff .diff-stats{margin-left:auto;font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:600;flex-shrink:0;display:flex;gap:8px}
.astep-diff .diff-stat-add{color:#50fa7b}
.astep-diff .diff-stat-del{color:#ff7b7b}
.astep-diff .diff-stat-new{color:var(--green);opacity:.85}
.diff-pre{margin:0;padding:8px 10px;overflow-x:auto;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.45}
.diff-pre span{display:block;white-space:pre}
.diff-pre .diff-add{background:color-mix(in srgb, #50fa7b 22%, transparent)}
.diff-pre .diff-del{background:color-mix(in srgb, #ff7b7b 22%, transparent)}
.diff-pre .diff-hunk{color:var(--green);opacity:.85}
.diff-pre .diff-meta{opacity:.55}
.diff-pre .diff-ctx{opacity:.8}
.a2ui-surface{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:14px;margin:6px 0}
.a2ui-title{font-weight:700;font-size:16px;margin:0 0 8px}
.a2ui-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:10px 12px;margin:8px 0}
.a2ui-card b{display:block;font-size:15px}
.a2ui-card .sub{color:var(--muted);font-size:13px;margin-top:2px}
.a2ui-card .badge{display:inline-block;background:var(--green);color:#fff;border-radius:999px;padding:1px 8px;font-size:12px;font-weight:700;margin-top:6px}
.a2ui-text{margin:6px 0}
.a2ui-text.cap{color:var(--muted);font-size:13px}
.a2ui-text.head{font-weight:700;font-size:16px}
.a2ui-btn{display:block;width:100%;background:var(--green);color:#fff;border:none;border-radius:999px;padding:10px 14px;font-size:15px;font-weight:700;cursor:pointer;margin:8px 0}
.a2ui-btn:active{opacity:.85}
.a2ui-chips{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.a2ui-chip{background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:999px;padding:7px 14px;font-size:14px;cursor:pointer}
.a2ui-chip:active{border-color:var(--green)}
.a2ui-divider{border:none;border-top:1px solid var(--border);margin:10px 0}
#wsPanel{margin:0 12px 10px;background:var(--panel);border:1px solid var(--border);border-radius:16px;overflow:hidden;flex-shrink:0}
#wsPanel .wsHead{display:flex;align-items:center;gap:8px;padding:8px 12px;border-bottom:1px solid var(--border);font-weight:700;font-size:14px}
#wsPanel .wsHead .wsPathLabel{flex:1;padding:0;font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:400}
#wsList{max-height:180px;overflow-y:auto;padding:6px 12px;font-size:13px}
.wsrow{display:flex;justify-content:space-between;gap:8px;padding:4px 0;white-space:nowrap;overflow:hidden}
.wsrow .nm{overflow:hidden;text-overflow:ellipsis}
.wsrow .sz{color:var(--muted);flex-shrink:0}
/* Themes, Browser, Calendar, PDF, Data Saver */
.themegrid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.themebtn{display:flex;align-items:center;gap:8px;background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:10px;padding:10px 12px;font-size:14px;cursor:pointer;text-align:left}
.themebtn.active{border-color:var(--green);color:var(--green)}
.themebtn .dots{display:flex;gap:3px;flex-shrink:0}
.themebtn .dot{width:14px;height:14px;border-radius:50%;border:1px solid var(--border)}
#browserBar{display:flex;gap:8px;flex-shrink:0}
#browserBar .fld{flex:1}
#browserView{flex:1;min-height:0;display:flex;flex-direction:column;gap:8px}
#browserFrame{flex:1;min-height:0;width:100%;border:1px solid var(--border);border-radius:12px;background:#fff}
#browserReader{flex:1;min-height:0;overflow-y:auto;background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px;line-height:1.6;white-space:pre-wrap;word-break:break-word}
#browserReader h3{margin:0 0 8px}
.browserlinks{display:flex;gap:12px;align-items:center;flex-wrap:wrap;flex-shrink:0}
.browserlinks a{color:var(--green);font-size:13px;cursor:pointer}
.browserlinks .bstatus{color:var(--muted);font-size:13px;flex:1}
.calhead{display:flex;align-items:center;gap:8px;flex-shrink:0}
.calhead .caltitle{flex:1;text-align:center;font-weight:700;font-size:16px}
.calgrid{display:grid;grid-template-columns:repeat(7,1fr);gap:4px}
.calday{border:1px solid var(--border);border-radius:10px;min-height:56px;padding:6px 6px 4px;font-size:13px;cursor:pointer;background:var(--panel)}
.calday.other{opacity:.35}
.calday.today{border-color:var(--green);box-shadow:inset 0 0 0 1px var(--green)}
.calday.sel{background:var(--panel2)}
.calday .dn{font-weight:700}
.calday .ev{font-size:11px;color:var(--green);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
.calweek{font-size:12px;color:var(--muted);text-align:center;padding:2px 0}
.tapload{display:flex;flex-direction:column;align-items:center;gap:8px;background:var(--panel2);border:1px dashed var(--border);border-radius:12px;padding:22px 12px;color:var(--muted);font-size:13px;text-align:center}
.tapload button{background:var(--green);border:none;color:#fff;border-radius:999px;padding:8px 18px;font-size:14px;font-weight:600;cursor:pointer}
.pdfview{width:100%;min-height:60vh;border:1px solid var(--border);border-radius:12px;background:#fff}
#hamburger{color:var(--text)}
#sendBtn{color:var(--text)}
#empty .logo{color:var(--green)}
@media print{
  #side,#composer,header,#toast,#modal,#featureView,#wsPanel,#stopAgentBtn{display:none!important}
  #main{width:100%}
  #log{overflow:visible;transform:none;width:auto}
  body{background:#fff;color:#000}
  .msg{max-width:none}
  .msg .body{border:1px solid #ccc}
}
@media (min-width:901px){
  #side.collapsed{display:none}
}
@media (max-width:900px){
  #side{position:fixed;left:0;top:0;bottom:0;transform:translateX(-105%);transition:transform .25s ease}
  #side.open{transform:translateX(0)}
  #closeSide{display:flex}
  #empty .wordmark{font-size:38px}
}
@media (max-width:600px){
  header{gap:4px;padding:10px 8px}
  .zoomctl{padding:6px 7px}
  #zoomLabel{min-width:46px}
  #log{padding:8px 10px 14px}
  .msg,.msg.term{max-width:94%}
  .msg .body{padding:9px 12px}
  #composer{margin:0 8px 10px}
}
</style>
</head>
<body>
<div id="app">
  <div id="overlay"></div>
  <aside id="side">
    <div class="sidebrand">Mawaxi AI</div>
    <div class="sidehead">
      <h2>Chats</h2>
      <button id="closeSide" class="iconbtn" title="Close">&#10005;</button>
    </div>
    <button id="newchat">+ New chat</button>
    <input id="chatSearch" placeholder="Search chats..." autocomplete="off">
    <div id="chatlist"></div>
    <div id="sideBtns">
      <button id="openWorkspace" class="sidebtn">Workspace</button>
      <button id="openMemory" class="sidebtn">Memory</button>
      <button id="openSkills" class="sidebtn">Skills</button>
      <button id="openNotes" class="sidebtn">Notes</button>
      <button id="openResearch" class="sidebtn">Deep Research</button>
      <button id="openCookbook" class="sidebtn">Cookbook</button>
      <button id="openLibrary" class="sidebtn">Library</button>
      <button id="readAloudSide" class="sidebtn">Read aloud</button>
      <button id="openBrowser" class="sidebtn">Browser</button>
      <button id="openCalendar" class="sidebtn">Calendar</button>
      <button id="openPdf" class="sidebtn">PDF</button>
    </div>
    <div id="sideFoot">
      <button id="openSettings" class="sidebtn">Settings</button>
    </div>
  </aside>
  <div id="main">
    <header>
      <span class="spacer"></span>
      <button class="title" id="titleBtn">Mawaxi AI <span class="chev">&#9662;</span></button>
      <button id="chatPdfBtn" class="mini" type="button" title="Download this chat as PDF">PDF</button>
      <button id="zoomOut" class="mini zoomctl" type="button" title="Zoom out">−</button>
      <button id="zoomLabel" class="mini zoomctl" type="button" title="Tap to reset zoom to 100%">100%</button>
      <button id="zoomIn" class="mini zoomctl" type="button" title="Zoom in">+</button>
      <button id="hamburger" title="Chats" aria-label="Chats">
        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18M3 12h18M3 18h18"/></svg>
      </button>
    </header>
    <div id="logWrap">
      <div id="log"></div>
      <div id="empty">
        <div class="logo">
          <img src="data:image/webp;base64,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czpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPjxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiIHhtbG5zOmlwdGNFeHQ9Imh0dHA6Ly9pcHRjLm9yZy9zdGQvSXB0YzR4bXBFeHQvMjAwOC0wMi0yOS8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgaXB0Y0V4dDpEaWdpdGFsU291cmNlVHlwZT0iaHR0cDovL2N2LmlwdGMub3JnL25ld3Njb2Rlcy9kaWdpdGFsc291cmNldHlwZS90cmFpbmVkQWxnb3JpdGhtaWNNZWRpYSI+PGRjOnB1Ymxpc2hlcj48cmRmOkJhZz48cmRmOmxpPk1ldGE8L3JkZjpsaT48L3JkZjpCYWc+PC9kYzpwdWJsaXNoZXI+PC9yZGY6RGVzY3JpcHRpb24+PC9yZGY6UkRGPjwveDp4bXBtZXRhPjw/eHBhY2tldCBlbmQ9InciPz4AQzJQQSKaAAAAAJoianVtYgAAAB5qdW1kYzJwYQARABCAAACqADibcQNjMnBhAAAASYFqdW1iAAAAR2p1bWRjMm1hABEAEIAAAKoAOJtxA3VybjpjMnBhOjNmNTU4YjUwLWQ5M2EtNjUyYi00OTc0LTQ3MDlhYmM1NzI2ZgAAAENXanVtYgAAAChqdW1kYzJjcwARABCAAACqADibcQNjMnBhLnNpZ25hdHVyZQAAAEMnY2JvctKEWQKtogEmGCFZAqUwggKhMIICRqADAgECAgYgJhACAgIwCgYIKoZIzj0EAwIwgZgxCzAJBgNVBAYTAlVTMRMwEQYDVQQIDApDYWxpZm9ybmlhMRMwEQYDVQQHDApNZW5sbyBQYXJrMQ0wCwYDVQQKDARNZXRhMScwJQYDVQQLDB5NZXRhIE1lZGlhIFByb2Nlc3NpbmcgU2VydmljZXMxJzAlBgNVBAMMHk1ldGEgTWVkaWEgUHJvY2Vzc2luZyBTZXJ2aWNlczAeFw0yNjEwMDIyMjAxNTNaFw0yNzEwMDIyMjAxNTNaMIGYMQswCQYDVQQGEwJVUzETMBEGA1UECAwKQ2FsaWZvcm5pYTETMBEGA1UEBwwKTWVubG8gUGFyazENMAsGA1UECgwETWV0YTEnMCUGA1UECwweTWV0YSBNZWRpYSBQcm9jZXNzaW5nIFNlcnZpY2VzMScwJQYDVQQDDB5NZXRhIE1lZGlhIFByb2Nlc3NpbmcgU2VydmljZXMwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAATT0QLJK894mm6H+yBswxMB3P7utU9leKTnWwR1//CdaOS4sJKGM83Z9zo7PItb5qk0ombrbg+pZ4Gee8HEk7HIo3oweDAMBgNVHRMBAf8EAjAAMA4GA1UdDwEB/wQEAwIHgDAYBgNVHSUBAf8EDjAMBgorBgEEAYPoXgIBMB0GA1UdDgQWBBRPzzpQDTp8ztrWhXfjxmyuBR/BAzAfBgNVHSMEGDAWgBSyDvV9GVzdhpXvpjLAJsyBCFZO+zAKBggqhkjOPQQDAgNJADBGAiEAjmvfhqK2osAvYwUQvj3elI8Ad1LcV82HIinddtzev8ICIQDo1FCRQbVngzpJxC30p9LqGUc/GRvGFjCV177Fg5hduaJjcGFkWUAaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAZHBhZDJCAAD2WED/P71UHxNK9Wy90gU5U2/rlTaoZJRJYdt93XyYYxVSeFw+LTH5R8KagQSGHTv5nFEjtXy2igRyv8lOAACVVzyXAAACKWp1bWIAAAAnanVtZGMyY2wAEQAQgAAAqgA4m3EDYzJwYS5jbGFpbS52MgAAAAH6Y2JvcqZqaW5zdGFuY2VJRHgkM2M5Nzk5MDItYzVkOS01YzA0LTMyNzQtNDM4NWJmN2QyMGI3dGNsYWltX2dlbmVyYXRvcl9pbmZvomRuYW1ldU1ldGEgTWVkaWEgRm91bmRhdGlvbmtzcGVjVmVyc2lvbmUyLjQuMHJjcmVhdGVkX2Fzc2VydGlvbnOComN1cmx4KnNlbGYjanVtYmY9YzJwYS5hc3NlcnRpb25zL2MycGEuYWN0aW9ucy52MmRoYXNoWCCNWhnAjI8sokCq7diKupaMC94AGFsI6MKM7u4d9Td5e6JjdXJseClzZWxmI2p1bWJmPWMycGEuYXNzZXJ0aW9ucy9jMnBhLmhhc2guZGF0YWRoYXNoWCCoXUHV0m2p/3tbVx8+konlpx8QVfas7zljS8w1bHYT4nNnYXRoZXJlZF9hc3NlcnRpb25zgaJjdXJseEdzZWxmI2p1bWJmPWMycGEuYXNzZXJ0aW9ucy9jb20ubWV0YS5nZW5lcmF0ZWQtbWVkaWEtcHJvdmVuYW5jZS5tZXRhZGF0YWRoYXNoWCC/We9p+LpDA0m+m9bCxecMXD1RDqtJY70iyfZmlJgQkGlzaWduYXR1cmV4GXNlbGYjanVtYmY9YzJwYS5zaWduYXR1cmVjYWxnZnNoYTI1NgAAA7JqdW1iAAAAKWp1bWRjMmFzABEAEIAAAKoAOJtxA2MycGEuYXNzZXJ0aW9ucwAAAAGuanVtYgAAAF5qdW1kanNvbgARABCAAACqADibcRNjb20ubWV0YS5nZW5lcmF0ZWQtbWVkaWEtcHJvdmVuYW5jZS5tZXRhZGF0YQAAAAAYYzJzaEarG1AhzVJyc2W2D4PMJ7EAAAFIanNvbnsiQGNvbnRleHQiOnsiQHZvY2FiIjoidGFnOm1ldGEuY29tLDIwMjY6YzJwYTpnZW5lcmF0ZWQtbWVkaWEtcHJvdmVuYW5jZToiLCJtZXRhIjoidGFnOm1ldGEuY29tLDIwMjY6YzJwYTpnZW5lcmF0ZWQtbWVkaWEtcHJvdmVuYW5jZToifSwiQHR5cGUiOiJtZXRhOkdlbmVyYXRlZE1lZGlhUHJvdmVuYW5jZSIsIm1ldGE6cHJvdmVuYW5jZVVybiI6InVybjp1dWlkOmZkNzZiNWM4LWM1Y2ItNGZlZi05MTg5LTYzYzQ0ZGRhOGM3YiIsIm1ldGE6c2NoZW1hIjoiY29tLm1ldGEuZ2VuZXJhdGVkLW1lZGlhLXByb3ZlbmFuY2UiLCJtZXRhOnNjaGVtYVZlcnNpb24iOjF9AAAAtGp1bWIAAABAanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5oYXNoLmRhdGEAAAAAGGMyc2h6+7B7wWS8lrtuQ7DjIhWaAAAAbGNib3KkamV4Y2x1c2lvbnOBomVzdGFydBoAAX3aZmxlbmd0aBlJp2NhbGdmc2hhMjU2ZGhhc2hYIDVGfHNp8q+q0OZeAmwWOblzp2vFRbhx8bNFaF4OWBILY3BhZEoAAAAAAAAAAAAAAAABH2p1bWIAAABBanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5hY3Rpb25zLnYyAAAAABhjMnNo+CIOxDXYx09C18PjZ73BKwAAANZjYm9yomdhY3Rpb25zgaRmYWN0aW9ubGMycGEuY3JlYXRlZHFkaWdpdGFsU291cmNlVHlwZXhGaHR0cDovL2N2LmlwdGMub3JnL25ld3Njb2Rlcy9kaWdpdGFsc291cmNldHlwZS90cmFpbmVkQWxnb3JpdGhtaWNNZWRpYW1zb2Z0d2FyZUFnZW50omRuYW1lZ01ldGEgQUlndmVyc2lvbmExZHdoZW7AdDIwMjYtMTEtMjVUMTE6MTQ6MDBacmFsbEFjdGlvbnNJbmNsdWRlZPUAAFB7anVtYgAAAEdqdW1kYzJtYQARABCAAACqADibcQN1cm46YzJwYTphYjhlOTY2OS0zMWUxLTI1ZTYtMWQ5Mi00YjdjNWIzNTUyYWIAAABDV2p1bWIAAAAoanVtZGMyY3MAEQAQgAAAqgA4m3EDYzJwYS5zaWduYXR1cmUAAABDJ2Nib3LShFkCraIBJhghWQKlMIICoTCCAkagAwIBAgIGICYQAgICMAoGCCqGSM49BAMCMIGYMQswCQYDVQQGEwJVUzETMBEGA1UECAwKQ2FsaWZvcm5pYTETMBEGA1UEBwwKTWVubG8gUGFyazENMAsGA1UECgwETWV0YTEnMCUGA1UECwweTWV0YSBNZWRpYSBQcm9jZXNzaW5nIFNlcnZpY2VzMScwJQYDVQQDDB5NZXRhIE1lZGlhIFByb2Nlc3NpbmcgU2VydmljZXMwHhcNMjYxMDAyMjIwMTUzWhcNMjcxMDAyMjIwMTUzWjCBmDELMAkGA1UEBhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExEzARBgNVBAcMCk1lbmxvIFBhcmsxDTALBgNVBAoMBE1ldGExJzAlBgNVBAsMHk1ldGEgTWVkaWEgUHJvY2Vzc2luZyBTZXJ2aWNlczEnMCUGA1UEAwweTWV0YSBNZWRpYSBQcm9jZXNzaW5nIFNlcnZpY2VzMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE09ECySvPeJpuh/sgbMMTAdz+7rVPZXik51sEdf/wnWjkuLCShjPN2fc6OzyLW+apNKJm624PqWeBnnvBxJOxyKN6MHgwDAYDVR0TAQH/BAIwADAOBgNVHQ8BAf8EBAMCB4AwGAYDVR0lAQH/BA4wDAYKKwYBBAGD6F4CATAdBgNVHQ4EFgQUT886UA06fM7a1oV348ZsrgUfwQMwHwYDVR0jBBgwFoAUsg71fRlc3YaV76YywCbMgQhWTvswCgYIKoZIzj0EAwIDSQAwRgIhAI5r34aitqLAL2MFEL493pSPAHdS3FfNhyIp3Xbc3r/CAiEA6NRQkUG1Z4M6ScQt9KfS6hlHPxkbxhYwlde+xYOYXbmiY3BhZFlAGgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGRwYWQyQgAA9lhArKYn+Esqkl943N9K9Nfgb/pOAZr5illIlzOdlRFzeqAXN5xf9Amh2qeHzZTFsBkwk21wL6sAEAHqo3MX6pzOfQAAAoRqdW1iAAAAJ2p1bWRjMmNsABEAEIAAAKoAOJtxA2MycGEuY2xhaW0udjIAAAACVWNib3Kmamluc3RhbmNlSUR4JDljNWI5MDM3LWFiNWUtOGNlZi02ZGI1LTQ2N2YxMmZjMTNkYXRjbGFpbV9nZW5lcmF0b3JfaW5mb6JkbmFtZXVNZXRhIE1lZGlhIEZvdW5kYXRpb25rc3BlY1ZlcnNpb25lMi40LjByY3JlYXRlZF9hc3NlcnRpb25zg6JjdXJseC1zZWxmI2p1bWJmPWMycGEuYXNzZXJ0aW9ucy9jMnBhLmluZ3JlZGllbnQudjNkaGFzaFggIxzglg+HQ3uYgNEyspJ37N+UpYoFLlAde5rBLQ9rd1miY3VybHgqc2VsZiNqdW1iZj1jMnBhLmFzc2VydGlvbnMvYzJwYS5hY3Rpb25zLnYyZGhhc2hYIL1+a6DZS2B0dF+hH7FyQbQwbBTCl654pP1AcI7SZERhomN1cmx4KXNlbGYjanVtYmY9YzJwYS5hc3NlcnRpb25zL2MycGEuaGFzaC5kYXRhZGhhc2hYIOUQQ8CKxF5fM78jZRPir+Lxrds/HZLicxgEJeH9bS/Qc2dhdGhlcmVkX2Fzc2VydGlvbnOBomN1cmx4R3NlbGYjanVtYmY9YzJwYS5hc3NlcnRpb25zL2NvbS5tZXRhLmdlbmVyYXRlZC1tZWRpYS1wcm92ZW5hbmNlLm1ldGFkYXRhZGhhc2hYIIrIWreOjMTMp6/j1Q/Bz6iWeqLDGJFp6FwPhxfVYga2aXNpZ25hdHVyZXgZc2VsZiNqdW1iZj1jMnBhLnNpZ25hdHVyZWNhbGdmc2hhMjU2AAAKUWp1bWIAAAApanVtZGMyYXMAEQAQgAAAqgA4m3EDYzJwYS5hc3NlcnRpb25zAAAAAa5qdW1iAAAAXmp1bWRqc29uABEAEIAAAKoAOJtxE2NvbS5tZXRhLmdlbmVyYXRlZC1tZWRpYS1wcm92ZW5hbmNlLm1ldGFkYXRhAAAAABhjMnNoxLQhsFUm8a+wHgJFuHD7jwAAAUhqc29ueyJAY29udGV4dCI6eyJAdm9jYWIiOiJ0YWc6bWV0YS5jb20sMjAyNjpjMnBhOmdlbmVyYXRlZC1tZWRpYS1wcm92ZW5hbmNlOiIsIm1ldGEiOiJ0YWc6bWV0YS5jb20sMjAyNjpjMnBhOmdlbmVyYXRlZC1tZWRpYS1wcm92ZW5hbmNlOiJ9LCJAdHlwZSI6Im1ldGE6R2VuZXJhdGVkTWVkaWFQcm92ZW5hbmNlIiwibWV0YTpwcm92ZW5hbmNlVXJuIjoidXJuOnV1aWQ6MWRjODI2NWMtNzdkMS00YWM0LWI4ZWEtM2RhY2NkN2ZhYjE4IiwibWV0YTpzY2hlbWEiOiJjb20ubWV0YS5nZW5lcmF0ZWQtbWVkaWEtcHJvdmVuYW5jZSIsIm1ldGE6c2NoZW1hVmVyc2lvbiI6MX0AAAC0anVtYgAAAEBqdW1kY2JvcgARABCAAACqADibcRNjMnBhLmhhc2guZGF0YQAAAAAYYzJzaPpmkrGL+stBxKe6uvFpBtQAAABsY2JvcqRqZXhjbHVzaW9uc4GiZXN0YXJ0GgABoH5mbGVuZ3RoGZoiY2FsZ2ZzaGEyNTZkaGFzaFggqsSSz8y6wmViZEonzkNsj8AjZRxGHfHGWWAX4xIR3VtjcGFkSgAAAAAAAAAAAAAAAAHZanVtYgAAAEFqdW1kY2JvcgARABCAAACqADibcRNjMnBhLmFjdGlvbnMudjIAAAAAGGMyc2hINZq0Rd1gAEZzmBd+SZpkAAABkGNib3KiZ2FjdGlvbnOCpGZhY3Rpb25rYzJwYS5vcGVuZWRtc29mdHdhcmVBZ2VudKJkbmFtZWdNZXRhIEFJZ3ZlcnNpb25hMWpwYXJhbWV0ZXJzoWtpbmdyZWRpZW50c4GiY3VybHgtc2VsZiNqdW1iZj1jMnBhLmFzc2VydGlvbnMvYzJwYS5pbmdyZWRpZW50LnYzZGhhc2hYICMc4JYPh0N7mIDRMrKSd+zflKWKBS5QHXuawS0Pa3dZZHdoZW7AdDIwMjYtMTEtMjVUMTE6MTQ6MDBao2ZhY3Rpb25rYzJwYS5lZGl0ZWRxZGlnaXRhbFNvdXJjZVR5cGV4U2h0dHA6Ly9jdi5pcHRjLm9yZy9uZXdzY29kZXMvZGlnaXRhbHNvdXJjZXR5cGUvY29tcG9zaXRlV2l0aFRyYWluZWRBbGdvcml0aG1pY01lZGlhbXNvZnR3YXJlQWdlbnSiZG5hbWVnTWV0YSBBSWd2ZXJzaW9uYTFyYWxsQWN0aW9uc0luY2x1ZGVk9QAABeVqdW1iAAAARGp1bWRjYm9yABEAEIAAAKoAOJtxE2MycGEuaW5ncmVkaWVudC52MwAAAAAYYzJzaN1Ob3Rh0szoRbZkaGT5HncAAAWZY2JvcqZoZGM6dGl0bGVsU291cmNlIGFzc2V0aWRjOmZvcm1hdGppbWFnZS93ZWJwbHJlbGF0aW9uc2hpcGhwYXJlbnRPZnF2YWxpZGF0aW9uUmVzdWx0c6FuYWN0aXZlTWFuaWZlc3SjZ2ZhaWx1cmWAZ3N1Y2Nlc3OHomRjb2RleBlzaWduaW5nQ3JlZGVudGlhbC50cnVzdGVkY3VybHhNc2VsZiNqdW1iZj0vYzJwYS91cm46YzJwYTozZjU1OGI1MC1kOTNhLTY1MmItNDk3NC00NzA5YWJjNTcyNmYvYzJwYS5zaWduYXR1cmWiZGNvZGV4HWNsYWltU2lnbmF0dXJlLmluc2lkZVZhbGlkaXR5Y3VybHhNc2VsZiNqdW1iZj0vYzJwYS91cm46YzJwYTozZjU1OGI1MC1kOTNhLTY1MmItNDk3NC00NzA5YWJjNTcyNmYvYzJwYS5zaWduYXR1cmWiZGNvZGV4GGNsYWltU2lnbmF0dXJlLnZhbGlkYXRlZGN1cmx4TXNlbGYjanVtYmY9L2MycGEvdXJuOmMycGE6M2Y1NThiNTAtZDkzYS02NTJiLTQ5NzQtNDcwOWFiYzU3MjZmL2MycGEuc2lnbmF0dXJlomRjb2RleBlhc3NlcnRpb24uaGFzaGVkVVJJLm1hdGNoY3VybHhec2VsZiNqdW1iZj0vYzJwYS91cm46YzJwYTozZjU1OGI1MC1kOTNhLTY1MmItNDk3NC00NzA5YWJjNTcyNmYvYzJwYS5hc3NlcnRpb25zL2MycGEuYWN0aW9ucy52MqJkY29kZXgZYXNzZXJ0aW9uLmhhc2hlZFVSSS5tYXRjaGN1cmx4XXNlbGYjanVtYmY9L2MycGEvdXJuOmMycGE6M2Y1NThiNTAtZDkzYS02NTJiLTQ5NzQtNDcwOWFiYzU3MjZmL2MycGEuYXNzZXJ0aW9ucy9jMnBhLmhhc2guZGF0YaJkY29kZXgZYXNzZXJ0aW9uLmhhc2hlZFVSSS5tYXRjaGN1cmx4e3NlbGYjanVtYmY9L2MycGEvdXJuOmMycGE6M2Y1NThiNTAtZDkzYS02NTJiLTQ5NzQtNDcwOWFiYzU3MjZmL2MycGEuYXNzZXJ0aW9ucy9jb20ubWV0YS5nZW5lcmF0ZWQtbWVkaWEtcHJvdmVuYW5jZS5tZXRhZGF0YaJkY29kZXgYYXNzZXJ0aW9uLmRhdGFIYXNoLm1hdGNoY3VybHhdc2VsZiNqdW1iZj0vYzJwYS91cm46YzJwYTozZjU1OGI1MC1kOTNhLTY1MmItNDk3NC00NzA5YWJjNTcyNmYvYzJwYS5hc3NlcnRpb25zL2MycGEuaGFzaC5kYXRhbWluZm9ybWF0aW9uYWyBomRjb2RleB5zaWduaW5nQ3JlZGVudGlhbC5vY3NwLnNraXBwZWRjdXJseE1zZWxmI2p1bWJmPS9jMnBhL3VybjpjMnBhOjNmNTU4YjUwLWQ5M2EtNjUyYi00OTc0LTQ3MDlhYmM1NzI2Zi9jMnBhLnNpZ25hdHVyZW5hY3RpdmVNYW5pZmVzdKJjdXJseD5zZWxmI2p1bWJmPS9jMnBhL3VybjpjMnBhOjNmNTU4YjUwLWQ5M2EtNjUyYi00OTc0LTQ3MDlhYmM1NzI2ZmRoYXNoWCBLz9OW3cQMF2VnmJfMmli8GqUABF76rOTZN0qsRUoliG5jbGFpbVNpZ25hdHVyZaJjdXJseE1zZWxmI2p1bWJmPS9jMnBhL3VybjpjMnBhOjNmNTU4YjUwLWQ5M2EtNjUyYi00OTc0LTQ3MDlhYmM1NzI2Zi9jMnBhLnNpZ25hdHVyZWRoYXNoWCA4hOKWd2hHeXCyw7ygWTfW+rIhgXjHwAFUkK7CSWC9iw==" width="96" height="96" alt="Mawaxi AI logo">
        </div>
        <div class="wordmark">Mawaxi AI</div>
        <div class="tagline">A voyage begins.</div>
        <button id="personaBtn" title="Persona — tap to change">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          <span id="personaBtnLabel">General</span>
        </button>
      </div>
    </div>
    <div id="wsPanel" style="display:none">
      <div class="wsHead"><span>Workspace</span><span id="wsPathLabel" class="wsPathLabel"></span><button id="wsRefresh" class="mini" type="button">Refresh</button><button id="wsPanelClose" class="iconbtn" type="button" title="Close">&#10005;</button></div>
      <div id="wsList"></div>
    </div>
    <form id="f">
      <div id="composer">
        <div id="skillChip" style="display:none"><span id="skillChipName"></span><button id="skillChipX" type="button" title="Clear skill">&#10005;</button></div>
        <div id="attachPreview" style="display:none"><img id="attachThumb" alt=""><span id="attachName"></span><button id="attachRemove" type="button" title="Remove image">&#10005;</button></div>
        <div class="input-row">
          <input id="q" autocomplete="off" placeholder="Message ...">
          <select id="model"></select>
        </div>
        <div class="action-row">
          <button type="submit" id="sendBtn" title="Send" aria-label="Send">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
          </button>
          <button type="button" id="attachBtn" class="iconbtn" title="Attach">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
          </button>
          <button type="button" id="imgBtn" class="iconbtn" title="Generate image">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
          </button>
          <button type="button" id="vidBtn" class="iconbtn" title="Generate video">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><rect x="2" y="5" width="15" height="14" rx="2"/><path d="M17 10l5-3v10l-5-3"/></svg>
          </button>
          <div id="modeToggle">
            <button type="button" id="modeAgent">Agent</button>
            <button type="button" id="modeChat" class="active">Chat</button>
          </div>
          <button type="button" id="stopAgentBtn" style="display:none">Stop</button>
          <button type="button" id="micBtn" title="Voice input" aria-label="Voice input">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="#fff"><path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3z"/><path d="M19 11a7 7 0 0 1-14 0H3a9 9 0 0 0 8 8.94V23h2v-3.06A9 9 0 0 0 21 11h-2z"/></svg>
          </button>
        </div>
      </div>
      <input type="file" id="imgFile" accept="image/*,application/pdf,.pdf" style="display:none">
    </form>
  </div>
</div>
<div id="modal">
  <div class="modal-panel">
    <div class="modal-head">
      <div class="modal-tabs">
        <button data-tab="endpoints" class="mtab active">Endpoints</button>
        <button data-tab="memory" class="mtab">Memory</button>
        <button data-tab="skills" class="mtab">Skills</button>
        <button data-tab="workspace" class="mtab">Workspace</button>
        <button data-tab="video" class="mtab">Video</button>
        <button data-tab="theme" class="mtab">Theme</button>
        <button data-tab="data" class="mtab">Data</button>
      </div>
      <button id="modalClose" class="iconbtn" title="Close">&#10005;</button>
    </div>
    <div class="modal-body">
      <div id="tab-endpoints" class="tabpage">
        <p class="hint">Pick a provider, then just type your API key. Base URL and model fill in themselves.</p>
        <select id="epProvider" class="fld"></select>
        <input id="epName" class="fld" placeholder="Name (e.g. NVIDIA)">
        <input id="epUrl" class="fld" placeholder="Base URL (e.g. https://integrate.api.nvidia.com/v1)">
        <input id="epKey" class="fld" type="password" placeholder="API key" autocomplete="off">
        <select id="epModelSel" class="fld" style="display:none"></select>
        <input id="epModel" class="fld" placeholder="Model (e.g. deepseek-v4.1-flash)">
        <button id="epFetchModels" class="btn" type="button">Fetch models</button>
        <div id="epModelNote" class="hint" style="display:none"></div>
        <button id="epAdd" class="btn" type="button">Add endpoint</button>
        <button id="copilotSignin" class="btn" type="button" style="display:none">Sign in with GitHub</button>
        <div id="copilotStatus" class="hint" style="display:none"></div>
        <div id="epList"></div>
      </div>
      <div id="tab-memory" class="tabpage" style="display:none">
        <p class="hint">Facts the AI remembers in every chat.</p>
        <div class="rowflex"><input id="memText" class="fld" placeholder="A fact to remember..."><button id="memAdd" class="btn" type="button">Add</button></div>
        <div id="memList"></div>
      </div>
      <div id="tab-skills" class="tabpage" style="display:none">
        <p class="hint">Skills are instructions used as the system prompt for a chat.</p>
        <input id="skName" class="fld" placeholder="Skill name (e.g. Summarise)">
        <textarea id="skInstr" class="fld" rows="3" placeholder="Instruction (e.g. Summarise the user's text clearly and briefly...)"></textarea>
        <button id="skAdd" class="btn" type="button">Add skill</button>
        <div id="skList"></div>
      </div>
      <div id="tab-workspace" class="tabpage" style="display:none">
        <p class="hint">Folder the agent starts in for files it creates and runs. Leave it empty to use the default workspace. The agent runs as root with no safety check and can reach any path.</p>
        <div class="rowflex"><input id="wsPath" class="fld" placeholder="/root/odysseus-rust/workspace"><button id="wsSave" class="btn" type="button">Save</button></div>
        <div id="wsMsg" class="hint"></div>
      </div>
      <div id="tab-video" class="tabpage" style="display:none">
        <p class="hint">Video generation. Free video (about one 5-second clip a day, animated from a still image) works with no key. For more video, add a Magic Hour key — free credits on signup at magichour.ai/developer, no card needed. The key stays on this phone and is never shown again.</p>
        <div class="rowflex"><input id="mhKey" class="fld" type="password" placeholder="Magic Hour API key" autocomplete="off"><button id="mhSave" class="btn" type="button">Save</button></div>
        <div id="mhMsg" class="hint"></div>
      </div>
      <div id="tab-theme" class="tabpage" style="display:none">
        <p class="hint">Pick a look for the whole app. It is saved on this phone.</p>
        <div id="themeGrid" class="themegrid"></div>
        <div id="themeMsg" class="hint"></div>
        <p class="hint" style="margin-top:14px">Chat text size — how big the text is inside chat bubbles.</p>
        <div id="fontSizeRow" class="themegrid"></div>
        <div id="fontSizeMsg" class="hint"></div>
        <p class="hint" style="margin-top:14px">Chat background — an image behind the chat bubbles. It is saved on this phone.</p>
        <div class="rowflex"><button id="bgUploadBtn" class="btn" type="button">Upload image</button><button id="bgRemoveBtn" class="btn" type="button">Remove</button></div>
        <input id="bgFile" type="file" accept="image/*" style="display:none">
        <div id="bgPreview" style="margin-top:8px"></div>
        <div id="bgMsg" class="hint"></div>
      </div>
      <div id="tab-data" class="tabpage" style="display:none">
        <p class="hint">Data Saver cuts mobile data use. When it is on: images generate at 512px instead of 1024px, images and videos only load when you tap them, and free video uses a smaller still image. Off means everything loads straight away as before.</p>
        <div class="rowflex"><button id="dsToggle" class="btn" type="button">Turn Data Saver on</button></div>
        <div id="dsMsg" class="hint"></div>
      </div>
      </div>
    </div>
  </div>
</div>
<div id="featureView">
  <div class="fv-head"><span id="fvTitle" class="fv-title"></span><button id="fvClose" class="iconbtn" title="Close">&#10005;</button></div>
  <div id="fvBody" class="fv-body"></div>
</div>
<div id="toast"></div>
<script>
const log=document.getElementById('log'), f=document.getElementById('f'), q=document.getElementById('q'),
  chatlist=document.getElementById('chatlist'), newchat=document.getElementById('newchat'),
  modelSel=document.getElementById('model'), emptyEl=document.getElementById('empty'),
  personaSel=document.getElementById('persona'),
  side=document.getElementById('side'), overlay=document.getElementById('overlay'),
  hamburger=document.getElementById('hamburger'), closeSide=document.getElementById('closeSide'),
  titleBtn=document.getElementById('titleBtn'), toast=document.getElementById('toast'),
  modeAgent=document.getElementById('modeAgent'), modeChat=document.getElementById('modeChat'),
  micBtn=document.getElementById('micBtn'), attachBtn=document.getElementById('attachBtn'),
  personaBtn=document.getElementById('personaBtn'), personaBtnLabel=document.getElementById('personaBtnLabel'), chatSearch=document.getElementById('chatSearch'),
  imgBtn=document.getElementById('imgBtn'), vidBtn=document.getElementById('vidBtn'),
  modal=document.getElementById('modal'), modalClose=document.getElementById('modalClose'),
  openSettings=document.getElementById('openSettings'), openMemory=document.getElementById('openMemory'),
  openSkills=document.getElementById('openSkills'),
  epName=document.getElementById('epName'), epUrl=document.getElementById('epUrl'),
  epKey=document.getElementById('epKey'), epModel=document.getElementById('epModel'),
  epModelSel=document.getElementById('epModelSel'), epFetchModels=document.getElementById('epFetchModels'),
  epModelNote=document.getElementById('epModelNote'),
  epAdd=document.getElementById('epAdd'), epList=document.getElementById('epList'),
  memText=document.getElementById('memText'), memAdd=document.getElementById('memAdd'),
  memList=document.getElementById('memList'),
  skName=document.getElementById('skName'), skInstr=document.getElementById('skInstr'),
  skAdd=document.getElementById('skAdd'), skList=document.getElementById('skList'),
  skillChip=document.getElementById('skillChip'), skillChipName=document.getElementById('skillChipName'),
  skillChipX=document.getElementById('skillChipX'),
  imgFile=document.getElementById('imgFile'),
  attachPreview=document.getElementById('attachPreview'), attachThumb=document.getElementById('attachThumb'),
  attachName=document.getElementById('attachName'), attachRemove=document.getElementById('attachRemove'),
  openWorkspace=document.getElementById('openWorkspace'),
  openNotes=document.getElementById('openNotes'),
  openResearch=document.getElementById('openResearch'),
  openCookbook=document.getElementById('openCookbook'),
  openLibrary=document.getElementById('openLibrary'),
  readAloudSide=document.getElementById('readAloudSide'),
  featureView=document.getElementById('featureView'),
  fvTitle=document.getElementById('fvTitle'),
  fvBody=document.getElementById('fvBody'),
  fvClose=document.getElementById('fvClose'),
  stopAgentBtn=document.getElementById('stopAgentBtn'), wsPanelClose=document.getElementById('wsPanelClose'),
  wsPanel=document.getElementById('wsPanel'),
  wsList=document.getElementById('wsList'),
  wsRefresh=document.getElementById('wsRefresh'), wsPathLabel=document.getElementById('wsPathLabel'),
  wsPath=document.getElementById('wsPath'), wsSave=document.getElementById('wsSave'),
  wsMsg=document.getElementById('wsMsg'),
  mhKey=document.getElementById('mhKey'), mhSave=document.getElementById('mhSave'),
  mhMsg=document.getElementById('mhMsg'),
  openBrowser=document.getElementById('openBrowser'),
  openCalendar=document.getElementById('openCalendar'),
  openPdf=document.getElementById('openPdf'),
  chatPdfBtn=document.getElementById('chatPdfBtn'),
  themeGrid=document.getElementById('themeGrid'),
  themeMsg=document.getElementById('themeMsg'),
  fontSizeRow=document.getElementById('fontSizeRow'),
  fontSizeMsg=document.getElementById('fontSizeMsg'),
  bgUploadBtn=document.getElementById('bgUploadBtn'),
  bgRemoveBtn=document.getElementById('bgRemoveBtn'),
  bgFile=document.getElementById('bgFile'),
  bgPreview=document.getElementById('bgPreview'),
  bgMsg=document.getElementById('bgMsg'),
  zoomOut=document.getElementById('zoomOut'),
  zoomIn=document.getElementById('zoomIn'),
  zoomLabel=document.getElementById('zoomLabel'),
  dsToggle=document.getElementById('dsToggle'),
  dsMsg=document.getElementById('dsMsg');
const DEFAULT_MODEL='qwen3-abliterated-0.6b';
let chats=[]; let current=null; let currentModel=DEFAULT_MODEL;
let toastTimer=null; let searchTimer=null;
let endpoints=[]; let presets=[]; let memories=[]; let skills=[]; let epFetchedModels=[];
let activeEndpointId='';
try{ activeEndpointId=localStorage.getItem('mawaxi_active_endpoint')||''; }catch(e){}
let activePuterModel='';
try{ activePuterModel=localStorage.getItem('mawaxi_active_puter')||''; }catch(e){}
// Puter (puter.js) free higher-end models, shown as a "Puter" group in the model picker.
// The browser calls puter.ai.chat() directly; the server never sees a Puter token.
const PUTER_MODELS=[
  {id:'openai/gpt-5.5', label:'GPT-5.5'},
  {id:'anthropic/claude-opus-5-5', label:'Claude Opus 5.5'},
  {id:'deepseek/deepseek-v4-pro', label:'DeepSeek V4 Pro'},
  {id:'google/gemini-3.8-flash', label:'Gemini 3.8 Flash'},
  {id:'x-ai/grok-4.7', label:'Grok 4.7'},
  {id:'z-ai/glm-5.3', label:'GLM 5.3'},
  {id:'qwen/qwen3.8-max', label:'Qwen3.8 Max'},
  {id:'moonshotai/kimi-k3', label:'Kimi K3'},
  {id:'openai/gpt-5.3-codex', label:'GPT-5.3 Codex'},
  {id:'google/gemma-4-31b-it', label:'Gemma 4 31B'},
  {id:'meta-llama/llama-4-maverick', label:'Llama 4 Maverick'}
];
function puterReady(){ return (typeof window.puter!=='undefined' && window.puter && window.puter.ai); }
function setActivePuter(mid){
  activePuterModel=mid||'';
  try{
    if(activePuterModel) localStorage.setItem('mawaxi_active_puter', activePuterModel);
    else localStorage.removeItem('mawaxi_active_puter');
  }catch(e){}
  refreshModelSelect();
}

// ---------- Themes ----------
// Same names and colours as the Odysseus themes (theme.js THEMES):
// Spotify is the current default look; Snapchat, TikTok, WhatsApp,
// Instagram, Sunset and Lavender match Odysseus, plus Midnight and Light.
// Each theme sets the CSS variables the whole app uses, so one switch
// recolours chat, sidebar and panels. Choice persists in localStorage.
const THEMES={
  spotify:{name:'Spotify',bg:'#121212',panel:'#1a1a1a',panel2:'#242424',border:'#2c2c2c',green:'#1DB954',greenHover:'#169c46',text:'#ffffff',muted:'#9a9a9a',user:'#1DB954',userfg:'#ffffff',ai:'#242424',code:'#0d0d0d',side:'#000000',hover:'#2e2e2e'},
  snapchat:{name:'Snapchat',bg:'#ffffff',panel:'#f7f7f8',panel2:'#efeff1',border:'#e4e4e7',green:'#262626',greenHover:'#000000',text:'#262626',muted:'#8e8e93',user:'#FFFC00',userfg:'#262626',ai:'#efeff1',code:'#f0f0f2',side:'#ffffff',hover:'#e8e8ea'},
  tiktok:{name:'TikTok',bg:'#000000',panel:'#121212',panel2:'#1a1a1a',border:'#2a2a2a',green:'#FE2C55',greenHover:'#d92448',text:'#ffffff',muted:'#9a9a9a',user:'#FE2C55',userfg:'#ffffff',ai:'#1e1e1e',code:'#0d0d0d',side:'#000000',hover:'#2a2a2a'},
  whatsapp:{name:'WhatsApp',bg:'#0b141a',panel:'#111b21',panel2:'#1f2c34',border:'#222d34',green:'#25D366',greenHover:'#1fb857',text:'#e9edef',muted:'#8696a0',user:'#005c4b',userfg:'#e9edef',ai:'#1f2c34',code:'#0d1418',side:'#111b21',hover:'#2a3942'},
  instagram:{name:'Instagram',bg:'#ffffff',panel:'#fafafa',panel2:'#efefef',border:'#dbdbdb',green:'#E1306C',greenHover:'#c2275a',text:'#262626',muted:'#8e8e8e',user:'#E1306C',userfg:'#ffffff',ai:'#efefef',code:'#f0f0f0',side:'#ffffff',hover:'#e8e8e8'},
  sunset:{name:'Sunset',bg:'#221627',panel:'#1a101f',panel2:'#2c1e33',border:'#4a2c3f',green:'#ff6b35',greenHover:'#e55a28',text:'#ffe8d6',muted:'#b99a8e',user:'#ff6b35',userfg:'#221627',ai:'#2c1e33',code:'#160d18',side:'#1a101f',hover:'#3d2436'},
  lavender:{name:'Lavender',bg:'#f6f2ff',panel:'#ffffff',panel2:'#ece7f8',border:'#ddd0f5',green:'#8b5cf6',greenHover:'#7c4def',text:'#3b2d5c',muted:'#8a7ba8',user:'#8b5cf6',userfg:'#ffffff',ai:'#ece7f8',code:'#efeaf9',side:'#ffffff',hover:'#e2d9f7'},
  midnight:{name:'Midnight',bg:'#0d1117',panel:'#161b22',panel2:'#21262d',border:'#30363d',green:'#58a6ff',greenHover:'#1f6feb',text:'#c9d1d9',muted:'#8b949e',user:'#1f6feb',userfg:'#ffffff',ai:'#161b22',code:'#0d1117',side:'#010409',hover:'#30363d'},
  light:{name:'Light',bg:'#f3f5f7',panel:'#ffffff',panel2:'#eceff1',border:'#cbd3da',green:'#1DB954',greenHover:'#169c46',text:'#26313a',muted:'#6b7a88',user:'#1DB954',userfg:'#ffffff',ai:'#ffffff',code:'#e8ecef',side:'#ffffff',hover:'#dfe5e9'}
};
let currentTheme='spotify';
try{ const t=localStorage.getItem('mawaxi-theme'); if(t && THEMES[t]) currentTheme=t; }catch(e){}
function applyTheme(name){
  const t=THEMES[name]; if(!t) return;
  currentTheme=name;
  const r=document.documentElement.style;
  r.setProperty('--bg',t.bg); r.setProperty('--panel',t.panel); r.setProperty('--panel2',t.panel2);
  r.setProperty('--border',t.border); r.setProperty('--green',t.green); r.setProperty('--green-hover',t.greenHover);
  r.setProperty('--text',t.text); r.setProperty('--muted',t.muted); r.setProperty('--user',t.user);
  r.setProperty('--userfg',t.userfg); r.setProperty('--ai',t.ai); r.setProperty('--code',t.code);
  r.setProperty('--side',t.side); r.setProperty('--hover',t.hover);
  try{ localStorage.setItem('mawaxi-theme',name); }catch(e){}
  renderThemeGrid();
  if(themeMsg) themeMsg.textContent='Theme: '+t.name;
}
function renderThemeGrid(){
  if(!themeGrid) return;
  themeGrid.innerHTML='';
  for(const key in THEMES){
    const t=THEMES[key];
    const b=document.createElement('button'); b.type='button'; b.className='themebtn'+(key===currentTheme?' active':'');
    const dots=document.createElement('span'); dots.className='dots';
    [t.bg,t.user,t.green].forEach(function(c){ const d=document.createElement('span'); d.className='dot'; d.style.background=c; dots.appendChild(d); });
    const nm=document.createElement('span'); nm.textContent=t.name;
    b.appendChild(dots); b.appendChild(nm);
    b.addEventListener('click', function(){ applyTheme(key); showToast('Theme: '+t.name); });
    themeGrid.appendChild(b);
  }
}
applyTheme(currentTheme);

// ---------- Chat text size + zoom ----------
// Chat bubble font size: Small 14 / Normal 16 (today's size) / Large 18 /
// Extra Large 21. Zoom: 80%..150% in 10% steps, reset to 100%. Both persist
// in localStorage and are restored on load. Pinch-to-zoom also works: the
// viewport meta (width=device-width, initial-scale=1) does not block scaling.
const CHAT_FONT_SIZES={small:14,normal:16,large:18,xlarge:21};
const CHAT_FONT_NAMES={small:'Small',normal:'Normal',large:'Large',xlarge:'Extra Large'};
let chatFontSize='normal';
try{ const s=localStorage.getItem('mawaxi-chat-font-size'); if(s && CHAT_FONT_SIZES[s]) chatFontSize=s; }catch(e){}
function applyChatFontSize(name){
  if(!CHAT_FONT_SIZES[name]) return;
  chatFontSize=name;
  document.documentElement.style.setProperty('--chat-font-size',CHAT_FONT_SIZES[name]+'px');
  try{ localStorage.setItem('mawaxi-chat-font-size',name); }catch(e){}
  renderFontSizeRow();
  if(fontSizeMsg) fontSizeMsg.textContent='Chat text size: '+CHAT_FONT_NAMES[name];
}
function renderFontSizeRow(){
  if(!fontSizeRow) return;
  fontSizeRow.innerHTML='';
  for(const key in CHAT_FONT_SIZES){
    const b=document.createElement('button'); b.type='button'; b.className='themebtn'+(key===chatFontSize?' active':'');
    const nm=document.createElement('span'); nm.textContent=CHAT_FONT_NAMES[key]+' ('+CHAT_FONT_SIZES[key]+'px)';
    b.appendChild(nm);
    b.addEventListener('click',function(){ applyChatFontSize(key); showToast('Chat text size: '+CHAT_FONT_NAMES[key]); });
    fontSizeRow.appendChild(b);
  }
}
let chatZoom=1;
try{ const z=parseFloat(localStorage.getItem('mawaxi-chat-zoom')); if(z>=0.8&&z<=1.5) chatZoom=z; }catch(e){}
function applyChatZoom(z){
  chatZoom=Math.min(1.5,Math.max(0.8,Math.round(z*10)/10));
  document.documentElement.style.setProperty('--chat-zoom',chatZoom);
  try{ localStorage.setItem('mawaxi-chat-zoom',String(chatZoom)); }catch(e){}
  if(zoomLabel) zoomLabel.textContent=Math.round(chatZoom*100)+'%';
}
function zoomStep(d){ applyChatZoom(chatZoom+d); }
if(zoomOut) zoomOut.addEventListener('click',function(){ zoomStep(-0.1); });
if(zoomIn) zoomIn.addEventListener('click',function(){ zoomStep(0.1); });
if(zoomLabel) zoomLabel.addEventListener('click',function(){ applyChatZoom(1); showToast('Zoom: 100%'); });
applyChatFontSize(chatFontSize);
applyChatZoom(chatZoom);

// ---------- Chat background image ----------
// An image behind the chat bubbles, set from Settings > Theme. The image is
// downscaled to at most 1024px on its longest side and stored as a JPEG data
// URL in localStorage ('mawaxi-chat-bg'), so it stays on this phone and
// survives reloads. The bubbles keep their solid theme colours, so text stays
// readable over the picture.
function applyChatBg(dataUrl){
  if(!dataUrl){ clearChatBg(); return; }
  log.style.backgroundImage='url("'+dataUrl+'")';
  log.style.backgroundSize='cover';
  log.style.backgroundPosition='center';
  try{ localStorage.setItem('mawaxi-chat-bg',dataUrl); }catch(e){}
  renderBgPreview();
  if(bgMsg) bgMsg.textContent='Chat background: set';
}
function clearChatBg(){
  log.style.backgroundImage='';
  log.style.backgroundSize='';
  log.style.backgroundPosition='';
  try{ localStorage.removeItem('mawaxi-chat-bg'); }catch(e){}
  renderBgPreview();
  if(bgMsg) bgMsg.textContent='Chat background: none';
}
function renderBgPreview(){
  if(!bgPreview) return;
  let dataUrl=null;
  try{ dataUrl=localStorage.getItem('mawaxi-chat-bg'); }catch(e){}
  bgPreview.innerHTML='';
  if(dataUrl){
    const img=document.createElement('img');
    img.src=dataUrl; img.alt='Chat background';
    img.style.cssText='max-width:120px;max-height:80px;border-radius:8px;border:1px solid var(--border)';
    bgPreview.appendChild(img);
  }
}
function handleBgFile(file){
  if(!file || !file.type || file.type.indexOf('image/')!==0){ showToast('Pick an image file'); return; }
  const reader=new FileReader();
  reader.onload=function(){
    const img=new Image();
    img.onload=function(){
      const maxSide=1024;
      let w=img.naturalWidth, h=img.naturalHeight;
      if(w>maxSide || h>maxSide){
        const s=Math.min(maxSide/w, maxSide/h);
        w=Math.round(w*s); h=Math.round(h*s);
      }
      const cv=document.createElement('canvas'); cv.width=w; cv.height=h;
      cv.getContext('2d').drawImage(img,0,0,w,h);
      let dataUrl='';
      try{ dataUrl=cv.toDataURL('image/jpeg',0.82); }catch(e){ showToast('Could not use that image'); return; }
      applyChatBg(dataUrl);
      showToast('Chat background set');
    };
    img.onerror=function(){ showToast('Could not read that image'); };
    img.src=reader.result;
  };
  reader.onerror=function(){ showToast('Could not read that image'); };
  reader.readAsDataURL(file);
}
if(bgUploadBtn) bgUploadBtn.addEventListener('click',function(){ if(bgFile) bgFile.click(); });
if(bgFile) bgFile.addEventListener('change',function(){ if(bgFile.files && bgFile.files[0]) handleBgFile(bgFile.files[0]); bgFile.value=''; });
if(bgRemoveBtn) bgRemoveBtn.addEventListener('click',function(){ clearChatBg(); showToast('Chat background removed'); });
try{ const savedBg=localStorage.getItem('mawaxi-chat-bg'); if(savedBg) applyChatBg(savedBg); else renderBgPreview(); }catch(e){ renderBgPreview(); }

// ---------- Personas ----------
// Built-in reply styles: General (today's behaviour, no extra
// instruction), Medical Lab Expert, Teacher, Coder. The choice is sent
// with every chat/agent request as `persona` and persists in localStorage.
const PERSONAS={general:'General',medical:'Medical Lab Expert',teacher:'Teacher',coder:'Coder'};
let currentPersona='general';
try{ const p=localStorage.getItem('mawaxi-persona'); if(p && PERSONAS[p]) currentPersona=p; }catch(e){}
function applyPersona(id){
  if(!PERSONAS[id]) return;
  currentPersona=id;
  try{ localStorage.setItem('mawaxi-persona',id); }catch(e){}
  if(personaSel) personaSel.value=id;
  if(personaBtnLabel) personaBtnLabel.textContent=PERSONAS[id];
}
if(personaSel){ personaSel.value=currentPersona; personaSel.addEventListener('change',function(){ applyPersona(personaSel.value); showToast('Persona: '+PERSONAS[currentPersona]); }); }
  if(personaBtnLabel) personaBtnLabel.textContent=PERSONAS[currentPersona];

// ---------- Data Saver ----------
// When on: images generate at 512px, images/videos only load when tapped,
// and free video uses a smaller still. Persists in localStorage.
let dataSaver=false;
try{ dataSaver=localStorage.getItem('mawaxi-data-saver')==='1'; }catch(e){}
function setDataSaver(on){
  dataSaver=!!on;
  try{ localStorage.setItem('mawaxi-data-saver', dataSaver?'1':'0'); }catch(e){}
  if(dsToggle) dsToggle.textContent=dataSaver?'Turn Data Saver off':'Turn Data Saver on';
  if(dsMsg) dsMsg.textContent=dataSaver?'Data Saver is ON — images 512px, tap to load media.':'Data Saver is off — media loads straight away.';
}
// Tap-to-load placeholder used when Data Saver is on: the media is NOT
// downloaded until the user taps. Returns the button's container.
function tapToLoadBox(label, loadFn){
  const box=document.createElement('div'); box.className='tapload';
  const tx=document.createElement('div'); tx.textContent=label+' — Data Saver is on.';
  const btn=document.createElement('button'); btn.type='button'; btn.textContent='Tap to load';
  btn.addEventListener('click', function(){
    btn.disabled=true; btn.textContent='Loading…';
    loadFn(box);
  });
  box.appendChild(tx); box.appendChild(btn);
  return box;
}
if(dsToggle) dsToggle.addEventListener('click', function(){ setDataSaver(!dataSaver); showToast(dataSaver?'Data Saver on':'Data Saver off'); });
setDataSaver(dataSaver);
let pendingImage=null; let pendingPdf=null; let recActive=null;
let mode='chat'; let currentRunId=null; let agentRunning=false; let agentReader=null; let chatBusy=false;

function showToast(msg){
  toast.textContent=msg; toast.style.display='block';
  if(toastTimer) clearTimeout(toastTimer);
  toastTimer=setTimeout(function(){toast.style.display='none';},2200);
}
function openDrawer(){side.classList.add('open'); overlay.classList.add('show');}
function closeDrawer(){side.classList.remove('open'); overlay.classList.remove('show');}
function toggleDrawer(){
  if(window.matchMedia('(max-width: 900px)').matches){
    if(side.classList.contains('open')){closeDrawer();}else{openDrawer();}
  }else{
    side.classList.toggle('collapsed');
  }
}
hamburger.addEventListener('click', toggleDrawer);
closeSide.addEventListener('click', closeDrawer);
overlay.addEventListener('click', closeDrawer);
titleBtn.addEventListener('click', openDrawer);
attachBtn.addEventListener('click', function(){ imgFile.click(); });
imgFile.addEventListener('change', async function(){
  const file=imgFile.files && imgFile.files[0];
  imgFile.value='';
  if(!file) return;
  if(file.type==='application/pdf' || /\.pdf$/i.test(file.name||'')){ await attachPdfFile(file); return; }
  if(file.type.indexOf('image/')!==0){ showToast('Please pick an image or PDF file'); return; }
  try{
    pendingPdf=null;
    pendingImage=await processImageFile(file);
    attachThumb.src=pendingImage.dataUrl; attachThumb.style.display='';
    attachName.textContent=file.name||'image';
    attachPreview.style.display='flex';
    q.focus();
  }catch(err){ showToast('Could not read that image'); }
});
async function attachPdfFile(file){
  pendingImage=null;
  attachThumb.style.display='none';
  attachName.textContent='Uploading '+file.name+'\u2026';
  attachPreview.style.display='flex';
  try{
    const data=await new Promise(function(resolve, reject){
      const reader=new FileReader();
      reader.onload=function(){ resolve(reader.result); };
      reader.onerror=function(){ reject(new Error('could not read file')); };
      reader.readAsDataURL(file);
    });
    const res=await fetch('/api/pdf/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:file.name, data:data})});
    const t=await res.text();
    let j=null; try{ j=JSON.parse(t); }catch(e){}
    if(!res.ok){ throw new Error(t||('server error '+res.status)); }
    pendingPdf={id:j.id, name:j.name, text:(j.text||'')};
    if(j.extract_error){ showToast(j.extract_error); }
    attachName.textContent=j.name;
    attachThumb.style.display='none';
    attachPreview.style.display='flex';
    q.focus();
  }catch(err){
    attachPreview.style.display='none';
    showToast('PDF upload failed: '+err.message);
  }
}
attachRemove.addEventListener('click', function(){ pendingImage=null; pendingPdf=null; attachPreview.style.display='none'; });
function processImageFile(file){
  return new Promise(function(resolve, reject){
    const reader=new FileReader();
    reader.onload=function(){
      const img=new Image();
      img.onload=function(){
        const maxDim=1024;
        let w=img.naturalWidth, h=img.naturalHeight;
        const scale=Math.min(1, maxDim/Math.max(w,h));
        w=Math.max(1, Math.round(w*scale)); h=Math.max(1, Math.round(h*scale));
        const canvas=document.createElement('canvas'); canvas.width=w; canvas.height=h;
        const ctx=canvas.getContext('2d'); ctx.drawImage(img,0,0,w,h);
        const dataUrl=canvas.toDataURL('image/jpeg',0.8);
        resolve({dataUrl:dataUrl, b64:dataUrl.split(',')[1]});
      };
      img.onerror=function(){reject(new Error('bad image'));};
      img.src=reader.result;
    };
    reader.onerror=function(){reject(new Error('read failed'));};
    reader.readAsDataURL(file);
  });
}
imgBtn.addEventListener('click', function(){generateImageFromInput();});
vidBtn.addEventListener('click', function(){generateVideoFromInput();});
chatSearch.addEventListener('input', function(){
  if(searchTimer) clearTimeout(searchTimer);
  searchTimer=setTimeout(doSearch, 300);
});
async function doSearch(){
  const query=chatSearch.value.trim();
  if(!query){ await loadChats(); return; }
  try{
    const r=await fetch('/api/chats/search?q='+encodeURIComponent(query));
    chats=await r.json();
    renderChatList();
  }catch(e){}
}
micBtn.addEventListener('click', function(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){ showToast('Voice input is not supported in this browser'); return; }
  if(recActive){ try{recActive.stop();}catch(e){} return; }
  const rec=new SR();
  rec.lang='en-US'; rec.interimResults=false; rec.maxAlternatives=1;
  rec.onresult=function(e){
    const text=e.results[0][0].transcript;
    q.value=(q.value?q.value+' ':'')+text; q.focus();
  };
  rec.onerror=function(e){ showToast('Voice input failed: '+(e.error||'error')); };
  rec.onend=function(){ recActive=null; micBtn.classList.remove('listening'); };
  try{ rec.start(); recActive=rec; micBtn.classList.add('listening'); showToast('Listening...'); }
  catch(err){ showToast('Voice input failed'); }
});
function openModal(tab){
  modal.classList.add('show');
  showTab(tab||'endpoints');
  if(tab==='workspace'){ loadWorkspaceSetting(); }
}
function closeModal(){ modal.classList.remove('show'); }
function showTab(tab){
  document.querySelectorAll('.mtab').forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-tab')===tab); });
  document.querySelectorAll('.tabpage').forEach(function(p){ p.style.display = p.id==='tab-'+tab ? 'flex' : 'none'; });
  if(tab==='video'){ loadVideoSetting(); }
}
document.querySelectorAll('.mtab').forEach(function(b){ b.addEventListener('click', function(){ showTab(b.getAttribute('data-tab')); }); });
modalClose.addEventListener('click', closeModal);
modal.addEventListener('click', function(e){ if(e.target===modal) closeModal(); });
openSettings.addEventListener('click', function(){ openModal('endpoints'); closeDrawer(); });
openMemory.addEventListener('click', function(){ openModal('memory'); closeDrawer(); });
openSkills.addEventListener('click', function(){ openModal('skills'); closeDrawer(); });
async function loadWorkspaceSetting(){
  try{
    const r=await fetch('/api/settings');
    const j=await r.json();
    wsPath.value=j.workspace||'';
    wsMsg.textContent='';
  }catch(e){}
}
wsSave.addEventListener('click', async function(){
  wsMsg.textContent='Saving...';
  try{
    const r=await fetch('/api/settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({workspace:wsPath.value.trim()})});
    if(!r.ok){ wsMsg.textContent=await r.text(); return; }
    const j=await r.json();
    wsPath.value=j.workspace||'';
    wsMsg.textContent='Workspace saved: '+j.workspace;
    refreshWorkspace();
  }catch(e){ wsMsg.textContent='Could not save workspace'; }
});
async function loadVideoSetting(){
  try{
    const r=await fetch('/api/settings');
    const j=await r.json();
    mhMsg.textContent=j.magic_hour_configured ? 'Magic Hour key saved \u2014 video uses Magic Hour.' : 'No Magic Hour key \u2014 using free daily video.';
  }catch(e){}
}
mhSave.addEventListener('click', async function(){
  mhMsg.textContent='Saving...';
  try{
    const r=await fetch('/api/settings/magic-hour-key',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:mhKey.value.trim()})});
    if(!r.ok){ mhMsg.textContent=await r.text(); return; }
    const j=await r.json();
    mhKey.value='';
    mhMsg.textContent=j.configured ? 'Magic Hour key saved \u2014 video uses Magic Hour.' : 'Magic Hour key cleared \u2014 using free daily video.';
  }catch(e){ mhMsg.textContent='Could not save key'; }
});
skillChipX.addEventListener('click', function(){
  if(current){ current.skill_id=null; saveChat(); }
  updateSkillChip();
});
personaBtn.addEventListener('click', function(){ const ids=Object.keys(PERSONAS); const next=ids[(ids.indexOf(currentPersona)+1)%ids.length]; applyPersona(next); showToast('Persona: '+PERSONAS[currentPersona]); });
function setMode(m){
  mode=m;
  modeAgent.classList.toggle('active', m==='agent');
  modeChat.classList.toggle('active', m==='chat');
  if(m==='agent'){
    refreshWorkspace();
    showToast('Agent mode on — I can run commands and edit files in the workspace');
  }
}
modeAgent.addEventListener('click', function(){setMode('agent');});
modeChat.addEventListener('click', function(){setMode('chat');});
function setWsVisible(show){
  wsPanel.style.display=show?'block':'none';
  openWorkspace.classList.toggle('active', show);
  if(show) refreshWorkspace();
}
openWorkspace.addEventListener('click', function(){ setWsVisible(wsPanel.style.display==='none'); closeDrawer(); });
wsPanelClose.addEventListener('click', function(){ setWsVisible(false); });
wsRefresh.addEventListener('click', refreshWorkspace);
stopAgentBtn.addEventListener('click', stopAgent);

// ---------- Feature views: Notes, Deep Research, Cookbook, Library ----------
// One full-screen overlay view; each feature renders its own UI into fvBody.
function openFeature(title){
  fvTitle.textContent=title;
  fvBody.innerHTML='';
  featureView.classList.add('show');
}
function closeFeature(){ featureView.classList.remove('show'); }
fvClose.addEventListener('click', closeFeature);
openNotes.addEventListener('click', function(){ closeDrawer(); openNotesView(); });
openResearch.addEventListener('click', function(){ closeDrawer(); openResearchView(); });
openCookbook.addEventListener('click', function(){ closeDrawer(); openCookbookView(); });
openLibrary.addEventListener('click', function(){ closeDrawer(); openLibraryView(); });
openBrowser.addEventListener('click', function(){ closeDrawer(); openBrowserView(); });
openCalendar.addEventListener('click', function(){ closeDrawer(); openCalendarView(); });
openPdf.addEventListener('click', function(){ closeDrawer(); openPdfView(); });
chatPdfBtn.addEventListener('click', downloadChatPdf);
readAloudSide.addEventListener('click', function(){ closeDrawer(); speakLatest(); });

// Links tapped in chat and research sources open in the in-app browser.
// Relative links (downloads, internal files) are left alone.
document.addEventListener('click', function(e){
  const a=e.target && e.target.closest ? e.target.closest('a[href]') : null;
  if(!a) return;
  const href=a.getAttribute('href');
  if(href && /^https?:\/\//i.test(href)){
    e.preventDefault();
    openLinkInBrowser(href);
  }
});

// Notes: create, list, open/edit, delete. Stored in notes.json on the server.
let notes=[], editingNoteId=null;
async function openNotesView(){
  openFeature('Notes'); editingNoteId=null;
  fvBody.innerHTML='<p class="hint">Your notes, saved on the phone.</p>'
    +'<input id="noteTitle" class="fld" placeholder="Title">'
    +'<textarea id="noteBody" class="fld" rows="5" placeholder="Write your note..."></textarea>'
    +'<div class="rowflex"><button id="noteSave" class="btn" type="button">Save note</button><button id="noteCancel" class="btn" type="button" style="display:none;background:#333">Cancel</button></div>'
    +'<div id="noteList"></div>';
  document.getElementById('noteSave').addEventListener('click', saveNote);
  document.getElementById('noteCancel').addEventListener('click', function(){
    editingNoteId=null;
    document.getElementById('noteTitle').value='';
    document.getElementById('noteBody').value='';
    document.getElementById('noteCancel').style.display='none';
  });
  await loadNotes();
}
async function loadNotes(){
  try{ const r=await fetch('/api/notes'); notes=await r.json(); }catch(e){ notes=[]; }
  renderNoteList();
}
function renderNoteList(){
  const list=document.getElementById('noteList'); if(!list) return;
  list.innerHTML='';
  if(!notes.length){ const p=document.createElement('p'); p.className='hint'; p.textContent='No notes yet.'; list.appendChild(p); return; }
  for(const n of notes){
    const row=document.createElement('div'); row.className='listrow';
    const info=document.createElement('div'); info.className='info';
    const nm=document.createElement('div'); nm.className='nm'; nm.textContent=n.title||'Untitled';
    const sub=document.createElement('div'); sub.className='sub2'; sub.textContent=(n.body||'').slice(0,120);
    info.appendChild(nm); info.appendChild(sub);
    info.addEventListener('click', function(){
      editingNoteId=n.id;
      document.getElementById('noteTitle').value=n.title;
      document.getElementById('noteBody').value=n.body;
      document.getElementById('noteCancel').style.display='inline-block';
    });
    const del=document.createElement('button'); del.className='mini del'; del.textContent='x'; del.title='Delete note';
    del.addEventListener('click', async function(){
      await fetch('/api/notes/'+encodeURIComponent(n.id),{method:'DELETE'});
      if(editingNoteId===n.id){
        editingNoteId=null;
        document.getElementById('noteTitle').value='';
        document.getElementById('noteBody').value='';
        document.getElementById('noteCancel').style.display='none';
      }
      await loadNotes();
    });
    row.appendChild(info); row.appendChild(del); list.appendChild(row);
  }
}
async function saveNote(){
  const title=document.getElementById('noteTitle').value.trim();
  const body=document.getElementById('noteBody').value.trim();
  if(!title && !body){ showToast('Write something first'); return; }
  let r;
  if(editingNoteId){
    r=await fetch('/api/notes/'+encodeURIComponent(editingNoteId),{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:title, body:body})});
  } else {
    r=await fetch('/api/notes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:title, body:body})});
  }
  if(!r.ok){ showToast('Could not save note ('+r.status+')'); return; }
  editingNoteId=null;
  document.getElementById('noteTitle').value='';
  document.getElementById('noteBody').value='';
  document.getElementById('noteCancel').style.display='none';
  showToast('Note saved'); await loadNotes();
}

// Deep Research: iterative engine (plan -> search the web -> read pages ->
// synthesize -> sourced report), streamed as progress events. Mirrors the
// Odysseus IterResearch loop; the old one-call version is gone.
let rsRunId='', rsReader=null;
async function openResearchView(){
  openFeature('Deep Research');
  fvBody.innerHTML='<div class="rowflex" style="margin-bottom:8px">'
    +'<button id="rsTabNew" class="btn" type="button">Research</button>'
    +'<button id="rsTabHist" class="btn" type="button" style="background:#333">History <span id="rsHistCount"></span></button>'
    +'<button id="rsTabActive" class="btn" type="button" style="background:#333;display:none">Active <span id="rsActiveCount"></span></button>'
    +'</div>'
    +'<div id="rsNewTab">'
    +'<p class="hint">Real research: the model plans, searches the web, reads pages, and writes a sourced report. Works best with a Gemini model. The local model is too small for this.</p>'
    +'<label class="hint" for="rsEndpoint">Endpoint</label>'
    +'<select id="rsEndpoint" class="fld"></select>'
    +'<label class="hint" for="rsModel">Model for this research</label>'
    +'<select id="rsModel" class="fld"></select>'
    +'<p class="hint" id="rsModelNote"></p>'
    +'<label class="hint" for="rsRounds">Rounds</label>'
    +'<select id="rsRounds" class="fld"><option value="1">1 round</option><option value="2">2 rounds</option><option value="3">3 rounds</option><option value="5">5 rounds</option><option value="8">8 rounds</option></select>'
    +'<label class="hint" for="rsFormat">Report format</label>'
    +'<select id="rsFormat" class="fld"><option value="full">Full report</option><option value="brief">Brief summary</option><option value="bullets">Bullet points</option></select>'
    +'<label class="hint" for="rsEngine">Search engine</label>'
    +'<select id="rsEngine" class="fld"><option value="searxng">SearXNG</option><option value="bing">Bing</option><option value="google">Google</option><option value="duckduckgo">DuckDuckGo</option><option value="marginalia">Marginalia</option><option value="brave">Brave</option></select>'
    +'<p class="hint">SearXNG is the default — a dedicated instance running on this phone just for Mawaxi AI. Google is listed but Google blocks this phone, so it falls back to Bing. Marginalia is an independent index that works. DuckDuckGo is often blocked and falls back to Bing; Brave needs an API key and falls back to Bing.</p>'
    +'<input id="rsTopic" class="fld" placeholder="Topic to research...">'
    +'<button id="rsGo" class="btn" type="button">Research</button>'
    +'<button id="rsCancel" class="btn" type="button" style="display:none">Cancel</button>'
    +'</div>'
    +'<div id="rsHistTab" style="display:none">'
    +'<p class="hint" id="rsHistHead">Research reports saved in your library.</p>'
    +'<input id="rsHistSearch" class="fld" placeholder="Search research...">'
    +'<div class="rowflex"><button id="rsHistSort" class="btn" type="button" style="background:#333">Recent</button>'
    +'<button id="rsHistSelect" class="btn" type="button" style="background:#333">Select</button>'
    +'<button id="rsHistDelSel" class="btn" type="button" style="display:none;background:#7a2e2e">Delete selected</button></div>'
    +'<div class="rowflex" id="rsHistChips" style="flex-wrap:wrap"></div>'
    +'<div id="rsHistList"></div>'
    +'</div>'
    +'<div id="rsActiveTab" style="display:none">'
    +'<p class="hint" style="font-weight:600;color:var(--text);margin:0">Active</p>'
    +'<p class="hint" style="margin:0 0 8px">Queued and running research</p>'
    +'<div id="rsActiveList"></div>'
    +'</div>';
  await buildResearchControls();
  document.getElementById('rsGo').addEventListener('click', runResearch);
  document.getElementById('rsCancel').addEventListener('click', function(){ rsCancelRun(rsRunId); });
  document.getElementById('rsTabNew').addEventListener('click', function(){ showRsTab('new'); });
  document.getElementById('rsTabHist').addEventListener('click', function(){ showRsTab('hist'); });
  document.getElementById('rsTabActive').addEventListener('click', function(){ showRsTab('active'); });
  document.getElementById('rsHistSearch').addEventListener('input', function(){ rsHistQuery=this.value; renderResearchHistory(); });
  document.getElementById('rsHistSort').addEventListener('click', function(){ rsHistSortNewest=!rsHistSortNewest; this.textContent=rsHistSortNewest?'Recent':'Oldest'; renderResearchHistory(); });
  document.getElementById('rsHistSelect').addEventListener('click', function(){ rsHistSelectMode=!rsHistSelectMode; rsHistSelected.clear(); updateRsSelectUI(); renderResearchHistory(); });
  document.getElementById('rsHistDelSel').addEventListener('click', deleteSelectedResearch);
  await loadResearchHistory();
  for(const k in rsActiveRuns){ const r=rsActiveRuns[k]; r.elCard=null; r.elLog=null; r.elPhase=null; r.elFill=null; r.elRound=null; r.elSrc=null; r.elVis=null; r.elGlyph=null; r.elTime=null; r.elCancel=null; r.elView=null; r.elReport=null; r.elDetail=null; rsCreateActiveCard(r); }
  rsUpdateActiveTab(); rsEnsureTick();
}
// Deep Research options (endpoint, model, rounds, format, search engine).
// Defaults copy Odysseus: endpoint/model = the chat's selected ones
// (Odysseus research runs on the session's selected model); rounds = 8
// (DeepResearcher max_rounds default — the stop-decision usually ends a
// run after 2 rounds); format = full report (Odysseus has no format
// selector, it always writes the full structured report); engine = Bing
// (Odysseus defaults to SearXNG, self-hosted — not available on this
// phone; Bing is the closest working keyless equivalent). Puter models
// are excluded — Puter runs in the browser and cannot drive a
// server-side research loop. Last-used choices persist in localStorage
// and override these defaults.
function rsLoadOpts(){
  try{ return JSON.parse(localStorage.getItem('mawaxi-rs-opts')||'{}'); }catch(e){ return {}; }
}
function rsSaveOpts(){
  const o={};
  const ep=document.getElementById('rsEndpoint'); if(ep) o.endpoint=ep.value;
  const m=document.getElementById('rsModel'); if(m) o.model=m.value;
  const r=document.getElementById('rsRounds'); if(r) o.rounds=r.value;
  const f=document.getElementById('rsFormat'); if(f) o.format=f.value;
  const e=document.getElementById('rsEngine'); if(e) o.engine=e.value;
  try{ localStorage.setItem('mawaxi-rs-opts', JSON.stringify(o)); }catch(err){}
  return o;
}
async function buildResearchControls(){
  const saved=rsLoadOpts();
  const epSel=document.getElementById('rsEndpoint'); if(!epSel) return;
  let ollamaModels=[];
  try{ const r=await fetch('/api/models'); const j=await r.json(); ollamaModels=(j.models&&j.models.length)?j.models:[DEFAULT_MODEL]; }catch(e){ ollamaModels=[DEFAULT_MODEL]; }
  epSel.innerHTML='';
  const oLocal=document.createElement('option'); oLocal.value='local'; oLocal.textContent='Local Ollama'; epSel.appendChild(oLocal);
  for(const ep of endpoints){ const o=document.createElement('option'); o.value=ep.id; o.textContent=ep.name; epSel.appendChild(o); }
  // Default endpoint: saved choice, else the chat model's endpoint
  // (Gemini if chat is on Puter), else the first endpoint.
  let defEp='';
  if(saved.endpoint && (saved.endpoint==='local' || endpoints.some(function(e){ return e.id===saved.endpoint; }))) defEp=saved.endpoint;
  else if(activePuterModel){
    const gem=endpoints.find(function(e){ return (e.provider||'').toLowerCase()==='gemini'; });
    defEp=gem?gem.id:(endpoints.length?endpoints[0].id:'local');
  } else if(activeEndpointId){ defEp=activeEndpointId; }
  else defEp='local';
  epSel.value=defEp;
  const rSel=document.getElementById('rsRounds'); if(rSel) rSel.value=saved.rounds||'8';
  const fSel=document.getElementById('rsFormat'); if(fSel) fSel.value=saved.format||'full';
  const eSel=document.getElementById('rsEngine'); if(eSel) eSel.value=saved.engine||'searxng';
  buildResearchModelSelect(ollamaModels, saved);
  epSel.addEventListener('change', function(){ buildResearchModelSelect(ollamaModels, {}); rsSaveOpts(); });
  if(rSel) rSel.addEventListener('change', rsSaveOpts);
  if(fSel) fSel.addEventListener('change', rsSaveOpts);
  if(eSel) eSel.addEventListener('change', rsSaveOpts);
  const mSel=document.getElementById('rsModel');
  if(mSel) mSel.addEventListener('change', rsSaveOpts);
}
// Model picker inside Deep Research, filtered to the chosen endpoint
// (Local Ollama shows local models). Same values as the chat picker
// (endpoint:<id>:<model> for endpoints, plain name for local), no Puter.
function buildResearchModelSelect(ollamaModels, saved){
  const sel=document.getElementById('rsModel'); if(!sel) return;
  const epSel=document.getElementById('rsEndpoint');
  const epVal=epSel?epSel.value:'local';
  const note=document.getElementById('rsModelNote');
  sel.innerHTML='';
  let def='';
  if(epVal==='local'){
    const g=document.createElement('optgroup'); g.label='Local Ollama';
    for(const m of ollamaModels){ const o=document.createElement('option'); o.value=m; o.textContent=m; g.appendChild(o); }
    sel.appendChild(g);
    if(saved&&saved.model&&ollamaModels.indexOf(saved.model)>=0) def=saved.model;
    else if(ollamaModels.indexOf(currentModel)>=0) def=currentModel;
    else def=ollamaModels[0];
    if(note) note.textContent='The local model is too small for Deep Research — pick an endpoint above for a Gemini model.';
  } else {
    const ep=endpoints.find(function(e){ return e.id===epVal; });
    if(ep){
      const list=(ep.models&&ep.models.length)?ep.models:[ep.model];
      const g=document.createElement('optgroup'); g.label=ep.name;
      for(const m of list){ if(!m) continue; const o=document.createElement('option'); o.value='endpoint:'+ep.id+':'+m; o.textContent=ep.name+' · '+m; g.appendChild(o); }
      sel.appendChild(g);
      if(saved&&saved.model&&saved.model.indexOf('endpoint:'+ep.id+':')===0) def=saved.model;
      else def='endpoint:'+ep.id+':'+ep.model;
      if(note) note.textContent=activePuterModel?('Puter models cannot be used for research — using your '+ep.name+' model instead.'):'';
    }
  }
  let found=false;
  for(const o of sel.options){ if(o.value===def){ found=true; break; } }
  sel.value=found?def:(sel.options[0]?sel.options[0].value:'');
}
// Deep Research History: saved reports, stored server-side in
// research_history.json (survives restarts). Two tabs like Odysseus:
// Research (new run) and History (saved reports, newest first).
let rsHistory=[], rsHistQuery='', rsHistFilter='all', rsHistSortNewest=true, rsHistSelectMode=false, rsHistSelected=new Set();
function rsFmtDur(s){ s=Math.max(0,Math.floor(s||0)); return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }
function showRsTab(which){
  const n=document.getElementById('rsNewTab'), h=document.getElementById('rsHistTab'), a=document.getElementById('rsActiveTab');
  const bn=document.getElementById('rsTabNew'), bh=document.getElementById('rsTabHist'), ba=document.getElementById('rsTabActive');
  if(!n||!h||!a) return;
  if(which!=='active') rsPruneDoneActive();
  n.style.display=which==='new'?'':'none';
  h.style.display=which==='hist'?'':'none';
  a.style.display=which==='active'?'':'none';
  if(bn) bn.style.background=which==='new'?'':'#333';
  if(bh) bh.style.background=which==='hist'?'':'#333';
  if(ba) ba.style.background=which==='active'?'':'#333';
  if(which==='hist') loadResearchHistory();
  if(which==='active') renderRsActive();
}
async function loadResearchHistory(){
  try{
    const r=await fetch('/api/research/history'); if(!r.ok) return;
    rsHistory=await r.json();
  }catch(e){ return; }
  const c=document.getElementById('rsHistCount'); if(c) c.textContent=rsHistory.length?'('+rsHistory.length+')':'';
  renderResearchHistory();
}
function rsHistFiltered(){
  let list=rsHistory.slice();
  if(rsHistFilter==='standard') list=list.filter(function(e){ return e.status==='completed'; });
  else if(rsHistFilter==='failed') list=list.filter(function(e){ return e.status==='failed'; });
  if(rsHistQuery){ const q=rsHistQuery.toLowerCase(); list=list.filter(function(e){ return (e.topic||'').toLowerCase().indexOf(q)>=0; }); }
  list.sort(function(a,b){ return rsHistSortNewest?((b.created_at||0)-(a.created_at||0)):((a.created_at||0)-(b.created_at||0)); });
  return list;
}
function renderResearchHistory(){
  const listEl=document.getElementById('rsHistList'); if(!listEl) return;
  const total=rsHistory.length;
  const std=rsHistory.filter(function(e){ return e.status==='completed'; }).length;
  const failed=total-std;
  const head=document.getElementById('rsHistHead');
  const shown=rsHistFiltered().length;
  if(head) head.textContent='Research '+shown+' of '+total+' — Completed research reports saved in your library.';
  const chips=document.getElementById('rsHistChips');
  if(chips){
    let h='<button class="btn rsChip" type="button" data-f="all" style="background:'+(rsHistFilter==='all'?'':'#333')+'">All '+total+'</button>'
      +'<button class="btn rsChip" type="button" data-f="standard" style="background:'+(rsHistFilter==='standard'?'':'#333')+'">Standard '+std+'</button>';
    if(failed>0) h+='<button class="btn rsChip" type="button" data-f="failed" style="background:'+(rsHistFilter==='failed'?'':'#333')+'">Failed '+failed+'</button>';
    chips.innerHTML=h;
    chips.querySelectorAll('.rsChip').forEach(function(b){ b.addEventListener('click', function(){ rsHistFilter=b.getAttribute('data-f'); renderResearchHistory(); }); });
  }
  const list=rsHistFiltered();
  if(!list.length){ listEl.innerHTML='<p class="hint">No saved research yet. Run a research above and it will appear here.</p>'; return; }
  listEl.innerHTML='';
  list.forEach(function(e){
    const row=document.createElement('div'); row.className='msg'; row.style.cursor='pointer'; row.setAttribute('data-id', e.id);
    let h='';
    if(rsHistSelectMode) h+='<input type="checkbox" class="rsHistCheck" data-id="'+e.id+'"'+(rsHistSelected.has(e.id)?' checked':'')+'> ';
    h+='<b>'+esc(e.topic)+'</b>';
    h+='<p class="hint">'+esc(rsFmtDur(e.duration_secs))+' · '+e.sources.length+' sources · '+esc(e.status||'completed')+'</p>';
    if(e.status==='failed'&&e.error) h+='<p style="color:#ff6b6b;margin:2px 0">'+esc(e.error)+'</p>';
    h+='<div class="rowflex"><button class="btn rsVis" type="button" data-id="'+e.id+'">VISUAL</button>'
      +'<button class="btn rsChatBtn" type="button" data-id="'+e.id+'">CHAT</button>'
      +'<button class="btn rsDlBtn" type="button" data-id="'+e.id+'">DOWNLOAD</button>'
      +'<button class="btn rsRowMenu" type="button" data-id="'+e.id+'" style="background:#333">...</button>'
      +'<button class="btn rsRowDel" type="button" data-id="'+e.id+'" style="display:none;background:#7a2e2e">Delete</button></div>';
    row.innerHTML=h;
    listEl.appendChild(row);
  });
  listEl.querySelectorAll('.rsHistCheck').forEach(function(cb){ cb.addEventListener('click', function(ev){ ev.stopPropagation(); if(cb.checked) rsHistSelected.add(cb.getAttribute('data-id')); else rsHistSelected.delete(cb.getAttribute('data-id')); updateRsSelectUI(); }); });
  listEl.querySelectorAll('.rsVis').forEach(function(b){ b.addEventListener('click', function(ev){ ev.stopPropagation(); visualResearchReport(b.getAttribute('data-id')); }); });
  listEl.querySelectorAll('.rsChatBtn').forEach(function(b){ b.addEventListener('click', function(ev){ ev.stopPropagation(); chatAboutResearch(b.getAttribute('data-id')); }); });
  listEl.querySelectorAll('.rsDlBtn').forEach(function(b){ b.addEventListener('click', function(ev){ ev.stopPropagation(); rsDownloadReport(b.getAttribute('data-id')); }); });
  listEl.querySelectorAll('.rsRowMenu').forEach(function(b){ b.addEventListener('click', function(ev){ ev.stopPropagation(); const row=b.closest('.msg'); const del=row?row.querySelector('.rsRowDel'):null; if(del) del.style.display=del.style.display==='none'?'':'none'; }); });
  listEl.querySelectorAll('.rsRowDel').forEach(function(b){ b.addEventListener('click', async function(ev){ ev.stopPropagation(); await deleteResearchReport(b.getAttribute('data-id')); }); });
  listEl.querySelectorAll('.msg').forEach(function(row){ row.addEventListener('click', function(){ openResearchReport(row.getAttribute('data-id')); }); });
}
function updateRsSelectUI(){
  const sel=document.getElementById('rsHistSelect'); if(sel) sel.textContent=rsHistSelectMode?'Done':'Select';
  const del=document.getElementById('rsHistDelSel'); if(del){ del.style.display=rsHistSelectMode?'':'none'; del.textContent='Delete selected'+(rsHistSelected.size?' ('+rsHistSelected.size+')':''); }
}
async function deleteResearchReport(id){
  await fetch('/api/research/history/'+encodeURIComponent(id),{method:'DELETE'});
  await loadResearchHistory();
}
async function deleteSelectedResearch(){
  const ids=Array.from(rsHistSelected);
  if(!ids.length) return;
  await fetch('/api/research/history/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:ids})});
  rsHistSelected.clear(); rsHistSelectMode=false; updateRsSelectUI();
  await loadResearchHistory();
}
async function openResearchReport(id){
  const r=await fetch('/api/research/history/'+encodeURIComponent(id)); if(!r.ok){ showToast('Report not found'); return; }
  const e=await r.json();
  const run={id:'rs-hist-'+e.id+'-'+Date.now(), topic:e.topic, model:e.model||'', endpoint:e.endpoint||'', totalRounds:e.rounds||1, format:e.format||'full', engine:e.engine||'bing', startedAt:Date.now()-((e.duration_secs||0)*1000), state:'done', phase:'writing', round:e.rounds||1, doneRounds:e.rounds||1, queries:0, sources:e.sources||[], logData:[], report:e.report||'', duration:rsFmtDur(e.duration_secs||0), error:''};
  run.histId=e.id;
  rsActiveRuns[run.id]=run;
  rsCreateActiveCard(run);
  if(run.elTime) run.elTime.textContent=run.duration;
  if(run.elVis) run.elVis.textContent=rsFmtClock(e.duration_secs||0);
  rsUpdateActiveTab();
  showRsTab('active');
  rsToggleReport(run);
}
// Print a research report as a COLOURED PDF. Browsers strip background
// colours in print by default, so every coloured element carries
// print-color-adjust: exact. Opens the coloured report in a new window
// and prints it — in the print dialog choose "Save as PDF".
function researchPrintDoc(e, imgUrl){
  const srcs=e.sources||[];
  const dur=rsFmtDur(e.duration_secs);
  const dom=function(u){ try{ return new URL(u).hostname.replace(/^www\./,''); }catch(err){ return u; } };
  const body=(e.report||'').replace(/\n## Sources[\s\S]*$/,'');
  const css='<style>'
  +'*{ -webkit-print-color-adjust:exact!important; print-color-adjust:exact!important; }'
  +'@page{size:A4;margin:12mm}'
  +'body{margin:0;background:#fbf9f4;color:#1a1817;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.7;font-size:15px}'
  +'.printHint{font-size:14px;color:#5a5651;text-align:center;padding:14px 12px 4px}'
  +'.printBtn{display:block;margin:6px auto 18px;background:#b8543a;color:#fff;border:none;border-radius:999px;padding:10px 24px;font-size:15px;font-weight:600;cursor:pointer}'
  +'.prHero{text-align:center;padding:28px 20px 18px;background:radial-gradient(60% 90% at 20% 0%,rgba(184,84,58,.12),transparent 60%),radial-gradient(55% 80% at 85% 10%,rgba(201,149,46,.12),transparent 60%),#fbf9f4;border-bottom:1px solid rgba(0,0,0,.08)}'
  +'.prLabel{text-transform:uppercase;letter-spacing:.28em;font-size:11px;font-weight:700;color:#b8543a;margin-bottom:12px}'
  +'.prHero h1{font-family:Charter,"Iowan Old Style",Georgia,serif;font-size:28px;font-weight:600;line-height:1.2;margin:0 auto;max-width:640px;color:#1a1817}'
  +'.prHeroImg{max-width:680px;margin:18px auto 0}'
  +'.prHeroImg img{width:100%;aspect-ratio:2/1;object-fit:cover;border:1px solid rgba(0,0,0,.16);border-radius:8px;display:block}'
  +'.prStats{display:flex;justify-content:center;gap:22px;flex-wrap:wrap;padding:12px 16px;background:#ffffff;border-top:1px solid rgba(0,0,0,.08);border-bottom:1px solid rgba(0,0,0,.08);font-size:13px;color:#5a5651}'
  +'.prStats b{color:#1a1817;font-weight:700}'
  +'.prContent{max-width:680px;margin:0 auto;padding:28px 20px 20px;counter-reset:prsec}'
  +'.prContent h2{counter-increment:prsec;font-family:Charter,"Iowan Old Style",Georgia,serif;font-size:22px;font-weight:600;margin:32px 0 12px;padding-bottom:8px;border-bottom:1px solid transparent;border-image:linear-gradient(90deg,#b8543a 0%,transparent 65%) 1;color:#1a1817}'
  +'.prContent h2::before{content:counter(prsec,decimal-leading-zero);display:block;margin-bottom:6px;color:#b8543a;font-family:system-ui,sans-serif;font-size:11px;font-weight:700;letter-spacing:.12em}'
  +'.prContent h3{font-family:Charter,"Iowan Old Style",Georgia,serif;font-size:18px;font-weight:600;margin:24px 0 8px;color:#1a1817}'
  +'.prContent p{margin:0 0 14px;color:#1a1817}'
  +'.prContent a{color:#b8543a}'
  +'.prContent ul,.prContent ol{margin:0 0 14px;padding-left:20px}'
  +'.prContent li{margin-bottom:5px}'
  +'.prContent blockquote{margin:0 0 14px;padding:10px 16px;border-left:3px solid #b8543a;background:rgba(184,84,58,.06);border-radius:0 8px 8px 0;color:#5a5651}'
  +'.prSources{max-width:680px;margin:0 auto;padding:0 20px 32px}'
  +'.prSources h3{font-family:Charter,"Iowan Old Style",Georgia,serif;font-size:18px;font-weight:600;margin:0 0 12px;color:#1a1817}'
  +'.prSrc{display:block;background:#ffffff;border:1px solid rgba(0,0,0,.08);border-left:3px solid #b8543a;border-radius:10px;padding:12px 16px;margin-bottom:10px;text-decoration:none}'
  +'.prSrcTitle{color:#1a1817;font-weight:600;font-size:14px;line-height:1.4}'
  +'.prSrcDom{color:#8a8580;font-size:12px;margin-top:2px}'
  +'@media print{.printHint,.printBtn{display:none!important}.prHero{break-after:avoid}.prContent h2,.prSources h3{break-after:avoid}.prSrc{break-inside:avoid}}'
  +'</style>';
  let h='<div class="prHero"><div class="prLabel">Deep Research</div><h1>'+esc(e.topic)+'</h1></div>';
  if(imgUrl){ h+='<div class="prHeroImg"><img src="'+esc(imgUrl)+'" alt=""></div>'; }
  h+='<div class="prStats">'
    +'<span><b>'+srcs.length+'</b> sources</span>'
    +'<span><b>'+esc(dur)+'</b></span>'
    +(e.model?'<span>'+esc(e.model)+'</span>':'')
    +(e.endpoint?'<span>'+esc(e.endpoint)+'</span>':'')
    +'</div>';
  h+='<div class="prContent">'+md(body||'No report saved.')+'</div>';
  if(srcs.length){
    h+='<div class="prSources"><h3>Sources</h3>'+srcs.map(function(x){
      return '<a class="prSrc" href="'+esc(x.url)+'" target="_blank"><div class="prSrcTitle">'+esc(x.title||x.url)+'</div><div class="prSrcDom">'+esc(dom(x.url))+'</div></a>';
    }).join('')+'</div>';
  }
  return '<!DOCTYPE html><html><head><meta charset="utf-8"><title>'+esc(e.topic)+' — Research report</title>'+css+'</head><body>'
    +'<p class="printHint">This is the coloured report. In the print dialog, choose \u201cSave as PDF\u201d to download it in colour.</p>'
    +'<button class="printBtn" type="button" onclick="window.print()">Save as PDF</button>'
    +h+'</body></html>';
}
async function printResearchReport(id){
  const r=await fetch('/api/research/history/'+encodeURIComponent(id));
  if(!r.ok){ showToast('Report not found'); return; }
  const e=await r.json();
  let imgUrl='';
  try{
    const ir=await fetch('/api/image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:'Vivid editorial illustration for a research report about '+e.topic})});
    if(ir.ok){ const ij=await ir.json(); if(ij.url) imgUrl=ij.url; }
  }catch(err){}
  const w=window.open('', '_blank');
  if(!w){ showToast('Allow pop-ups to save the PDF'); return; }
  w.document.write(researchPrintDoc(e, imgUrl));
  w.document.close();
  const go=function(){ try{ w.focus(); w.print(); }catch(err){} };
  const imgs=w.document.images;
  if(imgs.length){
    let n=0, done=false;
    const one=function(){ if(done) return; if(++n>=imgs.length){ done=true; setTimeout(go, 300); } };
    for(const im of imgs){ if(im.complete) one(); else { im.onload=one; im.onerror=one; } }
    setTimeout(function(){ if(!done){ done=true; go(); } }, 5000);
  } else setTimeout(go, 300);
}
async function visualResearchReport(id){
  const r=await fetch('/api/research/history/'+encodeURIComponent(id)); if(!r.ok){ showToast('Report not found'); return; }
  const e=await r.json();
  openFeature('Research visual');
  fvBody.innerHTML='<p class="hint">Loading visual...</p>';
  let imgUrl='';
  try{
    const ir=await fetch('/api/image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:'Vivid editorial illustration for a research report about '+e.topic})});
    if(ir.ok){ const ij=await ir.json(); if(ij.url) imgUrl=ij.url; }
  }catch(err){}
  const srcs=e.sources||[];
  const dur=rsFmtDur(e.duration_secs);
  const dom=function(u){ try{ return new URL(u).hostname.replace(/^www\./,''); }catch(err){ return u; } };
  const body=(e.report||'').replace(/\n## Sources[\s\S]*$/,'');
  const css='<style>'
  +'.rsVisual{--rv-bg:#131214;--rv-surface:#1c1a1e;--rv-surface-alt:#25232a;--rv-border:rgba(255,255,255,.07);--rv-border-strong:rgba(255,255,255,.16);--rv-text:#ece8e2;--rv-dim:#a8a39c;--rv-muted:#6f6b66;--rv-accent:#e88f73;--rv-accent-light:#f4ad95;--rv-accent-bg:rgba(232,143,115,.09);--rv-gold:#e8c05a;--rv-gold-bg:rgba(232,192,90,.09);--rv-aurora-a:rgba(232,143,115,.13);--rv-aurora-b:rgba(232,192,90,.09);--rv-aurora-c:rgba(125,180,224,.10);position:relative;background:var(--rv-bg);color:var(--rv-text);border-radius:12px;overflow:hidden;flex-shrink:0;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.75;font-size:16px;margin:-8px}'
  +'@media (prefers-color-scheme: light){.rsVisual{--rv-bg:#fbf9f4;--rv-surface:#ffffff;--rv-surface-alt:#f1ede4;--rv-border:rgba(0,0,0,.08);--rv-border-strong:rgba(0,0,0,.16);--rv-text:#1a1817;--rv-dim:#5a5651;--rv-muted:#8a8580;--rv-accent:#b8543a;--rv-accent-light:#d97a5e;--rv-accent-bg:rgba(184,84,58,.06);--rv-gold:#c9952e;--rv-gold-bg:rgba(201,149,46,.09);--rv-aurora-a:rgba(184,84,58,.10);--rv-aurora-b:rgba(201,149,46,.08);--rv-aurora-c:rgba(64,98,128,.07)}}'
  +'.rsVisual::before{content:"";position:absolute;inset:-20vh -20vw;z-index:0;background:radial-gradient(40vw 50vh at 18% 22%,var(--rv-aurora-a) 0%,transparent 60%),radial-gradient(45vw 55vh at 82% 12%,var(--rv-aurora-b) 0%,transparent 65%),radial-gradient(55vw 60vh at 50% 88%,var(--rv-aurora-c) 0%,transparent 70%);filter:blur(20px);animation:rvDrift 28s ease-in-out infinite alternate;pointer-events:none}'
  +'@keyframes rvDrift{0%{transform:translate3d(0,0,0) scale(1)}50%{transform:translate3d(2vw,-1vh,0) scale(1.04)}100%{transform:translate3d(-1vw,1.5vh,0) scale(1.02)}}'
  +'.rsVisual>*{position:relative;z-index:1}'
  +'.rvProg{height:3px;background:transparent}'
  +'.rvProgBar{height:100%;width:0;background:var(--rv-accent);box-shadow:0 0 10px var(--rv-accent);transition:width .08s linear}'
  +'.rvHero{text-align:center;padding:2.2rem 1.4rem 1.6rem;position:relative}'
  +'.rvHero::after{content:"";position:absolute;left:50%;bottom:0;width:min(60%,320px);height:1px;transform:translateX(-50%);background:linear-gradient(90deg,transparent,var(--rv-border-strong),transparent)}'
  +'.rvLabel{text-transform:uppercase;letter-spacing:.28em;font-size:.68rem;font-weight:600;color:var(--rv-accent);margin-bottom:1rem}'
  +'.rvHero h1{font-family:Charter,"Iowan Old Style",Georgia,serif;font-size:clamp(1.7rem,4.5vw,2.4rem);font-weight:600;line-height:1.15;letter-spacing:-.02em;margin:0 auto;max-width:640px;color:var(--rv-text)}'
  +'.rvHeroImg{max-width:760px;margin:0 auto;padding:1.4rem 1.2rem 0}'
  +'.rvHeroImg img{width:100%;aspect-ratio:2/1;object-fit:cover;border:1px solid var(--rv-border-strong);border-radius:8px;box-shadow:0 4px 24px rgba(0,0,0,.25);display:block}'
  +'.rvHeroImg figcaption{display:flex;justify-content:space-between;gap:1rem;padding:.5rem .25rem 0;color:var(--rv-muted);font-size:.68rem}'
  +'.rvStats{display:flex;justify-content:center;gap:1.4rem;flex-wrap:wrap;padding:.9rem 1.2rem;background:var(--rv-surface);border-top:1px solid var(--rv-border);border-bottom:1px solid var(--rv-border);font-size:.82rem;color:var(--rv-dim);margin-top:1.4rem}'
  +'.rvStat{display:flex;align-items:center;gap:.35rem}'
  +'.rvStat b{color:var(--rv-text);font-weight:600}'
  +'.rvContent{max-width:720px;margin:0 auto;padding:2rem 1.4rem 2.5rem;counter-reset:rvsec}'
  +'.rvContent h2{counter-increment:rvsec;font-family:Charter,"Iowan Old Style",Georgia,serif;font-size:clamp(1.4rem,2.4vw,1.7rem);font-weight:600;margin:2.4rem 0 .9rem;padding-bottom:.5rem;border-bottom:1px solid transparent;border-image:linear-gradient(90deg,var(--rv-accent) 0%,transparent 65%) 1;letter-spacing:-.02em;line-height:1.2;color:var(--rv-text)}'
  +'.rvContent h2::before{content:counter(rvsec,decimal-leading-zero);display:block;margin-bottom:.45rem;color:var(--rv-accent);font-family:system-ui,sans-serif;font-size:.65rem;font-weight:700;letter-spacing:.12em}'
  +'.rvContent h3{font-family:Charter,"Iowan Old Style",Georgia,serif;font-size:1.15rem;font-weight:600;margin:1.8rem 0 .5rem;color:var(--rv-text)}'
  +'.rvContent p{margin:0 0 1rem;color:var(--rv-text)}'
  +'.rvContent a{color:var(--rv-accent)}'
  +'.rvContent ul,.rvContent ol{margin:0 0 1rem;padding-left:1.3rem}'
  +'.rvContent li{margin-bottom:.35rem}'
  +'.rvContent blockquote{margin:0 0 1rem;padding:.6rem 1rem;border-left:3px solid var(--rv-accent);background:var(--rv-accent-bg);border-radius:0 8px 8px 0;color:var(--rv-dim)}'
  +'.rvSources{max-width:720px;margin:0 auto;padding:0 1.4rem 2.5rem}'
  +'.rvSources h3{font-family:Charter,"Iowan Old Style",Georgia,serif;font-size:1.15rem;font-weight:600;margin:0 0 .9rem;color:var(--rv-text)}'
  +'.rvSrc{display:block;background:var(--rv-surface);border:1px solid var(--rv-border);border-left:3px solid var(--rv-accent);border-radius:10px;padding:.8rem 1rem;margin-bottom:.6rem;text-decoration:none}'
  +'.rvSrcTitle{color:var(--rv-text);font-weight:600;font-size:.92rem;line-height:1.4}'
  +'.rvSrcDom{color:var(--rv-muted);font-size:.72rem;margin-top:.15rem}'
  +'.rvBack{margin:0 1.4rem 2rem}'
  +'</style>';
  let h=css+'<div class="rsVisual"><div class="rvProg"><div class="rvProgBar" id="rvProgBar"></div></div>'
    +'<div class="rvHero"><div class="rvLabel">Deep Research</div><h1>'+esc(e.topic)+'</h1></div>';
  if(imgUrl){ h+='<figure class="rvHeroImg"><img src="'+esc(imgUrl)+'" alt=""><figcaption><span>'+esc(e.topic)+'</span><span>'+srcs.length+' sources</span></figcaption></figure>'; }
  h+='<div class="rvStats">'
    +'<span class="rvStat"><b>'+srcs.length+'</b>&nbsp;sources</span>'
    +'<span class="rvStat"><b>'+esc(dur)+'</b></span>'
    +(e.model?'<span class="rvStat">'+esc(e.model)+'</span>':'')
    +(e.endpoint?'<span class="rvStat">'+esc(e.endpoint)+'</span>':'')
    +'</div>';
  h+='<div class="rvContent">'+md(body||'No report saved.')+'</div>';
  if(srcs.length){
    h+='<div class="rvSources"><h3>Sources</h3>'+srcs.map(function(x){
      return '<a class="rvSrc" href="'+esc(x.url)+'" target="_blank"><div class="rvSrcTitle">'+esc(x.title||x.url)+'</div><div class="rvSrcDom">'+esc(dom(x.url))+'</div></a>';
    }).join('')+'</div>';
  }
  h+='<div class="rowflex" style="margin:0 1.4rem 2rem"><button id="rsVisBack" class="btn" type="button">Back to Deep Research</button><button id="rsVisPdf" class="btn" type="button">Save as PDF (colour)</button></div></div>';
  fvBody.innerHTML=h;
  document.getElementById('rsVisBack').addEventListener('click', openResearchView);
  document.getElementById('rsVisPdf').addEventListener('click', function(){ printResearchReport(id); });
  const bar=document.getElementById('rvProgBar');
  const sc=function(){ const max=fvBody.scrollHeight-fvBody.clientHeight; if(max>0){ bar.style.width=((fvBody.scrollTop/max)*100)+'%'; } };
  if(window._rvScroll){ fvBody.removeEventListener('scroll', window._rvScroll); }
  window._rvScroll=sc;
  fvBody.addEventListener('scroll', sc);
  sc();
}
async function chatAboutResearch(id){
  const r=await fetch('/api/research/history/'+encodeURIComponent(id)); if(!r.ok){ showToast('Report not found'); return; }
  const e=await r.json();
  const body=(e.report||'').replace(/\n## Sources[\s\S]*$/,'');
  const summary=body.length>1200?body.slice(0,1200)+'…':body;
  const c=newChatObj();
  c.title='Research: '+e.topic;
  c.messages=[{role:'assistant', content:'Here is what I found on "'+e.topic+'":\n\n'+(summary||'No report text was saved.')+'\n\nSources: '+e.sources.length+' — see Deep Research → History for the links. Ask me anything about this research.'}];
  current=c;
  try{ await saveChat(); }catch(err){}
  closeFeature(); renderMessages(); renderChatList();
}
function resetRsButtons(){ const g=document.getElementById('rsGo'), c=document.getElementById('rsCancel'); if(g) g.style.display=''; if(c) c.style.display='none'; }
// Research progress is rendered into the Active-tab card for the run,
// not a plain text log. One structured row per action, Odysseus style.
function handleRsEvent(ev, run){
  if(!run) return;
  if(ev.type==='started'){
    const sid=ev.run_id||'';
    if(sid&&sid!==run.id){ delete rsActiveRuns[run.id]; run.id=sid; rsActiveRuns[sid]=run; if(run.elCard) run.elCard.id='rsCard-'+sid; }
    rsRunId=sid||run.id;
  }
  else if(ev.type==='plan'){
    run.phase='planning';
    const n=(ev.plan&&ev.plan.sub_questions)?ev.plan.sub_questions.length:0;
    rsAddLogRow(run,'Plan','Plan created · '+n+' sub-questions',(ev.plan&&ev.plan.success_criteria)||'');
    rsUpdateActiveCard(run);
  }
  else if(ev.type==='queries'){
    run.round=ev.round||run.round; run.phase='searching'; run.queries=(ev.queries||[]).length;
    (ev.queries||[]).forEach(function(q){ rsAddLogRow(run,'Round '+run.round,'web_search: '+q+' planner'); });
    rsUpdateActiveCard(run);
  }
  else if(ev.type==='search'){
    if(ev.error){ rsAddLogRow(run,'Round '+run.round,'search failed: '+ev.error); }
    else { run.phase='reading'; rsUpdateActiveCard(run); }
  }
  else if(ev.type==='note'){ rsAddLogRow(run,'Note',(ev.message||'').slice(0,220)); }
  else if(ev.type==='source'){
    run.sources.push({title:ev.title||ev.url, url:ev.url});
    run.phase='reading';
    rsAddLogRow(run,'Round '+run.round,'fetch: '+(ev.title||ev.url));
    rsUpdateActiveCard(run);
  }
  else if(ev.type==='synth'){
    run.doneRounds=Math.max(run.doneRounds, ev.round||run.round);
    run.phase='synthesising';
    rsAddLogRow(run,'Round '+run.round,'synthesised'+(ev.note?': '+ev.note:''));
    rsUpdateActiveCard(run);
  }
  else if(ev.type==='stop'){ rsAddLogRow(run,'Stop',ev.decision||''); }
  else if(ev.type==='report'){
    run.state='done'; run.phase='writing';
    run.report=ev.report||''; run.sources=ev.sources||run.sources;
    run.duration=ev.duration||rsFmtDur(Math.floor((Date.now()-run.startedAt)/1000));
    rsAddLogRow(run,'Done','Finished in '+run.duration+' · Saved to History');
    rsUpdateActiveCard(run); rsUpdateActiveTab(); rsStopTickIfIdle(); loadResearchHistory();
  }
  else if(ev.type==='error'){
    if(run.state==='running'){ run.state='error'; run.error=ev.message||'error'; rsAddLogRow(run,'Error',run.error); rsUpdateActiveCard(run); rsUpdateActiveTab(); rsStopTickIfIdle(); loadResearchHistory(); }
  }
  else if(ev.type==='cancelled'){
    if(run.state==='running'){ run.state='cancelled'; run.phase='cancelled'; rsAddLogRow(run,'Cancelled','Cancelled by user'); rsUpdateActiveCard(run); rsUpdateActiveTab(); rsStopTickIfIdle(); loadResearchHistory(); }
  }
}
// Deep Research Active view: running runs live in an "Active" tab like
// Odysseus — one card per run with a live timer, a status line, a visual
// panel (dark grid + scanning line + animated glyphs), a progress bar and
// a structured activity log. The old plain-text progress log is gone.
let rsActiveRuns={}, rsActiveTick=null, rsActiveSeq=0;
function rsFmtClock(s){ s=Math.max(0,Math.floor(s||0)); return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0'); }
function rsActiveCount(){ let n=0; for(const k in rsActiveRuns){ if(rsActiveRuns[k].state==='running') n++; } return n; }
function rsUpdateActiveTab(){
  const tab=document.getElementById('rsTabActive'), c=document.getElementById('rsActiveCount');
  if(!tab) return;
  const running=rsActiveCount();
  const any=Object.keys(rsActiveRuns).length>0;
  tab.style.display=any?'':'none';
  if(c) c.textContent=running?'('+running+')':'';
}
function rsPruneDoneActive(){
  let changed=false;
  for(const k in rsActiveRuns){ if(rsActiveRuns[k].state!=='running'){ const r=rsActiveRuns[k]; if(r.elCard) r.elCard.remove(); delete rsActiveRuns[k]; changed=true; } }
  if(changed){ renderRsActive(); rsUpdateActiveTab(); }
}
function rsGlyphs(seed){
  const ch='·×#≡/\\|◦○◆.·:;';
  let s='', x=(seed*2654435761)%4294967296;
  for(let i=0;i<28;i++){ x=(x*1664525+1013904223)%4294967296; s+=ch[x%ch.length]; if(i%7===6) s+=' '; }
  return s;
}
function rsTickActive(){
  const now=Date.now();
  for(const k in rsActiveRuns){
    const r=rsActiveRuns[k];
    if(r.state==='running'){
      const el=Math.floor((now-r.startedAt)/1000);
      if(r.elTime) r.elTime.textContent=rsFmtClock(el);
      if(r.elVis) r.elVis.textContent=rsFmtClock(el);
      if(r.elGlyph) r.elGlyph.textContent=rsGlyphs(el);
    }
  }
}
function rsEnsureTick(){ if(rsActiveTick) return; if(rsActiveCount()===0) return; rsActiveTick=setInterval(rsTickActive, 1000); }
function rsStopTickIfIdle(){ if(rsActiveCount()===0&&rsActiveTick){ clearInterval(rsActiveTick); rsActiveTick=null; } }
function rsPhaseText(r){
  if(r.state==='done') return 'Finished in '+r.duration+' · Saved to History';
  if(r.state==='cancelled') return 'Cancelled';
  if(r.state==='error') return 'Error: '+(r.error||'failed');
  const rn=r.round?('Round '+r.round+'/'+r.totalRounds+': '):'';
  if(r.phase==='planning') return 'Planning research strategy...';
  if(r.phase==='searching') return rn+'Searching ('+(r.queries||0)+' queries)';
  if(r.phase==='reading') return rn+'Reading '+r.sources.length+' sources';
  if(r.phase==='synthesising') return rn+'Synthesising';
  if(r.phase==='writing') return 'Writing report — '+r.sources.length+' sources';
  return 'Starting...';
}
function rsProgressPct(r){
  if(r.state==='done') return 100;
  if(r.state==='cancelled'||r.state==='error') return Math.round((r.doneRounds/Math.max(1,r.totalRounds))*100);
  let f=r.doneRounds;
  if(r.phase==='searching') f+=0.3; else if(r.phase==='reading') f+=0.5; else if(r.phase==='synthesising') f+=0.7; else if(r.phase==='writing') f+=0.9; else if(r.phase==='planning') f+=0.1;
  return Math.max(2, Math.min(99, Math.round((f/Math.max(1,r.totalRounds))*100)));
}
function rsUpdateActiveCard(r){
  if(r.elPhase) r.elPhase.innerHTML='<b>'+esc(rsPhaseText(r))+'</b>';
  if(r.elFill) r.elFill.style.width=rsProgressPct(r)+'%';
  if(r.elRound) r.elRound.textContent='round '+(r.round||1);
  if(r.elSrc) r.elSrc.textContent=r.sources.length+' sources';
  if(r.elCancel) r.elCancel.style.display=r.state==='running'?'':'none';
  if(r.elView) r.elView.style.display=(r.state==='done'&&r.report)?'':'none';
}
function rsAddLogRow(r, tag, text, sub){
  const tm=rsFmtClock((Date.now()-r.startedAt)/1000);
  r.logData.push({tag:tag, text:text, sub:sub||'', time:tm});
  if(r.logData.length>200) r.logData.shift();
  if(!r.elLog) return;
  rsAppendLogRow(r, r.logData[r.logData.length-1]);
  if(r.elLog.children.length>200) r.elLog.removeChild(r.elLog.firstChild);
}
function rsAppendLogRow(r, d){
  const row=document.createElement('div'); row.className='rsActRow';
  const t=document.createElement('span'); t.className='rsActRowTag'; t.textContent=d.tag; row.appendChild(t);
  const tx=document.createElement('div'); tx.className='rsActRowText'; tx.textContent=d.text;
  if(d.sub){ const s=document.createElement('div'); s.className='sub'; s.textContent=d.sub; tx.appendChild(s); }
  row.appendChild(tx);
  const tm=document.createElement('span'); tm.className='rsActRowTime'; tm.textContent=d.time; row.appendChild(tm);
  r.elLog.appendChild(row);
}
function rsCreateActiveCard(r){
  const list=document.getElementById('rsActiveList'); if(!list) return;
  list.querySelectorAll('.rsActEmpty').forEach(function(e){ e.remove(); });
  const card=document.createElement('div'); card.className='rsActCard'; card.id='rsCard-'+r.id;
  const head=document.createElement('div'); head.className='rsActHead';
  const title=document.createElement('span'); title.className='rsActTitle'; title.textContent=r.topic; head.appendChild(title);
  const time=document.createElement('span'); time.className='rsActTime'; time.textContent='0:00'; head.appendChild(time);
  r.elTime=time;
  const chev=document.createElement('span'); chev.className='rsActTime'; chev.textContent='▾'; head.appendChild(chev);
  r.elChev=chev;
  const menuBtn=document.createElement('button'); menuBtn.className='rsActMenuBtn'; menuBtn.type='button'; menuBtn.textContent='⋮'; head.appendChild(menuBtn);
  r.elMenuBtn=menuBtn;
  card.appendChild(head);
  const detail=document.createElement('div'); detail.className='rsActDetail';
  const phase=document.createElement('div'); phase.className='rsActPhase'; detail.appendChild(phase);
  r.elPhase=phase;
  const vis=document.createElement('div'); vis.className='rsActVis';
  vis.innerHTML='<div class="rsActVisGrid"></div><div class="rsActScan"></div><div class="rsActVisInner"><div class="rsActRound">round 1</div><div class="rsActBig"></div><div class="rsActMeta"><span class="rsActSrc">0 sources</span><span class="rsActElapsed">00:00</span></div></div>';
  detail.appendChild(vis);
  r.elRound=vis.querySelector('.rsActRound');
  r.elGlyph=vis.querySelector('.rsActBig');
  r.elSrc=vis.querySelector('.rsActSrc');
  r.elVis=vis.querySelector('.rsActElapsed');
  const bar=document.createElement('div'); bar.className='rsActBar';
  const fill=document.createElement('div'); fill.className='rsActFill'; fill.style.width='2%'; bar.appendChild(fill);
  detail.appendChild(bar);
  r.elFill=fill;
  const log=document.createElement('div'); log.className='rsActLog';
  detail.appendChild(log);
  r.elLog=log;
  r.logData.forEach(function(d){ rsAppendLogRow(r, d); });
  const btns=document.createElement('div'); btns.className='rsActBtns';
  const cancel=document.createElement('button'); cancel.className='btn'; cancel.type='button'; cancel.textContent='Cancel'; cancel.style.background='#7a2e2e';
  cancel.addEventListener('click', function(){ rsCancelRun(r.id); });
  btns.appendChild(cancel);
  r.elCancel=cancel;
  const view=document.createElement('button'); view.className='btn'; view.type='button'; view.textContent='View report'; view.style.display='none';
  view.addEventListener('click', function(){ rsToggleReport(r); });
  btns.appendChild(view);
  r.elView=view;
  detail.appendChild(btns);
  const rep=document.createElement('div'); rep.className='rsActReport'; rep.style.display='none';
  detail.appendChild(rep);
  r.elReport=rep;
  card.appendChild(detail);
  r.elDetail=detail;
  head.addEventListener('click', function(ev){
    if(ev.target===menuBtn) return;
    const hidden=detail.style.display==='none';
    detail.style.display=hidden?'':'none';
    chev.textContent=hidden?'▾':'▸';
  });
  menuBtn.addEventListener('click', function(ev){ ev.stopPropagation(); rsToggleMenu(r, menuBtn); });
  list.prepend(card);
  r.elCard=card;
  rsUpdateActiveCard(r);
}
function rsCloseMenus(){ document.querySelectorAll('.rsActMenu').forEach(function(m){ m.remove(); }); }
function rsToggleMenu(r, anchor){
  rsCloseMenus();
  const menu=document.createElement('div'); menu.className='rsActMenu';
  if(r.state==='running'){
    const c=document.createElement('button'); c.className='danger'; c.textContent='Cancel research';
    c.addEventListener('click', function(){ rsCloseMenus(); rsCancelRun(r.id); });
    menu.appendChild(c);
  }
  if(r.state==='done'&&r.report){
    const v=document.createElement('button'); v.textContent='View report';
    v.addEventListener('click', function(){ rsCloseMenus(); rsToggleReport(r); });
    menu.appendChild(v);
    const d=document.createElement('button'); d.textContent='Download .md';
    d.addEventListener('click', function(){ rsCloseMenus(); rsDownloadReport(r.histId||r.id); });
    menu.appendChild(d);
    const p=document.createElement('button'); p.textContent='Download PDF';
    p.addEventListener('click', function(){ rsCloseMenus(); rsDownloadReportPdf(r); });
    menu.appendChild(p);
  }
  const rm=document.createElement('button'); rm.textContent='Remove from Active';
  rm.addEventListener('click', function(){ rsCloseMenus(); delete rsActiveRuns[r.id]; if(r.elCard) r.elCard.remove(); rsUpdateActiveTab(); rsStopTickIfIdle(); });
  menu.appendChild(rm);
  anchor.parentElement.appendChild(menu);
  setTimeout(function(){ document.addEventListener('click', function h(ev){ if(!menu.contains(ev.target)){ rsCloseMenus(); document.removeEventListener('click', h); } }); }, 0);
}
// Download a saved report as a Markdown file (server endpoint, attachment).
function rsDownloadReport(histId){
  if(!histId) return;
  const a=document.createElement('a');
  a.href='/api/research/history/'+encodeURIComponent(histId)+'/download';
  a.style.display='none';
  document.body.appendChild(a);
  a.click();
  a.remove();
}
// Download a report as a PDF that keeps the coloured VISUAL format
// (opens a print view; choose "Save as PDF" in the print dialog).
function rsDownloadReportPdf(r){
  if(!r) return;
  printResearchReport(r.histId||r.id);
}
function rsToggleReport(r){
  if(!r.elReport||!r.report) return;
  if(r.elReport.style.display!=='none'){ r.elReport.style.display='none'; return; }
  let h='<div class="rowflex"><button class="btn rsDlBtn" type="button">Download .md</button><button class="btn rsPdfBtn" type="button">Download PDF</button></div><div class="msg">'+md(r.report)+'</div>';
  if(r.sources&&r.sources.length){
    h+='<div class="msg"><b>Sources</b><ol>';
    r.sources.forEach(function(x){ h+='<li><a href="'+esc(x.url)+'" target="_blank">'+esc(x.title||x.url)+'</a></li>'; });
    h+='</ol></div>';
  }
  r.elReport.innerHTML=h;
  const dl=r.elReport.querySelector('.rsDlBtn');
  if(dl) dl.addEventListener('click', function(){ rsDownloadReport(r.histId||r.id); });
  const pdf=r.elReport.querySelector('.rsPdfBtn');
  if(pdf) pdf.addEventListener('click', function(){ rsDownloadReportPdf(r); });
  r.elReport.style.display='';
}
function renderRsActive(){
  const list=document.getElementById('rsActiveList'); if(!list) return;
  if(!Object.keys(rsActiveRuns).length){ list.innerHTML='<p class="hint rsActEmpty">No research running right now. Start one from the Research tab.</p>'; return; }
  list.querySelectorAll('.rsActEmpty').forEach(function(e){ e.remove(); });
}
async function rsCancelRun(id){
  const r=rsActiveRuns[id];
  if(r&&r.state==='running'){
    try{ await fetch('/api/research/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id:id})}); }catch(e){}
  }
  if(rsReader){ try{ await rsReader.cancel(); }catch(e){} }
  if(r&&r.state==='running'){ r.state='cancelled'; r.phase='cancelled'; rsAddLogRow(r,'Cancelled','Cancelled by user'); rsUpdateActiveCard(r); rsUpdateActiveTab(); rsStopTickIfIdle(); loadResearchHistory(); }
  resetRsButtons();
}
async function runResearch(){
  const topic=document.getElementById('rsTopic').value.trim();
  if(!topic){ showToast('Type a topic first'); return; }
  const rsSel=document.getElementById('rsModel');
  const rsVal=rsSel?rsSel.value:'';
  let rsModel='', rsEndpointId=null;
  if(rsVal.indexOf('endpoint:')===0){
    const parts=rsVal.split(':'); rsEndpointId=parts[1]; rsModel=parts.slice(2).join(':');
  } else {
    rsModel=rsVal||currentModel;
  }
  if(!rsEndpointId){
    showToast('Deep Research needs a capable model. The local model is too small for this — pick a Gemini model from the list above first.');
    return;
  }
  const rsRounds=document.getElementById('rsRounds');
  const rsFormat=document.getElementById('rsFormat');
  const rsEngine=document.getElementById('rsEngine');
  const rsRoundsVal=rsRounds?parseInt(rsRounds.value,10):8;
  const rsFormatVal=rsFormat?rsFormat.value:'full';
  const rsEngineVal=rsEngine?rsEngine.value:'bing';
  rsSaveOpts();
  const run={id:'rs-'+(++rsActiveSeq)+'-'+Date.now(), topic:topic, model:rsModel, endpoint:rsEndpointId, totalRounds:rsRoundsVal, format:rsFormatVal, engine:rsEngineVal, startedAt:Date.now(), state:'running', phase:'starting', round:0, doneRounds:0, queries:0, sources:[], logData:[], report:'', duration:'', error:''};
  rsActiveRuns[run.id]=run;
  rsRunId=run.id;
  document.getElementById('rsGo').style.display='none';
  document.getElementById('rsCancel').style.display='';
  rsCreateActiveCard(run);
  rsUpdateActiveTab();
  rsEnsureTick();
  showRsTab('active');
  rsAddLogRow(run,'Started','Starting research with '+rsModel+' ('+rsRoundsVal+' rounds, '+rsFormatVal+', '+rsEngineVal+')...');
  try{
    const r=await fetch('/api/research',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic:topic, model:rsModel, endpoint_id:rsEndpointId, rounds:rsRoundsVal, format:rsFormatVal, engine:rsEngineVal})});
    if(!r.ok){
      run.state='error'; run.error='Research failed ('+r.status+'): '+await r.text();
      rsAddLogRow(run,'Error',run.error);
      rsUpdateActiveCard(run); rsUpdateActiveTab(); rsStopTickIfIdle(); resetRsButtons(); return;
    }
    const reader=r.body.getReader(); rsReader=reader;
    const dec=new TextDecoder(); let buf=''; let finished=false;
    while(!finished){
      const chunk=await reader.read();
      if(chunk.done) break;
      buf+=dec.decode(chunk.value,{stream:true});
      let idx;
      while((idx=buf.indexOf('\n\n'))>=0){
        const block=buf.slice(0,idx); buf=buf.slice(idx+2);
        const dataLines=block.split('\n').filter(function(l){ return l.indexOf('data:')===0; }).map(function(l){ return l.slice(5).trim(); });
        if(!dataLines.length) continue;
        const data=dataLines.join('\n');
        if(data==='[DONE]'){ finished=true; break; }
        let ev; try{ ev=JSON.parse(data); }catch(e){ continue; }
        handleRsEvent(ev, run);
        if(ev.type==='done'||ev.type==='error'||ev.type==='cancelled'){ finished=true; break; }
      }
    }
    try{ await reader.cancel(); }catch(e){}
  }catch(e){
    if(run.state==='running'){ run.state='error'; run.error='Research failed: '+e.message; rsAddLogRow(run,'Error',run.error); rsUpdateActiveCard(run); rsUpdateActiveTab(); rsStopTickIfIdle(); }
  }
  resetRsButtons();
}


// Cookbook: model manager, mirroring the Odysseus Cookbook (model serving).
// Search HuggingFace, list cached models, download, list running, stop.
// NOT a recipe book. The old recipe data file cookbook.json is left on disk.
function fmtSize(n){
  if(n==null) return '';
  if(n>=1e9) return (n/1e9).toFixed(2)+' GB';
  if(n>=1e6) return (n/1e6).toFixed(1)+' MB';
  if(n>=1e3) return (n/1e3).toFixed(0)+' KB';
  return n+' B';
}
function armStop(btn, run){
  // Two taps to confirm a stop: first tap arms the button, second runs it.
  if(btn.dataset.armed){ delete btn.dataset.armed; btn.textContent='Stop'; run(); return; }
  btn.dataset.armed='1'; btn.textContent='Sure?';
  setTimeout(function(){ if(btn.dataset.armed){ delete btn.dataset.armed; btn.textContent='Stop'; } }, 5000);
}
async function openCookbookView(){
  openFeature('Cookbook');
  fvBody.innerHTML='<p class="hint">Models on this phone. Only small models fit — check free space before downloading a big one.</p>'
    +'<div class="fv-sec">Search HuggingFace</div>'
    +'<input id="cbQ" class="fld" placeholder="Search models, e.g. qwen3">'
    +'<button id="cbSearch" class="btn" type="button">Search</button>'
    +'<div id="cbResults"></div>'
    +'<div class="fv-sec">Download a GGUF file</div>'
    +'<input id="cbGgufUrl" class="fld" placeholder="https://huggingface.co/.../model.gguf">'
    +'<input id="cbGgufName" class="fld" placeholder="File name, e.g. model.gguf">'
    +'<button id="cbGgufDl" class="btn" type="button">Download GGUF</button>'
    +'<div class="fv-sec">Downloads</div><div id="cbTasks"></div>'
    +'<div class="fv-sec">Cached models</div><div id="cbCached"></div>'
    +'<div class="fv-sec">Running models</div><div id="cbRunning"></div>'
    +'<button id="cbRefreshPicker" class="btn" type="button">Refresh model picker</button>';
  document.getElementById('cbSearch').addEventListener('click', searchCbModels);
  document.getElementById('cbGgufDl').addEventListener('click', downloadCbGguf);
  document.getElementById('cbRefreshPicker').addEventListener('click', async function(){
    await refreshModelSelect(); showToast('Model picker refreshed');
  });
  await loadCbAll();
}
async function searchCbModels(){
  const q=document.getElementById('cbQ').value.trim();
  const box=document.getElementById('cbResults'); if(!box) return;
  if(!q){ showToast('Type a search'); return; }
  box.innerHTML='<p class="hint">Searching HuggingFace...</p>';
  try{
    const r=await fetch('/api/cookbook/search?q='+encodeURIComponent(q));
    if(!r.ok){ box.innerHTML='<p class="hint">Search failed ('+r.status+'): '+esc(await r.text())+'</p>'; return; }
    const list=await r.json();
    if(!list.length){ box.innerHTML='<p class="hint">No models found.</p>'; return; }
    box.innerHTML='';
    for(const m of list){
      const row=document.createElement('div'); row.className='listrow';
      const info=document.createElement('div'); info.className='info';
      const nm=document.createElement('div'); nm.className='nm'; nm.textContent=m.id;
      info.appendChild(nm);
      const sub=document.createElement('div'); sub.className='sub';
      sub.textContent=(m.author||'')+(m.size_bytes!=null?' · '+fmtSize(m.size_bytes):' · size unknown')+' · '+(m.downloads||0)+' downloads · '+(m.likes||0)+' likes'+(m.pipeline_tag?' · '+m.pipeline_tag:'');
      info.appendChild(sub);
      row.appendChild(info);
      if(m.gguf_files && m.gguf_files.length){
        const sel=document.createElement('select'); sel.className='mini';
        for(const f of m.gguf_files){
          const o=document.createElement('option'); o.value=f.filename;
          o.textContent=f.filename+(f.size!=null?' ('+fmtSize(f.size)+')':'');
          sel.appendChild(o);
        }
        row.appendChild(sel);
        const dl=document.createElement('button'); dl.className='mini'; dl.textContent='Download GGUF';
        dl.addEventListener('click', async function(){
          const filename=sel.value;
          const url='https://huggingface.co/'+m.id+'/resolve/main/'+filename;
          const rr=await fetch('/api/cookbook/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'gguf', url:url, filename:filename})});
          if(!rr.ok){ showToast('Download failed ('+rr.status+'): '+await rr.text()); return; }
          showToast('GGUF download started: '+filename); await loadCbTasks();
        });
        row.appendChild(dl);
        const imp=document.createElement('button'); imp.className='mini'; imp.textContent='Import';
        imp.addEventListener('click', async function(){
          const rr=await fetch('/api/cookbook/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:sel.value})});
          if(!rr.ok){ showToast('Import failed ('+rr.status+'): '+await rr.text()); return; }
          showToast('Import started: '+sel.value); await loadCbTasks();
        });
        row.appendChild(imp);
      } else {
        const no=document.createElement('span'); no.className='sub';
        no.textContent='No GGUF file — cannot download as a chat model';
        row.appendChild(no);
      }
      box.appendChild(row);
    }
  }catch(e){ box.innerHTML='<p class="hint">Search failed: '+esc(e.message)+'</p>'; }
}
async function downloadCbGguf(){
  const url=document.getElementById('cbGgufUrl').value.trim();
  const filename=document.getElementById('cbGgufName').value.trim();
  if(!url || !filename){ showToast('Paste the GGUF link and a file name'); return; }
  const r=await fetch('/api/cookbook/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'gguf', url:url, filename:filename})});
  if(!r.ok){ showToast('Download failed ('+r.status+'): '+await r.text()); return; }
  showToast('GGUF download started'); await loadCbTasks();
}
async function loadCbTasks(){
  const box=document.getElementById('cbTasks'); if(!box) return;
  try{
    const r=await fetch('/api/cookbook/downloads'); const tasks=await r.json();
    if(!tasks.length){ box.innerHTML='<p class="hint">No downloads yet.</p>'; return; }
    box.innerHTML='';
    for(const t of tasks){
      const row=document.createElement('div'); row.className='listrow';
      const info=document.createElement('div'); info.className='info';
      const nm=document.createElement('div'); nm.className='nm'; nm.textContent=t.name;
      info.appendChild(nm);
      const sub=document.createElement('div'); sub.className='sub'; sub.textContent=t.kind+' · '+t.status+(t.detail?' · '+t.detail:'');
      info.appendChild(sub);
      row.appendChild(info); box.appendChild(row);
    }
  }catch(e){ box.innerHTML='<p class="hint">Could not load downloads.</p>'; }
}
async function loadCbCached(){
  const box=document.getElementById('cbCached'); if(!box) return;
  try{
    const r=await fetch('/api/cookbook/cached'); const j=await r.json();
    box.innerHTML='';
    if(j.disk_free_bytes!=null){
      const p=document.createElement('p'); p.className='hint';
      p.textContent='Free space: '+fmtSize(j.disk_free_bytes)+' of '+fmtSize(j.disk_total_bytes)+'.';
      box.appendChild(p);
    }
    if(!j.ollama.length && !j.gguf.length){ box.innerHTML+='<p class="hint">No models cached yet.</p>'; return; }
    for(const m of j.ollama){
      const row=document.createElement('div'); row.className='listrow';
      const info=document.createElement('div'); info.className='info';
      const nm=document.createElement('div'); nm.className='nm'; nm.textContent=m.name;
      info.appendChild(nm);
      const sub=document.createElement('div'); sub.className='sub';
      sub.textContent='Ollama · '+fmtSize(m.size)+(m.parameter_size?' · '+m.parameter_size:'')+(m.quantization?' · '+m.quantization:'')+(m.modified_at?' · '+m.modified_at.slice(0,10):'');
      info.appendChild(sub);
      row.appendChild(info);
      const use=document.createElement('button'); use.className='mini'; use.textContent='Use';
      use.addEventListener('click', async function(){
        await refreshModelSelect();
        modelSel.value=m.name; currentModel=m.name;
        if(current){current.model=currentModel; saveChat();}
        showToast('Using '+m.name);
      });
      row.appendChild(use);
      box.appendChild(row);
    }
    for(const g of j.gguf){
      const row=document.createElement('div'); row.className='listrow';
      const info=document.createElement('div'); info.className='info';
      const nm=document.createElement('div'); nm.className='nm'; nm.textContent=g.name;
      info.appendChild(nm);
      const sub=document.createElement('div'); sub.className='sub';
      sub.textContent='GGUF file · '+fmtSize(g.size)+(g.modified?' · '+new Date(g.modified*1000).toISOString().slice(0,10):'');
      info.appendChild(sub);
      row.appendChild(info);
      const imp=document.createElement('button'); imp.className='mini'; imp.textContent='Import';
      imp.addEventListener('click', async function(){
        const rr=await fetch('/api/cookbook/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:g.name})});
        if(!rr.ok){ showToast('Import failed ('+rr.status+'): '+await rr.text()); return; }
        showToast('Import started: '+g.name); await loadCbTasks();
      });
      row.appendChild(imp);
      box.appendChild(row);
    }
  }catch(e){ box.innerHTML='<p class="hint">Could not load cached models.</p>'; }
}
async function loadCbRunning(){
  const box=document.getElementById('cbRunning'); if(!box) return;
  try{
    const r=await fetch('/api/cookbook/running'); const j=await r.json();
    box.innerHTML='';
    if(!j.ollama.length && !j.llama_server){ box.innerHTML='<p class="hint">Nothing running.</p>'; return; }
    for(const m of j.ollama){
      const row=document.createElement('div'); row.className='listrow';
      const info=document.createElement('div'); info.className='info';
      const nm=document.createElement('div'); nm.className='nm'; nm.textContent=m.name||'';
      info.appendChild(nm);
      const sub=document.createElement('div'); sub.className='sub'; sub.textContent='Ollama · running · '+fmtSize(m.size||0);
      info.appendChild(sub);
      row.appendChild(info);
      const stop=document.createElement('button'); stop.className='mini del'; stop.textContent='Stop';
      stop.addEventListener('click', function(){
        armStop(stop, async function(){
          const rr=await fetch('/api/cookbook/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'ollama', name:m.name})});
          if(!rr.ok){ showToast('Stop failed ('+rr.status+')'); return; }
          showToast('Stopped '+m.name); await loadCbRunning();
        });
      });
      row.appendChild(stop); box.appendChild(row);
    }
    if(j.llama_server){
      const row=document.createElement('div'); row.className='listrow';
      const info=document.createElement('div'); info.className='info';
      const nm=document.createElement('div'); nm.className='nm'; nm.textContent='llama-server (pid '+j.llama_server.pid+')';
      info.appendChild(nm);
      const sub=document.createElement('div'); sub.className='sub'; sub.textContent=j.llama_server.cmd||'';
      info.appendChild(sub);
      row.appendChild(info);
      const stop=document.createElement('button'); stop.className='mini del'; stop.textContent='Stop';
      stop.addEventListener('click', function(){
        armStop(stop, async function(){
          const rr=await fetch('/api/cookbook/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'llama', pid:j.llama_server.pid})});
          if(!rr.ok){ showToast('Stop failed ('+rr.status+')'); return; }
          showToast('llama-server stopped'); await loadCbRunning();
        });
      });
      row.appendChild(stop); box.appendChild(row);
    }
  }catch(e){ box.innerHTML='<p class="hint">Could not load running models.</p>'; }
}
async function loadCbAll(){
  await loadCbTasks();
  await loadCbCached();
  await loadCbRunning();
}

// Library: read-only browser of workspace files, saved chats, generated images.
async function openLibraryView(){
  openFeature('Library');
  fvBody.innerHTML='<p class="hint">Everything saved on the phone.</p>'
    +'<div class="fv-sec">Workspace files</div><div id="libWs"></div>'
    +'<div class="fv-sec">Saved chats</div><div id="libChats"></div>'
    +'<div class="fv-sec">Generated images</div><div id="libImgs"></div>';
  try{
    const r=await fetch('/api/workspace'); const items=await r.json();
    const box=document.getElementById('libWs');
    if(!items.length){ box.innerHTML='<p class="hint">Workspace is empty.</p>'; }
    else{
      for(const it of items){
        const row=document.createElement('div'); row.className='listrow';
        const info=document.createElement('div'); info.className='info';
        const nm=document.createElement('div'); nm.className='nm'; nm.textContent=it.path+(it.is_dir?' /':'');
        info.appendChild(nm);
        row.appendChild(info);
        if(!it.is_dir){
          const open=document.createElement('button'); open.className='mini'; open.textContent='Open';
          open.addEventListener('click', function(){ openWsFile(it.path); });
          row.appendChild(open);
        }
        box.appendChild(row);
      }
    }
  }catch(e){ document.getElementById('libWs').innerHTML='<p class="hint">Could not load workspace.</p>'; }
  try{
    const r=await fetch('/api/chats'); const list=await r.json();
    const box=document.getElementById('libChats');
    if(!list.length){ box.innerHTML='<p class="hint">No saved chats yet.</p>'; }
    else{
      for(const c of list){
        const row=document.createElement('div'); row.className='listrow';
        const info=document.createElement('div'); info.className='info';
        const nm=document.createElement('div'); nm.className='nm'; nm.textContent=c.title||'Untitled';
        const sub=document.createElement('div'); sub.className='sub'; sub.textContent=fmtChatDate(c.updated_at);
        info.appendChild(nm); info.appendChild(sub);
        info.addEventListener('click', async function(){ closeFeature(); await openChat(c.id); });
        row.appendChild(info); box.appendChild(row);
      }
    }
  }catch(e){ document.getElementById('libChats').innerHTML='<p class="hint">Could not load chats.</p>'; }
  const imgBox=document.getElementById('libImgs');
  const seen={}; let found=0;
  for(const c of chats.slice(0,20)){
    try{
      const r=await fetch('/api/chats/'+encodeURIComponent(c.id));
      if(!r.ok) continue;
      const full=await r.json();
      for(const m of (full.messages||[])){
        if(m.image && !seen[m.image]){
          seen[m.image]=true; found++;
          const row=document.createElement('div'); row.className='listrow';
          const thumb=document.createElement('img'); thumb.alt=''; thumb.style.cssText='width:56px;height:56px;object-fit:cover;border-radius:8px;flex-shrink:0';
          if(dataSaver){
            // Data Saver: thumbnails do not download until tapped.
            thumb.src='';
            thumb.style.cssText='width:56px;height:56px;object-fit:cover;border-radius:8px;flex-shrink:0;background:var(--panel2);border:1px dashed var(--border);cursor:pointer;';
            thumb.alt='Tap to load';
            thumb.addEventListener('click', function(){ thumb.src=m.image; thumb.style.cursor='default'; });
          }else{
            thumb.src=m.image;
          }
          const info=document.createElement('div'); info.className='info';
          const nm=document.createElement('div'); nm.className='sub2'; nm.textContent=(m.content||'Image').slice(0,100);
          info.appendChild(nm);
          info.addEventListener('click', function(){ window.open(m.image, '_blank'); });
          row.appendChild(thumb); row.appendChild(info); imgBox.appendChild(row);
        }
      }
    }catch(e){}
  }
  if(!found){ imgBox.innerHTML='<p class="hint">No generated images yet.</p>'; }
}

// ---------- In-app browser ----------
// A simple browser inside the app: an address bar, an iframe for pages
// that allow framing, and a server-side readable text view for pages that
// block framing (X-Frame-Options / CSP). A bare word searches with Bing.
// Links tapped in chat and research sources open here.
let browserLast=null;
async function openBrowserView(url){
  openFeature('Browser');
  fvBody.innerHTML='<div id="browserBar"><input id="browserUrl" class="fld" placeholder="Type a web address or search..." autocomplete="off"><button id="browserGo" class="btn" type="button">Go</button></div>'
    +'<div class="browserlinks"><span id="browserStatus" class="bstatus"></span><a id="browserReaderLink">Reader view</a><a id="browserPageLink">Page view</a><a id="browserExtLink">Open externally</a></div>'
    +'<div id="browserView"><iframe id="browserFrame" title="Browser"></iframe><div id="browserReader" style="display:none"></div></div>';
  const input=document.getElementById('browserUrl');
  const go=function(){ browseTo(input.value); };
  document.getElementById('browserGo').addEventListener('click', go);
  input.addEventListener('keydown', function(e){ if(e.key==='Enter'){ e.preventDefault(); go(); } });
  document.getElementById('browserReaderLink').addEventListener('click', function(){ showBrowserMode('reader'); });
  document.getElementById('browserPageLink').addEventListener('click', function(){ showBrowserMode('page'); });
  document.getElementById('browserExtLink').addEventListener('click', function(){
    if(browserLast && browserLast.final_url) window.open(browserLast.final_url, '_blank');
  });
  if(url){ input.value=url; await browseTo(url); }
  else{ input.focus(); }
}
function showBrowserMode(mode){
  const fr=document.getElementById('browserFrame'), rd=document.getElementById('browserReader');
  if(!fr || !rd) return;
  if(mode==='reader'){ fr.style.display='none'; rd.style.display='block'; }
  else{ rd.style.display='none'; fr.style.display='block'; }
}
async function browseTo(raw){
  const input=document.getElementById('browserUrl');
  const status=document.getElementById('browserStatus');
  const fr=document.getElementById('browserFrame');
  const rd=document.getElementById('browserReader');
  if(!input || !fr || !rd) return;
  const q=(raw!==undefined && raw!==null ? String(raw) : input.value).trim();
  if(!q){ if(status) status.textContent='Type a web address or a search.'; return; }
  if(status) status.textContent='Loading…';
  try{
    const r=await fetch('/api/browser/fetch?url='+encodeURIComponent(q));
    if(!r.ok){ const t=await r.text(); throw new Error(t||('server error '+r.status)); }
    const j=await r.json();
    browserLast=j;
    input.value=j.final_url||j.url;
    const isMedia=/^(image|video|application\/pdf)/i.test(j.content_type||'');
    if(j.frameable || isMedia){
      rd.style.display='none'; fr.style.display='block';
      fr.src=j.final_url||j.url;
      if(status) status.textContent=(j.title?j.title+' — ':'')+(j.final_url||j.url);
    }else{
      fr.style.display='none'; rd.style.display='block';
      rd.innerHTML='';
      const h=document.createElement('h3'); h.textContent=j.title||j.final_url||j.url;
      const note=document.createElement('p'); note.className='hint'; note.textContent='This site blocks embedding, so here is a readable text view.';
      const body=document.createElement('div'); body.textContent=j.text||'No readable text found on this page.';
      rd.appendChild(h); rd.appendChild(note); rd.appendChild(body);
      if(status) status.textContent=(j.title?j.title+' — ':'')+'Reader view (framing blocked)';
    }
  }catch(err){
    if(status) status.textContent='Could not load page: '+err.message;
  }
}
function openLinkInBrowser(url){
  if(!url || !/^https?:\/\//i.test(url)) return false;
  openBrowserView(url);
  return true;
}

// ---------- Calendar ----------
// Local-only month view. Events live in calendar.json on the phone.
let calYear=0, calMonth=0, calSel=null, calEvents=[];
const CAL_MONTHS=['January','February','March','April','May','June','July','August','September','October','November','December'];
function fmtCalDate(d){ return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0'); }
async function loadCalendarEvents(){
  try{ const r=await fetch('/api/calendar'); calEvents=await r.json(); }catch(e){ calEvents=[]; }
}
async function openCalendarView(){
  openFeature('Calendar');
  const now=new Date();
  calYear=now.getFullYear(); calMonth=now.getMonth(); calSel=fmtCalDate(now);
  await loadCalendarEvents();
  renderCalendar();
}
function renderCalendar(){
  const first=new Date(calYear, calMonth, 1);
  const daysInMonth=new Date(calYear, calMonth+1, 0).getDate();
  const startOff=(first.getDay()+6)%7; // Monday-first
  const todayStr=fmtCalDate(new Date());
  fvBody.innerHTML='<div class="calhead"><button id="calPrev" class="mini" type="button">‹</button>'
    +'<div class="caltitle">'+CAL_MONTHS[calMonth]+' '+calYear+'</div>'
    +'<button id="calNext" class="mini" type="button">›</button></div>'
    +'<div class="calgrid">'
    +['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].map(function(d){ return '<div class="calweek">'+d+'</div>'; }).join('')
    +'</div><div class="calgrid" id="calDays"></div>'
    +'<div id="calDayBox"></div>';
  const daysBox=document.getElementById('calDays');
  for(let i=0;i<startOff;i++){
    const blank=document.createElement('div'); blank.className='calday other'; daysBox.appendChild(blank);
  }
  for(let d=1; d<=daysInMonth; d++){
    const dateStr=calYear+'-'+String(calMonth+1).padStart(2,'0')+'-'+String(d).padStart(2,'0');
    const cell=document.createElement('div');
    cell.className='calday'+(dateStr===todayStr?' today':'')+(dateStr===calSel?' sel':'');
    const dn=document.createElement('div'); dn.className='dn'; dn.textContent=d;
    cell.appendChild(dn);
    const evs=calEvents.filter(function(e){ return e.date===dateStr; });
    evs.slice(0,2).forEach(function(e){
      const ev=document.createElement('div'); ev.className='ev'; ev.textContent=(e.time?e.time+' ':'')+e.title;
      cell.appendChild(ev);
    });
    if(evs.length>2){
      const more=document.createElement('div'); more.className='ev'; more.textContent='+'+(evs.length-2)+' more';
      cell.appendChild(more);
    }
    cell.addEventListener('click', function(){ calSel=dateStr; renderCalendar(); });
    daysBox.appendChild(cell);
  }
  document.getElementById('calPrev').addEventListener('click', function(){
    calMonth--; if(calMonth<0){ calMonth=11; calYear--; } renderCalendar();
  });
  document.getElementById('calNext').addEventListener('click', function(){
    calMonth++; if(calMonth>11){ calMonth=0; calYear++; } renderCalendar();
  });
  renderCalDay();
}
function renderCalDay(){
  const box=document.getElementById('calDayBox'); if(!box) return;
  box.innerHTML='';
  const head=document.createElement('div'); head.className='fv-sec'; head.textContent=calSel;
  box.appendChild(head);
  const evs=calEvents.filter(function(e){ return e.date===calSel; });
  if(!evs.length){
    const p=document.createElement('p'); p.className='hint'; p.textContent='No events on this day.';
    box.appendChild(p);
  }
  for(const e of evs){
    const row=document.createElement('div'); row.className='listrow';
    const info=document.createElement('div'); info.className='info';
    const nm=document.createElement('div'); nm.className='nm'; nm.textContent=(e.time?e.time+' — ':'')+e.title;
    info.appendChild(nm);
    if(e.notes){
      const sub=document.createElement('div'); sub.className='sub2'; sub.textContent=e.notes;
      info.appendChild(sub);
    }
    const del=document.createElement('button'); del.className='mini del'; del.textContent='×'; del.title='Delete event';
    del.addEventListener('click', async function(){
      await fetch('/api/calendar/'+encodeURIComponent(e.id),{method:'DELETE'});
      await loadCalendarEvents(); renderCalendar();
    });
    row.appendChild(info); row.appendChild(del); box.appendChild(row);
  }
  const form=document.createElement('div');
  form.innerHTML='<p class="hint">Add an event on '+calSel+'</p>'
    +'<input id="calTitle" class="fld" placeholder="Event title">'
    +'<div class="rowflex"><input id="calTime" class="fld" type="time"><input id="calNotes" class="fld" placeholder="Notes (optional)"></div>'
    +'<button id="calAdd" class="btn" type="button">Add event</button>';
  box.appendChild(form);
  form.querySelector('#calAdd').addEventListener('click', async function(){
    const title=form.querySelector('#calTitle').value.trim();
    if(!title){ showToast('Type an event title first'); return; }
    const res=await fetch('/api/calendar',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:title, date:calSel, time:form.querySelector('#calTime').value, notes:form.querySelector('#calNotes').value.trim()})});
    if(!res.ok){ showToast('Could not save event: '+await res.text()); return; }
    await loadCalendarEvents(); renderCalendar();
    showToast('Event saved');
  });
}

// ---------- PDF ----------
// Create: a chat or research report becomes a real PDF (made by the
// server, no library). Edit: upload a PDF, its text is extracted with
// the phone's PyMuPDF helper, edit the text, and save it as a NEW PDF
// that keeps the original format — only the words you change are
// replaced in the original PDF. Originals are kept.
let pdfList=[], pdfTexts={};
async function openPdfView(){
  openFeature('PDF');
  fvBody.innerHTML='<p class="hint">Create a PDF from a chat or a research report, or upload a PDF to view it and edit its text. Edited PDFs keep the original format — only the words you change are replaced in the original PDF.</p>'
    +'<div class="rowflex"><button id="pdfChatBtn" class="btn" type="button">This chat as PDF</button>'
    +'<button id="pdfUploadBtn" class="btn" type="button">Upload PDF</button></div>'
    +'<input type="file" id="pdfFile" accept="application/pdf,.pdf" style="display:none">'
    +'<div id="pdfMsg" class="hint"></div>'
    +'<div class="fv-sec">Your PDFs</div><div id="pdfList"></div>';
  document.getElementById('pdfChatBtn').addEventListener('click', downloadChatPdf);
  document.getElementById('pdfUploadBtn').addEventListener('click', function(){ document.getElementById('pdfFile').click(); });
  document.getElementById('pdfFile').addEventListener('change', uploadPdfFile);
  await loadPdfList();
}
async function loadPdfList(){
  try{ const r=await fetch('/api/pdf/list'); pdfList=await r.json(); }catch(e){ pdfList=[]; }
  renderPdfList();
}
function renderPdfList(){
  const box=document.getElementById('pdfList'); if(!box) return;
  box.innerHTML='';
  if(!pdfList.length){ box.innerHTML='<p class="hint">No PDFs yet.</p>'; return; }
  for(const p of pdfList){
    const row=document.createElement('div'); row.className='listrow';
    const info=document.createElement('div'); info.className='info';
    const nm=document.createElement('div'); nm.className='nm'; nm.textContent=p.name;
    const sub=document.createElement('div'); sub.className='sub'; sub.textContent=fmtSize(p.size);
    info.appendChild(nm); info.appendChild(sub);
    const view=document.createElement('button'); view.className='mini'; view.textContent='View';
    view.addEventListener('click', function(){ viewPdf(p.id, p.name); });
    const edit=document.createElement('button'); edit.className='mini'; edit.textContent='Edit';
    edit.addEventListener('click', function(){ editPdf(p.id, p.name); });
    const dl=document.createElement('a'); dl.className='mini'; dl.textContent='Download'; dl.href='/api/pdf/file/'+encodeURIComponent(p.id); dl.download=p.name; dl.style.cssText='text-decoration:none;display:inline-flex;align-items:center;';
    const del=document.createElement('button'); del.className='mini del'; del.textContent='×'; del.title='Delete PDF';
    del.addEventListener('click', async function(){
      await fetch('/api/pdf/'+encodeURIComponent(p.id),{method:'DELETE'});
      await loadPdfList();
    });
    row.appendChild(info); row.appendChild(view); row.appendChild(edit); row.appendChild(dl); row.appendChild(del);
    box.appendChild(row);
  }
}
async function uploadPdfFile(){
  const input=document.getElementById('pdfFile');
  const file=input.files && input.files[0];
  input.value='';
  if(!file) return;
  const msg=document.getElementById('pdfMsg');
  if(msg) msg.textContent='Uploading '+file.name+'…';
  try{
    const data=await new Promise(function(resolve, reject){
      const reader=new FileReader();
      reader.onload=function(){ resolve(reader.result); };
      reader.onerror=function(){ reject(new Error('could not read file')); };
      reader.readAsDataURL(file);
    });
    const res=await fetch('/api/pdf/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:file.name, data:data})});
    const t=await res.text();
    let j=null; try{ j=JSON.parse(t); }catch(e){}
    if(!res.ok){ throw new Error(t||('server error '+res.status)); }
    if(j.text) pdfTexts[j.id]=j.text;
    if(msg) msg.textContent=j.extract_error ? ('Uploaded. '+j.extract_error+'. You can still view it.') : ('Uploaded '+j.name+' — text extracted, ready to edit.');
    await loadPdfList();
  }catch(err){
    if(msg) msg.textContent='Upload failed: '+err.message;
  }
}
function viewPdf(id, name){
  openFeature('PDF — '+name);
  fvBody.innerHTML='<iframe class="pdfview" src="/api/pdf/file/'+encodeURIComponent(id)+'" title="PDF"></iframe>'
    +'<div class="rowflex"><button id="pdfBack" class="btn" type="button">Back to PDFs</button>'
    +'<button id="pdfEdit2" class="btn" type="button">Edit text</button></div>';
  document.getElementById('pdfBack').addEventListener('click', openPdfView);
  document.getElementById('pdfEdit2').addEventListener('click', function(){ editPdf(id, name); });
}
async function editPdf(id, name){
  openFeature('Edit PDF — '+name);
  fvBody.innerHTML='<p class="hint">Edit the text below, then save. The original format is kept — only the words you change are replaced. It is saved as a NEW PDF; the original file is not changed.</p>'
    +'<textarea id="pdfText" class="fld" rows="14" placeholder="Loading text…"></textarea>'
    +'<div class="rowflex"><button id="pdfSaveEdit" class="btn" type="button">Save as new PDF</button>'
    +'<button id="pdfCancelEdit" class="btn" type="button" style="background:var(--hover)">Cancel</button></div>'
    +'<div id="pdfEditMsg" class="hint"></div>';
  const ta=document.getElementById('pdfText');
  const msg=document.getElementById('pdfEditMsg');
  let text=pdfTexts[id];
  if(text===undefined){
    try{
      const r=await fetch('/api/pdf/text/'+encodeURIComponent(id));
      const t=await r.text();
      let j=null; try{ j=JSON.parse(t); }catch(e){}
      if(!r.ok) throw new Error(t||('server error '+r.status));
      text=(j&&j.text)||'';
      pdfTexts[id]=text;
    }catch(err){
      msg.textContent='Could not get the text: '+err.message;
      ta.value='';
      ta.disabled=true;
      document.getElementById('pdfSaveEdit').disabled=true;
    }
  }
  if(text!==undefined) ta.value=text;
  document.getElementById('pdfCancelEdit').addEventListener('click', openPdfView);
  document.getElementById('pdfSaveEdit').addEventListener('click', async function(){
    msg.textContent='Saving…';
    try{
      const res=await fetch('/api/pdf/edit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id, text:ta.value})});
      const t=await res.text();
      let j=null; try{ j=JSON.parse(t); }catch(e){}
      if(!res.ok) throw new Error(t||('server error '+res.status));
      msg.textContent='Saved as '+j.name+'. ';
      const dl=document.createElement('a');
      dl.href='/api/pdf/file/'+encodeURIComponent(j.id);
      dl.download=j.name;
      dl.textContent='Download it here';
      dl.style.cssText='color:var(--green);font-weight:600;';
      msg.appendChild(dl);
      pdfTexts[j.id]=ta.value;
      await loadPdfList();
      showToast('Saved as '+j.name);
    }catch(err){
      msg.textContent='Save failed: '+err.message;
    }
  });
}
// Split chat messages into ordered PDF segments: text chunks and images,
// in reading order, so a photo lands in the PDF under the text it
// illustrates. Image messages carry the photo in m.image (title in
// m.content, source in m.source); agent replies carry photos as Markdown
// images in the text. Videos cannot go in a PDF — their text line is kept.
function segmentsForMessages(msgs){
  const segs=[];
  function pushText(t){ if(t && t.trim()) segs.push({kind:'text', text:t}); }
  for(const m of (msgs||[])){
    if(!m) continue;
    if(m.role==='user'){ pushText('You: '+(m.content||'')); continue; }
    if(m.image){
      segs.push({kind:'image', url:m.image, title:m.content||'Image', source:m.source||''});
      continue;
    }
    if(m.images && m.images.length){
      for(const u of m.images) segs.push({kind:'image', url:u, title:m.content||'Image', source:m.source||''});
      continue;
    }
    if(!m.content) continue;
    // Split Markdown images out of the text, in order. A "[Source: ...](url)"
    // link right after an image becomes that image's source.
    let rest=m.content;
    let prefix='Mawaxi AI: ';
    const re=/!\[([^\]]*)\]\(([^)\s]+)\)/g;
    let last=0, mm;
    while((mm=re.exec(rest))!==null){
      const before=rest.slice(last, mm.index);
      if(before.trim()){ pushText(prefix+before); prefix=''; }
      let src='';
      const after=rest.slice(re.lastIndex);
      const sm=after.match(/^\s*(?:\n\s*)?\[Source:[^\]]*\]\(([^)\s]+)\)/);
      if(sm){ src=sm[1]; re.lastIndex+=sm[0].length; }
      segs.push({kind:'image', url:mm[2], title:mm[1]||'Image', source:src});
      last=re.lastIndex;
    }
    const tail=rest.slice(last);
    if(tail.trim()) pushText(prefix+tail);
  }
  return segs;
}
async function downloadPdf(title, text, segments){
  try{
    const body={title:title, text:text};
    if(segments && segments.length) body.segments=segments;
    const res=await fetch('/api/pdf/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!res.ok){ showToast('PDF failed: '+await res.text()); return; }
    const blob=await res.blob();
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    const slug=(title||'document').replace(/[^\w\- ]+/g,'').trim().replace(/\s+/g,'-').toLowerCase()||'document';
    a.download=slug+'.pdf';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function(){ URL.revokeObjectURL(a.href); }, 5000);
    const nImg=parseInt(res.headers.get('X-Images-Embedded')||'0',10)||0;
    const nFail=parseInt(res.headers.get('X-Images-Failed')||'0',10)||0;
    showToast(nImg>0 ? ('PDF downloaded ('+nImg+' image'+(nImg===1?'':'s')+(nFail>0 ? ', '+nFail+' could not be loaded' : '')+')') : 'PDF downloaded');
  }catch(e){ showToast('PDF failed: '+e.message); }
}
async function downloadChatPdf(){
  if(!current || !current.messages.length){ showToast('No chat to save as PDF'); return; }
  const lines=[];
  for(const m of current.messages){
    lines.push((m.role==='user'?'You: ':'Mawaxi AI: ')+m.content);
    lines.push('');
  }
  const title=(current.title||'Mawaxi AI chat').slice(0,60);
  await downloadPdf(title, lines.join('\n'), segmentsForMessages(current.messages));
}
async function openWsFile(path){
  let j;
  try{
    const r=await fetch('/api/workspace/file?path='+encodeURIComponent(path));
    if(!r.ok){ showToast('Cannot open this file'); return; }
    j=await r.json();
  }catch(e){ showToast('Cannot open this file'); return; }
  openFeature(j.path);
  const back=document.createElement('button'); back.className='mini'; back.type='button'; back.textContent='Back to Library';
  back.addEventListener('click', openLibraryView);
  const pre=document.createElement('pre');
  pre.style.cssText='white-space:pre-wrap;word-break:break-word;font-size:13px;background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin:0';
  pre.textContent=j.content;
  fvBody.appendChild(back); fvBody.appendChild(pre);
}

function stripMd(s){
  return s.replace(/```[\s\S]*?```/g,' ').replace(/`([^`]*)`/g,'$1').replace(/\*\*([^*]*)\*\*/g,'$1').replace(/(^|[^\w*])\*([^\n*]+)\*/g,'$1$2').replace(/^#{1,6}\s+/gm,'').replace(/\[([^\]]*)\]\([^)]*\)/g,'$1').replace(/\n+/g,' ').trim();
}
function speakLatest(){
  if(!('speechSynthesis' in window)){ showToast('Read aloud is not supported in this browser'); return; }
  if(window.speechSynthesis.speaking){ window.speechSynthesis.cancel(); return; }
  if(!current || !current.messages || !current.messages.length){ showToast('Nothing to read yet'); return; }
  let txt='';
  for(let i=current.messages.length-1;i>=0;i--){ const m=current.messages[i]; if(m.role==='assistant' && m.content){ txt=m.content; break; } }
  txt=stripMd(txt||'');
  if(!txt){ showToast('Nothing to read yet'); return; }
  const u=new SpeechSynthesisUtterance(txt); u.lang='en-US'; window.speechSynthesis.speak(u);
}

function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
// A2UI (Google, v1.0): the agent answers with a real interface instead of a
// wall of text. It sends a fenced ```a2ui JSON surface; the app draws it with
// its own components. The agent never sends code — only components from the
// catalog below (Text, Card, Button, ChoiceChips, Divider). A tap sends the
// choice back as a normal message so the agent can continue.
function renderA2UI(jsonText){
  let surf;
  try{ surf=JSON.parse(jsonText); }catch(e){
    return '<pre><code>'+esc(jsonText)+'</code></pre>';
  }
  if(!surf || !Array.isArray(surf.components)) return '<pre><code>'+esc(jsonText)+'</code></pre>';
  let out='<div class="a2ui-surface">';
  if(surf.title) out+='<div class="a2ui-title">'+esc(String(surf.title))+'</div>';
  for(const c of surf.components){
    if(!c || typeof c!=='object') continue;
    const p=(c.props && typeof c.props==='object')? c.props : {};
    const kind=String(c.component||'').toLowerCase();
    if(kind==='text'){
      const v=String(p.variant||'body').toLowerCase();
      const cls=v==='caption'?'a2ui-text cap':(v==='heading'?'a2ui-text head':'a2ui-text');
      out+='<div class="'+cls+'">'+esc(String(p.text||''))+'</div>';
    } else if(kind==='card'){
      out+='<div class="a2ui-card"><b>'+esc(String(p.title||''))+'</b>';
      if(p.subtitle) out+='<div class="sub">'+esc(String(p.subtitle))+'</div>';
      if(p.badge) out+='<span class="badge">'+esc(String(p.badge))+'</span>';
      out+='</div>';
    } else if(kind==='button'){
      const label=String(p.label||'Continue');
      const act=String(p.action||label);
      out+='<button type="button" class="a2ui-btn" data-a2ui-action="'+esc(act)+'" data-a2ui-label="'+esc(label)+'">'+esc(label)+'</button>';
    } else if(kind==='choicechips'){
      const opts=Array.isArray(p.options)? p.options : [];
      out+='<div class="a2ui-chips">';
      for(const o of opts){
        const s=String(o);
        out+='<button type="button" class="a2ui-chip" data-a2ui-action="'+esc(s)+'" data-a2ui-label="'+esc(s)+'">'+esc(s)+'</button>';
      }
      out+='</div>';
    } else if(kind==='divider'){
      out+='<hr class="a2ui-divider">';
    }
    // Unknown components are skipped — the agent can only use the catalog.
  }
  out+='</div>';
  return out;
}
function a2uiAction(action, label){
  q.value=label||action||'';
  if(f && f.requestSubmit){ f.requestSubmit(); }
}
document.addEventListener('click',function(e){
  const b=e.target && e.target.closest ? e.target.closest('.a2ui-btn, .a2ui-chip') : null;
  if(!b) return;
  a2uiAction(b.getAttribute('data-a2ui-action'), b.getAttribute('data-a2ui-label'));
});
function md(src){
  const blocks=[];
  src=src.replace(/```([a-zA-Z0-9_-]*)\n?([\s\S]*?)(?:```|$)/g,function(m,lang,code){
    if(lang && lang.toLowerCase()==='a2ui'){ blocks.push(renderA2UI(code.replace(/\n+$/,''))); return '\u0000'+(blocks.length-1)+'\u0000'; }
    blocks.push('<pre><button type="button" class="copybtn">Copy</button><code>'+esc(code.replace(/\n+$/,''))+'</code></pre>');return '\u0000'+(blocks.length-1)+'\u0000';});
  const codes=[];
  src=src.replace(/`([^`\n]+)`/g,function(m,code){codes.push('<code>'+esc(code)+'</code>');return '\u0001'+(codes.length-1)+'\u0001';});
  src=src.replace(/([^\n#])(#{2,6}\s)/g,'$1\n$2');
  src=esc(src);
  src=src.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  src=src.replace(/\*\*([^\n*]+)\*\*/g,'<strong>$1</strong>');
  src=src.replace(/(^|[^\w*])\*([^\n*]+)\*/g,'$1<em>$2</em>');
  src=src.replace(/!\[([^\]\n]*)\]\((https?:[^)\s]+)\)/g,'<a href="$2" target="_blank" rel="noopener"><img src="$2" alt="$1" loading="lazy" onerror="var p=this.parentNode;var a=document.createElement(\'a\');a.href=this.src;a.target=\'_blank\';a.rel=\'noopener\';a.textContent=(this.alt||\'Image\')+\' (image link)\';p.replaceWith(a);"></a>');
  src=src.replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  const lines=src.split('\n'); let out='', inList=false;
  const flush=function(){if(inList){out+='</ul>';inList=false;}};
  for(const line of lines){
    if(/^\u0000\d+\u0000$/.test(line.trim())){flush();out+=line.trim().replace(/\u0000(\d+)\u0000/g,function(m,i){return blocks[+i];});continue;}
    const m=line.match(/^\s*[-*]\s+(.*)$/);
    if(m){if(!inList){out+='<ul>';inList=true;}out+='<li>'+m[1]+'</li>';continue;}
    flush();
    const h=line.match(/^(#{1,3})\s+(.*)$/);
    if(h){out+='<h'+h[1].length+'>'+h[2]+'</h'+h[1].length+'>';}
    else if(line.trim()===''){}
    else{out+='<p>'+line+'</p>';}
  }
  flush();
  out=out.replace(/\u0000(\d+)\u0000/g,function(m,i){return blocks[+i];});
  out=out.replace(/\u0001(\d+)\u0001/g,function(m,i){return codes[+i];});
  return out;
}

function fallbackCopy(txt){
  const ta=document.createElement('textarea'); ta.value=txt; ta.style.position='fixed'; ta.style.opacity='0';
  document.body.appendChild(ta); ta.select();
  try{ document.execCommand('copy'); }catch(_){}
  document.body.removeChild(ta);
}
document.addEventListener('click',function(e){
  const b=e.target && e.target.closest ? e.target.closest('.copybtn') : null;
  if(!b) return;
  const pre=b.closest('pre'); const code=pre && pre.querySelector('code');
  if(!code) return;
  const txt=code.textContent;
  const done=function(){ b.textContent='Copied'; setTimeout(function(){ b.textContent='Copy'; },2000); };
  if(navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(txt).then(done).catch(function(){ fallbackCopy(txt); done(); });
  } else { fallbackCopy(txt); done(); }
});
function newId(){return Date.now().toString(36)+Math.random().toString(36).slice(2,8);}
function newChatObj(){return {id:newId(), title:'New chat', model:currentModel, messages:[], updated_at:Math.floor(Date.now()/1000)};}

async function loadModels(){
  try{
    const r=await fetch('/api/models'); const j=await r.json();
    const models=(j.models && j.models.length)? j.models : [DEFAULT_MODEL];
    modelSel.innerHTML='';
    for(const m of models){const o=document.createElement('option'); o.value=m; o.textContent=m; modelSel.appendChild(o);}
    if(models.indexOf(currentModel)>=0){modelSel.value=currentModel;}
    else{currentModel=models[0]; modelSel.value=currentModel;}
  }catch(e){
    modelSel.innerHTML=''; const o=document.createElement('option'); o.value=DEFAULT_MODEL; o.textContent=DEFAULT_MODEL; modelSel.appendChild(o);
  }
}
modelSel.addEventListener('change', async function(){
  const v=modelSel.value;
  if(v.indexOf('puter:')===0){
    const mid=v.slice('puter:'.length);
    setActivePuter(mid);
    if(activeEndpointId) setActiveEndpoint('');
    currentModel=mid;
    if(current){current.model=currentModel; saveChat();}
    return;
  }
  if(v.indexOf('endpoint:')===0){
    const rest=v.slice('endpoint:'.length);
    const sep=rest.indexOf(':');
    const id=rest.slice(0,sep), model=rest.slice(sep+1);
    const ep=endpoints.find(function(e){return e.id===id;});
    if(ep){
      if(ep.model!==model){
        await fetch('/api/endpoints/'+encodeURIComponent(id),{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:model})});
        ep.model=model;
      }
      setActivePuter('');
      setActiveEndpoint(id);
      currentModel=model;
      if(current){current.model=currentModel; saveChat();}
      return;
    }
  }
  currentModel=v;
  if(activeEndpointId) setActiveEndpoint('');
  setActivePuter('');
  if(current){current.model=currentModel; saveChat();}
});

// Verified-free models (tested live 24 Nov 2026 - each answered a test
// prompt). Shown first in the picker as a FREE group so Steven picks a
// free model without guessing. Values match the picker format:
// plain name for local, 'endpoint:<id>:<model>' for endpoints.
const FREE_MODELS=[
 ['qwen3-abliterated-0.6b','Local \u00b7 qwen3-abliterated-0.6b'],
 ['endpoint:18e9cbc6a808cdb9:deepseek-v4-1-flash-1017','Conea \u00b7 deepseek-v4-1-flash-1017'],
 ['endpoint:18e9cbc6a808cdb9:gemini-3-8-flash','Conea \u00b7 gemini-3-8-flash'],
 ['endpoint:18e9cbc6a808cdb9:gpt-6-luna','Conea \u00b7 gpt-6-luna'],
 ['endpoint:18ea2d640f815753:gemini-2.5-flash','Gemini \u00b7 gemini-2.5-flash'],
 ['endpoint:18ea2d640f815753:gemini-2.5-flash-lite','Gemini \u00b7 gemini-2.5-flash-lite'],
 ['endpoint:18ea2d640f815753:gemini-3.5-flash-lite','Gemini \u00b7 gemini-3.5-flash-lite'],
 ['endpoint:18ea2d640f815753:gemini-3-flash-preview','Gemini \u00b7 gemini-3-flash-preview'],
 ['endpoint:18ea2d640f815753:gemini-flash-lite-latest','Gemini \u00b7 gemini-flash-lite-latest'],
 ['endpoint:18ea2d640f815753:gemini-3.6-flash','Gemini \u00b7 gemini-3.6-flash'],
 ['endpoint:18ea4fac9476a655:allam-2-7b','Groq \u00b7 allam-2-7b'],
 ['endpoint:18ea4fac9476a655:openai/gpt-oss-120b','Groq \u00b7 gpt-oss-120b'],
 ['endpoint:18ea4fac9476a655:openai/gpt-oss-20b','Groq \u00b7 gpt-oss-20b'],
 ['endpoint:18ea4fac9476a655:openai/gpt-oss-safeguard-20b','Groq \u00b7 gpt-oss-safeguard-20b'],
 ['endpoint:18ea4fac9476a655:qwen/qwen3.8-27b','Groq \u00b7 qwen3.8-27b'],
 ['endpoint:18ea2e312284c53c:agnes-2.5-flash','Agness \u00b7 agnes-2.5-flash'],
 ['endpoint:18ea2e312284c53c:agnes-3.0-flash','Agness \u00b7 agnes-3.0-flash'],
 ['endpoint:18eb423796aee255:particle','free-pi \u00b7 particle'],
 ['endpoint:18eb423796aee255:deepseek-v4.1-flash','free-pi \u00b7 deepseek-v4.1-flash'],
 ['endpoint:18eb423796aee255:glm-5.3-flash','free-pi \u00b7 glm-5.3-flash'],
 ['endpoint:18eb423796aee255:gpt-oss-20b','free-pi \u00b7 gpt-oss-20b'],
 ['endpoint:18eb423796aee255:solar-mini-4','free-pi \u00b7 solar-mini-4']
];
async function refreshModelSelect(){
  // Combined picker: Local Ollama models + every saved endpoint's models,
  // so Steven picks a model from a list and never types one.
  let ollamaModels=[];
  try{ const r=await fetch('/api/models'); const j=await r.json(); ollamaModels=(j.models&&j.models.length)?j.models:[DEFAULT_MODEL]; }catch(e){ ollamaModels=[DEFAULT_MODEL]; }
  modelSel.innerHTML='';
  const gFree=document.createElement('optgroup'); gFree.label='Free';
  for(const fm of FREE_MODELS){ const o=document.createElement('option'); o.value=fm[0]; o.textContent='FREE \u00b7 '+fm[1]; gFree.appendChild(o); }
  modelSel.appendChild(gFree);
  const gLocal=document.createElement('optgroup'); gLocal.label='Local Ollama';
  for(const m of ollamaModels){const o=document.createElement('option'); o.value=m; o.textContent=m; gLocal.appendChild(o);}
  modelSel.appendChild(gLocal);
  for(const ep of endpoints){
    const list=(ep.models&&ep.models.length)?ep.models:[ep.model];
    const g=document.createElement('optgroup'); g.label=ep.name;
    for(const m of list){
      if(!m) continue;
      const o=document.createElement('option'); o.value='endpoint:'+ep.id+':'+m; o.textContent=ep.name+' · '+m; g.appendChild(o);
    }
    modelSel.appendChild(g);
  }
  if(puterReady()){
    const gP=document.createElement('optgroup'); gP.label='Puter (free)';
    for(const pm of PUTER_MODELS){
      const o=document.createElement('option'); o.value='puter:'+pm.id; o.textContent='Puter · '+pm.label; gP.appendChild(o);
    }
    modelSel.appendChild(gP);
  }
  if(activePuterModel && puterReady()){
    const pv='puter:'+activePuterModel;
    let found=false;
    for(const o of modelSel.options){ if(o.value===pv){ found=true; break; } }
    if(found){ modelSel.value=pv; currentModel=activePuterModel; return; }
    // Clear inline (do NOT call setActivePuter here: it would re-enter refreshModelSelect).
    activePuterModel='';
    try{ localStorage.removeItem('mawaxi_active_puter'); }catch(e){}
  }
  if(activeEndpointId){
    const ep=endpoints.find(function(e){return e.id===activeEndpointId;});
    if(ep){ modelSel.value='endpoint:'+ep.id+':'+ep.model; currentModel=ep.model; return; }
    activeEndpointId='';
    try{ localStorage.removeItem('mawaxi_active_endpoint'); }catch(e){}
  }
  if(ollamaModels.indexOf(currentModel)>=0){ modelSel.value=currentModel; }
  else{ currentModel=ollamaModels[0]; modelSel.value=currentModel; }
}
function setActiveEndpoint(id){
  activeEndpointId=id||'';
  try{
    if(activeEndpointId) localStorage.setItem('mawaxi_active_endpoint', activeEndpointId);
    else localStorage.removeItem('mawaxi_active_endpoint');
  }catch(e){}
  renderEpList(); refreshModelSelect();
}
const PRESET_FALLBACK=[
  {provider:'gemini',label:'Gemini',base_url:'https://generativelanguage.googleapis.com/v1beta/openai/',model:'gemini-2.5-flash'},
  {provider:'nvidia',label:'NVIDIA',base_url:'https://integrate.api.nvidia.com/v1',model:'deepseek-v4.1-flash'},
  {provider:'openai',label:'OpenAI',base_url:'https://api.openai.com/v1',model:'gpt-4o'},
  {provider:'copilot',label:'GitHub Copilot',base_url:'https://api.githubcopilot.com',model:''},
  {provider:'custom',label:'Custom',base_url:'',model:''}
];
async function loadPresets(){
  try{ const r=await fetch('/api/endpoint-presets'); presets=await r.json(); if(!Array.isArray(presets)||!presets.length) presets=PRESET_FALLBACK; }catch(e){ presets=PRESET_FALLBACK; }
  epProvider.innerHTML='';
  for(const p of presets){
    const o=document.createElement('option'); o.value=p.provider; o.textContent=p.label; epProvider.appendChild(o);
  }
  epProvider.value='gemini';
  applyPreset();
}
function applyPreset(){
  const p=presets.find(function(x){return x.provider===epProvider.value;});
  if(!p) return;
  const nameAuto=!epName.value.trim() || presets.some(function(x){return x.label===epName.value.trim();});
  if(p.provider==='custom'){ epUrl.value=''; epModel.value=''; if(nameAuto) epName.value=''; }
  else{
    epUrl.value=p.base_url; epModel.value=p.model;
    if(nameAuto) epName.value=p.label;
  }
  // Provider (or URL) changed: drop any previously fetched model list and
  // fetch this provider's models automatically once the key is typed.
  epFetchedModels=[];
  epModelSel.style.display='none'; epModel.style.display=''; epModelNote.style.display='none';
  // GitHub Copilot signs in with GitHub (device flow) instead of an API key:
  // hide the key/model fields and the Add button, show Sign in instead.
  const isCopilot=p.provider==='copilot';
  epKey.style.display=isCopilot?'none':'';
  epModel.style.display=isCopilot?'none':'';
  document.getElementById('epFetchModels').style.display=isCopilot?'none':'';
  epAdd.style.display=isCopilot?'none':'';
  document.getElementById('copilotSignin').style.display=isCopilot?'':'none';
  if(!isCopilot) document.getElementById('copilotStatus').style.display='none';
  if(epKey.value.trim() && !isCopilot) fetchEpModels();
}
epProvider.addEventListener('change', applyPreset);
// Fetch the provider's model list so Steven picks a model from a dropdown
// instead of typing one. The key is sent to the server, which calls the
// endpoint's /models route itself; the key never reaches the page JS.
async function fetchEpModels(){
  const url=epUrl.value.trim(), key=epKey.value.trim();
  if(!url){ showToast('Base URL is empty'); return; }
  if(!key){ showToast('Type your API key first'); return; }
  epModelNote.style.display='block'; epModelNote.textContent='Fetching models...';
  try{
    const r=await fetch('/api/endpoint-models',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({base_url:url, api_key:key})});
    if(!r.ok) throw new Error('HTTP '+r.status);
    const j=await r.json();
    const list=(j.models||[]).filter(function(m){return m;});
    if(!list.length) throw new Error('empty list');
    epFetchedModels=list;
    epModelSel.innerHTML='';
    for(const m of list){const o=document.createElement('option'); o.value=m; o.textContent=m; epModelSel.appendChild(o);}
    const preset=presets.find(function(x){return x.provider===epProvider.value;});
    if(preset && list.indexOf(preset.model)>=0) epModelSel.value=preset.model;
    else if(list.indexOf(epModel.value.trim())>=0) epModelSel.value=epModel.value.trim();
    epModelSel.style.display=''; epModel.style.display='none';
    epModelNote.textContent=list.length+' models found — pick one.';
  }catch(e){
    epFetchedModels=[];
    epModelSel.style.display='none'; epModel.style.display='';
    epModelNote.textContent='Could not fetch models — type one by hand.';
  }
}
epFetchModels.addEventListener('click', fetchEpModels);
epKey.addEventListener('change', function(){ if(epUrl.value.trim()) fetchEpModels(); });
async function loadEndpoints(){
  try{ const r=await fetch('/api/endpoints'); endpoints=await r.json(); }catch(e){ endpoints=[]; }
  renderEpList(); await refreshModelSelect();
  // An endpoint added before model caching has no cached list: fetch it once
  // in the background so its models appear in the picker automatically.
  if(activeEndpointId){
    const ep=endpoints.find(function(e){return e.id===activeEndpointId;});
    if(ep && !(ep.models && ep.models.length)){
      try{
        const r=await fetch('/api/endpoints/'+encodeURIComponent(ep.id)+'/models');
        if(r.ok){ const j=await r.json(); ep.models=j.models||[]; renderEpList(); await refreshModelSelect(); }
      }catch(e){}
    }
  }
}
function renderEpList(){
  epList.innerHTML='';
  const localRow=document.createElement('div'); localRow.className='listrow';
  const li=document.createElement('div'); li.className='info';
  const lnm=document.createElement('div'); lnm.className='nm'; lnm.textContent='Local Ollama';
  const lsub=document.createElement('div'); lsub.className='sub'; lsub.textContent='On-phone models (default)';
  li.appendChild(lnm); li.appendChild(lsub);
  const useLocal=document.createElement('button'); useLocal.className='mini'+(!activeEndpointId?' on':''); useLocal.textContent=activeEndpointId?'Use':'Active';
  useLocal.addEventListener('click', function(){ setActiveEndpoint(''); });
  localRow.appendChild(li); localRow.appendChild(useLocal); epList.appendChild(localRow);
  for(const ep of endpoints){
    const row=document.createElement('div'); row.className='listrow';
    const info=document.createElement('div'); info.className='info';
    const nm=document.createElement('div'); nm.className='nm'; nm.textContent=ep.name;
    const sub=document.createElement('div'); sub.className='sub'; sub.textContent=(ep.provider&&ep.provider!=='custom'?ep.provider+' · ':'')+ep.base_url+' · '+ep.model;
    info.appendChild(nm); info.appendChild(sub);
    if(ep.models && ep.models.length){
      const ms=document.createElement('select'); ms.className='fld'; ms.style.margin='6px 0 0';
      const opts=ep.models.indexOf(ep.model)>=0?ep.models.slice():[ep.model].concat(ep.models);
      for(const m of opts){const o=document.createElement('option'); o.value=m; o.textContent=m; ms.appendChild(o);}
      ms.value=ep.model;
      ms.addEventListener('change', async function(){
        await fetch('/api/endpoints/'+encodeURIComponent(ep.id),{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:ms.value})});
        await loadEndpoints();
      });
      info.appendChild(ms);
    }
    const use=document.createElement('button'); use.className='mini'+(activeEndpointId===ep.id?' on':''); use.textContent=activeEndpointId===ep.id?'Active':'Use';
    use.addEventListener('click', function(){ setActiveEndpoint(ep.id); });
    const ref=document.createElement('button'); ref.className='mini'; ref.textContent='Refresh'; ref.title='Refresh model list';
    ref.addEventListener('click', async function(){
      const r=await fetch('/api/endpoints/'+encodeURIComponent(ep.id)+'/models');
      if(!r.ok){ showToast('Could not fetch models'); return; }
      showToast('Models updated'); await loadEndpoints();
    });
    const del=document.createElement('button'); del.className='mini del'; del.textContent='x'; del.title='Delete endpoint';
    del.addEventListener('click', async function(){
      await fetch('/api/endpoints/'+encodeURIComponent(ep.id), {method:'DELETE'});
      if(activeEndpointId===ep.id){
        activeEndpointId='';
        try{ localStorage.removeItem('mawaxi_active_endpoint'); }catch(e){}
      }
      await loadEndpoints();
    });
    row.appendChild(info); row.appendChild(use); row.appendChild(ref); row.appendChild(del); epList.appendChild(row);
  }
}
epAdd.addEventListener('click', async function(){
  const name=epName.value.trim(), url=epUrl.value.trim(), key=epKey.value.trim();
  const model=(epModelSel.style.display!=='none' && epModelSel.value)?epModelSel.value:epModel.value.trim();
  if(!name||!url||!model){ showToast('Name, base URL and model are required'); return; }
  const r=await fetch('/api/endpoints',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name, base_url:url, api_key:key, model:model, provider:epProvider.value, models:epFetchedModels})});
  if(!r.ok){ showToast('Could not add endpoint ('+r.status+')'); return; }
  epName.value=''; epUrl.value=''; epKey.value=''; epModel.value=''; epFetchedModels=[];
  epModelSel.style.display='none'; epModel.style.display=''; epModelNote.style.display='none';
  epProvider.value='gemini'; applyPreset();
  showToast('Endpoint added'); await loadEndpoints();
});

// GitHub Copilot sign-in (device flow): the server asks GitHub for a code,
// Steven authorises it on github.com, then the server saves the Copilot
// endpoint automatically. No API key to paste.
let copilotPollTimer=null;
async function startCopilotSignin(){
  const status=document.getElementById('copilotStatus');
  status.style.display='block'; status.textContent='Asking GitHub for a sign-in code...';
  let r;
  try{ r=await fetch('/api/copilot/device/start',{method:'POST'}); }catch(e){ status.textContent='Could not reach the server.'; return; }
  if(!r.ok){ status.textContent='GitHub did not give a code (HTTP '+r.status+'). Try again.'; return; }
  const j=await r.json();
  const link=j.verification_uri_complete||j.verification_uri||'https://github.com/login/device';
  status.innerHTML='Your GitHub code is <b>'+j.user_code+'</b>. Open <a href="'+link+'" target="_blank" rel="noopener">github.com/login/device</a> and enter the code, then wait here.';
  if(copilotPollTimer) clearInterval(copilotPollTimer);
  const interval=Math.max(5,(j.interval||5))*1000;
  const deadline=Date.now()+Math.max(60,(j.expires_in||900))*1000;
  copilotPollTimer=setInterval(async function(){
    if(Date.now()>deadline){ clearInterval(copilotPollTimer); copilotPollTimer=null; status.textContent='The code expired. Tap Sign in with GitHub again for a new code.'; return; }
    let pr;
    try{ pr=await fetch('/api/copilot/device/poll',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({poll_id:j.poll_id})}); }catch(e){ return; }
    if(!pr.ok){ clearInterval(copilotPollTimer); copilotPollTimer=null; status.textContent='Sign-in check failed (HTTP '+pr.status+'). Try again.'; return; }
    const pj=await pr.json();
    if(pj.status==='authorized'){
      clearInterval(copilotPollTimer); copilotPollTimer=null;
      status.textContent='GitHub Copilot added. Pick a Copilot model from the model picker.';
      showToast('GitHub Copilot added');
      await loadEndpoints(); refreshModelSelect();
    }else if(pj.status==='failed'){
      clearInterval(copilotPollTimer); copilotPollTimer=null;
      status.textContent='Sign-in did not complete ('+(pj.error||'denied')+'). Tap Sign in with GitHub again.';
    }
  }, interval);
}
document.getElementById('copilotSignin').addEventListener('click', startCopilotSignin);
async function loadMemories(){
  try{ const r=await fetch('/api/memories'); memories=await r.json(); }catch(e){ memories=[]; }
  renderMemList();
}
function renderMemList(){
  memList.innerHTML='';
  if(!memories.length){ const p=document.createElement('p'); p.className='hint'; p.textContent='No memories yet. Add one above.'; memList.appendChild(p); return; }
  for(const m of memories){
    const row=document.createElement('div'); row.className='listrow';
    const info=document.createElement('div'); info.className='info';
    const sub=document.createElement('div'); sub.className='sub2'; sub.textContent=m.text;
    info.appendChild(sub);
    const del=document.createElement('button'); del.className='mini del'; del.textContent='x'; del.title='Delete memory';
    del.addEventListener('click', async function(){ await fetch('/api/memories/'+encodeURIComponent(m.id), {method:'DELETE'}); await loadMemories(); });
    row.appendChild(info); row.appendChild(del); memList.appendChild(row);
  }
}
memAdd.addEventListener('click', async function(){
  const text=memText.value.trim(); if(!text){ showToast('Type a fact first'); return; }
  const r=await fetch('/api/memories',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text})});
  if(!r.ok){ showToast('Could not add memory ('+r.status+')'); return; }
  memText.value=''; showToast('Memory saved'); await loadMemories();
});
async function loadSkills(){
  try{ const r=await fetch('/api/skills'); skills=await r.json(); }catch(e){ skills=[]; }
  renderSkList(); updateSkillChip();
}
function renderSkList(){
  skList.innerHTML='';
  if(!skills.length){ const p=document.createElement('p'); p.className='hint'; p.textContent='No skills yet. Add one above.'; skList.appendChild(p); return; }
  for(const s of skills){
    const row=document.createElement('div'); row.className='listrow';
    const info=document.createElement('div'); info.className='info';
    const nm=document.createElement('div'); nm.className='nm'; nm.textContent=s.name;
    const sub=document.createElement('div'); sub.className='sub2'; sub.textContent=s.instruction;
    info.appendChild(nm); info.appendChild(sub);
    const use=document.createElement('button'); use.className='mini'+(current && current.skill_id===s.id?' on':''); use.textContent=(current && current.skill_id===s.id)?'Active':'Use';
    use.addEventListener('click', function(){
      if(!current) current=newChatObj();
      current.skill_id=s.id; saveChat(); updateSkillChip(); renderSkList(); closeModal(); showToast('Skill "'+s.name+'" active for this chat');
    });
    const del=document.createElement('button'); del.className='mini del'; del.textContent='x'; del.title='Delete skill';
    del.addEventListener('click', async function(){
      await fetch('/api/skills/'+encodeURIComponent(s.id), {method:'DELETE'});
      if(current && current.skill_id===s.id){ current.skill_id=null; saveChat(); }
      await loadSkills();
    });
    row.appendChild(info); row.appendChild(use); row.appendChild(del); skList.appendChild(row);
  }
}
skAdd.addEventListener('click', async function(){
  const name=skName.value.trim(), instr=skInstr.value.trim();
  if(!name||!instr){ showToast('Name and instruction are required'); return; }
  const r=await fetch('/api/skills',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name, instruction:instr})});
  if(!r.ok){ showToast('Could not add skill ('+r.status+')'); return; }
  skName.value=''; skInstr.value=''; showToast('Skill added'); await loadSkills();
});
function updateSkillChip(){
  const sid=current && current.skill_id;
  if(sid){
    const sk=skills.find(function(s){return s.id===sid;});
    if(sk){ skillChipName.textContent='Skill: '+sk.name; skillChip.style.display='flex'; return; }
  }
  skillChip.style.display='none';
}
function effectiveModel(){
  if(activePuterModel){
    return {name: activePuterModel, vision: false, via: 'puter'};
  }
  if(activeEndpointId){
    const ep=endpoints.find(function(e){return e.id===activeEndpointId;});
    // Gemini endpoints support vision (image input). Other providers are
    // treated as no-vision until verified.
    if(ep) return {name: ep.model, vision: (ep.provider||'').toLowerCase()==='gemini', via: 'endpoint'};
  }
  // Local model: the model itself is text-only, but the server describes
  // an attached photo on-device with Mirage and gives the text description
  // to the model, so from the user's side it can see images now.
  return {name: currentModel, vision: true, via: 'local'};
}

async function loadChats(){
  const r=await fetch('/api/chats'); chats=await r.json();
  renderChatList();
}
function fmtChatDate(ts){
  if(!ts) return '';
  const d=new Date(ts*1000);
  const now=new Date();
  const months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const hh=String(d.getHours()).padStart(2,'0'), mm=String(d.getMinutes()).padStart(2,'0');
  const day=d.getDate()+' '+months[d.getMonth()];
  if(d.toDateString()===now.toDateString()) return 'Today '+hh+':'+mm;
  if(d.getFullYear()===now.getFullYear()) return day+' '+hh+':'+mm;
  return day+' '+d.getFullYear();
}
function renderChatList(){
  chatlist.innerHTML='';
  for(const c of chats){
    const div=document.createElement('div');
    div.className='chatitem'+(current && c.id===current.id?' active':'');
    const tx=document.createElement('div'); tx.className='tx';
    const t=document.createElement('span'); t.className='t'; t.textContent=c.title||'New chat';
    const dd=document.createElement('span'); dd.className='d'; dd.textContent=fmtChatDate(c.updated_at);
    tx.appendChild(t); tx.appendChild(dd);
    const del=document.createElement('button'); del.className='del'; del.textContent='x'; del.title='Delete chat';
    del.addEventListener('click', function(e){e.stopPropagation(); deleteChat(c.id);});
    div.appendChild(tx); div.appendChild(del);
    div.addEventListener('click', function(){openChat(c.id);});
    chatlist.appendChild(div);
  }
}
async function openChat(id){
  const r=await fetch('/api/chats/'+encodeURIComponent(id));
  if(!r.ok) return;
  current=await r.json();
  currentModel=current.model||DEFAULT_MODEL;
  if(currentModel.indexOf('puter:')===0){ setActivePuter(currentModel.slice(6)); currentModel=activePuterModel; if(activeEndpointId) setActiveEndpoint(''); }
  else if(PUTER_MODELS.some(function(p){return p.id===currentModel;})){ setActivePuter(currentModel); if(activeEndpointId) setActiveEndpoint(''); }
  else if(activeEndpointId){
    const ep=endpoints.find(function(e){return e.id===activeEndpointId;});
    if(ep) currentModel=ep.model;
  }
  let found=false;
  for(const o of modelSel.options){ if(o.value===currentModel){ found=true; break; } }
  if(found) modelSel.value=currentModel;
  renderMessages();
  renderChatList();
  updateSkillChip();
  closeDrawer();
}
function showEmpty(show){
  emptyEl.style.display=show?'flex':'none';
}
function renderMessages(){
  log.innerHTML='';
  if(!current || !current.messages.length){
    showEmpty(true); return;
  }
  showEmpty(false);
  for(let i=0;i<current.messages.length;i++){const m=current.messages[i];appendMessage(m.role, m.content, true, m.image||null, m.images||null, m.video||null, i, m.source||null);}
  log.scrollTop=log.scrollHeight;
}
function appendMessage(role, content, rendered, imageUrl, images, videoUrl, msgIndex, sourceUrl){
  showEmpty(false);
  const d=document.createElement('div');
  d.className='msg '+(role==='user'?'user':'ai');
  if(msgIndex!==undefined && msgIndex!==null){ d.dataset.msgIndex=msgIndex; }
  const b=document.createElement('div'); b.className='body';
  if(role==='user'){
    if(images && images.length){
      for(const im of images){
        const t=document.createElement('img'); t.className='imgthumb';
        t.src = im.indexOf('data:')===0 ? im : 'data:image/jpeg;base64,'+im;
        t.alt='Attached image'; b.appendChild(t);
      }
    }
    const tx=document.createElement('div'); tx.textContent=content; b.appendChild(tx);
    const rb=document.createElement('button');
    rb.type='button'; rb.className='resendbtn'; rb.textContent='Resend';
    rb.disabled=chatBusy||agentRunning;
    b.appendChild(rb);
  }
  else if(imageUrl){
    b.innerHTML='';
    const cap=document.createElement('div'); cap.textContent=content; cap.style.cssText='margin-bottom:8px;color:var(--muted);font-size:13px;';
    b.appendChild(cap);
    // Real web photo (sourceUrl set): a failed load becomes a link to
    // the photo itself, and a Source link is shown under it. Generated
    // images (no source) keep the old behaviour.
    const mkImg=function(){
      const img=document.createElement('img'); img.src=imageUrl; img.alt=content;
      img.style.cssText='max-width:100%;border-radius:12px;display:block;';
      if(sourceUrl){
        img.addEventListener('error', function(){
          const a=document.createElement('a'); a.href=imageUrl; a.target='_blank'; a.rel='noopener';
          a.textContent=(content||'Image')+' (image link)';
          img.replaceWith(a);
        });
      }else{
        img.addEventListener('error', function(){ b.textContent='Image failed to load — it may have expired. Generate it again.'; });
      }
      return img;
    };
    if(dataSaver){
      b.appendChild(tapToLoadBox(content, function(box){ box.replaceWith(mkImg()); }));
    }else{
      b.appendChild(mkImg());
    }
    if(sourceUrl){
      const src=document.createElement('a'); src.href=sourceUrl; src.target='_blank'; src.rel='noopener';
      src.textContent='Source';
      src.style.cssText='color:var(--muted);font-size:13px;display:inline-block;margin-top:8px;';
      b.appendChild(src);
    }
    b.dataset.fullText=content;
  }
  else if(videoUrl){
    b.innerHTML='';
    const cap=document.createElement('div'); cap.textContent=content; cap.style.cssText='margin-bottom:8px;color:var(--muted);font-size:13px;';
    b.appendChild(cap);
    if(dataSaver){
      b.appendChild(tapToLoadBox(content, function(box){
        box.remove();
        const vid=document.createElement('video'); vid.src=videoUrl; vid.controls=true; vid.loop=true; vid.muted=true; vid.playsInline=true;
        vid.style.cssText='max-width:100%;border-radius:12px;display:block;background:#000;';
        vid.addEventListener('error', function(){ b.textContent='Video failed to load — it may have expired. Generate it again.'; });
        const dl=document.createElement('a'); dl.href=videoUrl; dl.textContent='Download video'; dl.style.cssText='color:var(--muted);font-size:13px;display:inline-block;margin-top:8px;';
        b.appendChild(vid); b.appendChild(dl);
      }));
    }else{
      const vid=document.createElement('video'); vid.src=videoUrl; vid.controls=true; vid.loop=true; vid.muted=true; vid.playsInline=true;
      vid.style.cssText='max-width:100%;border-radius:12px;display:block;background:#000;';
      vid.addEventListener('error', function(){ b.textContent='Video failed to load — it may have expired. Generate it again.'; });
      const dl=document.createElement('a'); dl.href=videoUrl; dl.textContent='Download video'; dl.style.cssText='color:var(--muted);font-size:13px;display:inline-block;margin-top:8px;';
      b.appendChild(vid); b.appendChild(dl);
    }
    b.dataset.fullText=content;
  }
  else{
    b.innerHTML = rendered ? md(content) : esc(content);
    b.dataset.fullText=content;
  }
  d.appendChild(b);
  log.appendChild(d);
  return b;
}
async function saveChat(){
  if(!current || !current.messages.length) return;
  try{
    await fetch('/api/chats/'+encodeURIComponent(current.id), {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(current)});
    const r=await fetch('/api/chats'); chats=await r.json(); renderChatList();
  }catch(e){}
}
async function deleteChat(id){
  await fetch('/api/chats/'+encodeURIComponent(id), {method:'DELETE'});
  chats=chats.filter(function(c){return c.id!==id;});
  if(current && current.id===id){current=newChatObj(); renderMessages();}
  renderChatList();
}
newchat.addEventListener('click', function(){
  current=newChatObj();
  currentModel=modelSel.value||DEFAULT_MODEL; current.model=currentModel;
  if(currentModel.indexOf('puter:')===0){ setActivePuter(currentModel.slice(6)); currentModel=activePuterModel; current.model=currentModel; }
  renderMessages(); renderChatList(); updateSkillChip(); closeDrawer(); q.focus();
});

// Safety net only (auto-illustration removed 2026-11-21 at Steven's
// word: normal chat replies must not include AI-generated images).
// If a model ever emits a stray [[IMAGE: <prompt>]] line on its own,
// extract it and strip it from the text — the prompt is returned but
// NO image is generated from it. Lenient: case-insensitive, first
// marker only, and an unclosed marker at the very end still counts.
function extractImageMarker(text){
  var t=(text||'');
  var m=t.match(/\[\[IMAGE:\s*([^\]\n]+)\]\]/i);
  if(m){
    var prompt=m[1].trim();
    var clean=t.replace(m[0],'').replace(/\n{3,}/g,'\n\n').trim();
    return {text:clean, prompt:prompt||null};
  }
  var m2=t.match(/\[\[IMAGE:\s*([^\n]+)$/i);
  if(m2){
    var prompt2=m2[1].replace(/\]\]$/,'').trim();
    var clean2=t.slice(0,m2.index).replace(/\n{3,}/g,'\n\n').trim();
    return {text:clean2, prompt:prompt2||null};
  }
  return {text:t, prompt:null};
}
// While streaming, hide a marker that is still arriving token by token
// (the model puts it last, so nothing follows it) as well as completed
// ones, so the marker is never visible to the user.
function stripMarkerLive(text){
  var t=(text||'');
  var m=t.match(/\[\[IMAGE:/i);
  if(m) return t.slice(0,m.index);
  var tail='[[IMAGE:';
  for(var n=tail.length-1;n>=1;n--){
    if(t.endsWith(tail.slice(0,n))) return t.slice(0,t.length-n);
  }
  return t;
}

// Real web photos in plain chat (added 2026-11-21 at Steven's word:
// a reply about something visible should show REAL photos from the
// web, never AI-generated ones). The model ends an image-worthy reply
// with [[WEBIMAGE: <search query>]]. Extract it and strip it — the
// query is returned but the marker is never shown. Lenient:
// case-insensitive, first marker only, and an unclosed marker at the
// very end still counts. Mirrors extractImageMarker, which is a
// different marker: [[IMAGE: ...]] is stripped and NEVER turned into
// an image, while [[WEBIMAGE: ...]] triggers a real web photo search.
function extractWebImageMarker(text){
  var t=(text||'');
  var m=t.match(/\[\[WEBIMAGE:\s*([^\]\n]+)\]\]/i);
  if(m){
    var query=m[1].trim();
    var clean=t.replace(m[0],'').replace(/\n{3,}/g,'\n\n').trim();
    return {text:clean, query:query||null};
  }
  var m2=t.match(/\[\[WEBIMAGE:\s*([^\n]+)$/i);
  if(m2){
    var query2=m2[1].replace(/\]\]$/,'').trim();
    var clean2=t.slice(0,m2.index).replace(/\n{3,}/g,'\n\n').trim();
    return {text:clean2, query:query2||null};
  }
  return {text:t, query:null};
}
// While streaming, hide a [[WEBIMAGE: ...]] marker that is still
// arriving token by token (the model puts it last, so nothing follows
// it) as well as completed ones, so the marker is never visible.
function stripWebImageMarkerLive(text){
  var t=(text||'');
  var m=t.match(/\[\[WEBIMAGE:/i);
  if(m) return t.slice(0,m.index);
  var tail='[[WEBIMAGE:';
  for(var n=tail.length-1;n>=1;n--){
    if(t.endsWith(tail.slice(0,n))) return t.slice(0,t.length-n);
  }
  return t;
}
// Fetch real web photos for a chat reply and show them under it.
// Uses POST /api/web-image (SearXNG, the same search the agent's
// search_images tool uses). Up to 3 photos, each saved as its own
// assistant message so reopening the chat shows them under the reply.
// Data Saver is respected by the image bubble (tap to load). If the
// search finds nothing, only a small muted line is shown — the reply
// text is never blocked or broken.
async function attachWebImages(query){
  let results=[];
  try{
    const res=await fetch('/api/web-image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:query})});
    if(!res.ok) throw new Error('server error '+res.status);
    const j=await res.json(); results=j.results||[];
  }catch(err){
    results=[];
  }
  if(!results.length){
    const note=document.createElement('div');
    note.textContent='No image found for this.';
    note.style.cssText='color:var(--muted);font-size:13px;margin:4px 0 8px;';
    log.appendChild(note);
    log.scrollTop=log.scrollHeight;
    return;
  }
  for(const r of results.slice(0,3)){
    if(!r.image_url) continue;
    const title=r.title||'Web image';
    current.messages.push({role:'assistant', content:title, image:r.image_url, source:r.source_url||null});
    appendMessage('assistant', title, true, r.image_url, null, null, current.messages.length-1, r.source_url||null);
  }
  log.scrollTop=log.scrollHeight;
  saveChat();
}

// Detect a plain-language image request and return the image prompt,
// or null when the message is not an image request. Lets the user just
// ask for an image in chat ("make an image of a red bicycle") instead
// of needing the "image:" prefix. Kept conservative so questions about
// images ("what is an image", "describe this image") still go to chat.

// Detect a plain-language PDF request and return what to put in the PDF,
// or null when the message is not a PDF request. Mirrors the image/video
// detectors: "make a PDF of this chat" just works, no prefix.
// Questions about PDFs ("what is a PDF", "how do I open a PDF") still
// go to chat. kind: 'chat' = whole chat transcript, 'prev' = latest
// assistant reply, 'topic' = latest assistant reply titled by the topic
// (or the topic text itself when there is no reply yet),
// 'text' = the text the user typed after a "pdf:" prefix.
function pdfRequestFromText(msg){
  var m=(msg||'').trim();
  if(!m) return null;
  var low=m.toLowerCase();
  var pre=low.match(/^pdf\s*:\s*/);
  if(pre){
    var t=m.slice(pre[0].length).trim();
    return t ? {kind:'text', topic:t} : null;
  }
  var isReq=/^(?:please\s+)?(?:can you\s+|could you\s+)?(?:make|create|generate|produce|build|give me|send me|put|write)\b.*\bpdf\b/i.test(m)
    || /^(?:please\s+)?(?:save|export|download|convert|turn)\b.*\bpdf\b/i.test(m)
    || /^i\s+(?:want|need|would like|'d\s+like)\b.*\bpdf\b/i.test(m)
    || /^pdf\s+of\b/i.test(m);
  if(!isReq) return null;
  if(/\b(chat|conversation|discussion|talk)\b/i.test(m)) return {kind:'chat', topic:''};
  if(/\b(answer|reply|response|explanation)\b/i.test(m)) return {kind:'prev', topic:''};
  var topic='';
  var tm=m.match(/\bpdf\s+(?:of|about|on|for)\s+(.+)$/i);
  if(tm){ topic=tm[1].trim(); }
  else{
    var tm2=m.match(/\bpdf\b\s+(.+)$/i);
    if(tm2){ topic=tm2[1].trim().replace(/^(?:of|about|on|for)\s+/i,''); }
  }
  topic=topic.replace(/\s+(?:please|for me)\s*$/i,'').trim();
  if(!topic) return {kind:'chat', topic:''};
  if(/^(this|that|it|above|the above|the previous|the last|that one|this one)$/i.test(topic)) return {kind:'prev', topic:''};
  return {kind:'topic', topic:topic};
}

function lastAssistantText(){
  if(!current) return null;
  for(var i=current.messages.length-1;i>=0;i--){
    if(current.messages[i].role==='assistant' && current.messages[i].content) return current.messages[i].content;
  }
  return null;
}

async function generatePdfFromInput(req, skipUser){
  if(!current) current=newChatObj();
  if(!skipUser){
    var userText=q.value.trim();
    q.value='';
    if(!current.messages.length){current.title=userText.slice(0,40);}
    current.messages.push({role:'user', content:userText});
    appendMessage('user', userText, true, null, null, null, current.messages.length-1);
    log.scrollTop=log.scrollHeight;
  }
  var title='', text='', segments=[];
  // Photos that belong to the content being PDF'd, in reading order.
  function replySegments(){
    var aiIdx=-1;
    for(var i=current.messages.length-1;i>=0;i--){ if(current.messages[i].role==='assistant' && current.messages[i].content){ aiIdx=i; break; } }
    if(aiIdx<0) return segmentsForMessages(current.messages);
    return segmentsForMessages(current.messages.slice(aiIdx).filter(function(x){ return x.role==='assistant'; }));
  }
  if(req.kind==='chat'){
    var lines=[];
    for(const m of current.messages){
      lines.push((m.role==='user'?'You: ':'Mawaxi AI: ')+m.content);
      lines.push('');
    }
    text=lines.join('\n');
    title=(current.title||'Mawaxi AI chat').slice(0,60);
    segments=segmentsForMessages(current.messages);
  }else if(req.kind==='text'){
    text=req.topic;
    title=req.topic.replace(/\s+/g,' ').slice(0,40);
  }else if(req.kind==='prev'){
    text=lastAssistantText()||'';
    if(!text){
      var lines2=[];
      for(const m of current.messages){
        lines2.push((m.role==='user'?'You: ':'Mawaxi AI: ')+m.content);
        lines2.push('');
      }
      text=lines2.join('\n');
      title=(current.title||'Mawaxi AI chat').slice(0,60);
      segments=segmentsForMessages(current.messages);
    }else{
      title=text.replace(/\s+/g,' ').slice(0,40);
      segments=replySegments();
    }
  }else{
    var prev=lastAssistantText();
    // A 'topic' PDF needs the content about the topic, not just the
    // topic words and not whatever the assistant said last about
    // something else. Use the earlier reply only when it is actually
    // about this topic; otherwise ask the model to write it first,
    // then PDF that reply. (Steven, 2026-11-23: "make a PDF of
    // African countries" made an empty PDF; "make a PDF of Uganda
    // anthem" PDF'd the previous download message instead.)
    var stop={of:1,the:1,a:1,an:1,and:1,with:1,about:1,on:1,for:1,to:1,in:1,by:1,from:1,at:1,is:1,are:1};
    var tw=String(req.topic||'').toLowerCase().replace(/[^a-z0-9 ]+/g,' ').split(/\s+/).filter(function(w){ return w.length>=3 && !stop[w]; });
    var prevAbout=false;
    if(prev && tw.length){
      var pl=prev.toLowerCase(); var hits=0;
      for(var i=0;i<tw.length;i++){ if(pl.indexOf(tw[i])>=0) hits++; }
      prevAbout = hits*2>=tw.length;
    }
    if(!prevAbout){
      const genBody=appendMessage('assistant', 'Writing about '+req.topic+'\u2026', false);
      let genText='';
      try{
        const gres=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:current.model||currentModel, messages:[{role:'user', content:'Write about: '+req.topic}], endpoint_id: activeEndpointId||null, skill_id: null, persona: currentPersona})});
        if(!gres.ok) throw new Error('server error '+gres.status);
        const reader=gres.body.getReader(); const dec=new TextDecoder();
        while(true){
          const r2=await reader.read(); if(r2.done) break;
          const chunk=dec.decode(r2.value,{stream:true});
          for(const line of chunk.split('\n')){
            if(!line.startsWith('data:')) continue;
            let d=line.slice(5); if(d.startsWith(' ')) d=d.slice(1);
            if(d==='[DONE]'||d==='') continue;
            genText+=d;
            genBody.textContent=stripWebImageMarkerLive(stripMarkerLive(genText));
          }
          log.scrollTop=log.scrollHeight;
        }
      }catch(e){
        genBody.textContent='(could not write about this topic: '+e.message+')';
        current.messages.push({role:'assistant', content:genBody.textContent});
        saveChat(); return;
      }
      genText=extractWebImageMarker(extractImageMarker(genText).text).text;
      genBody.innerHTML=md(genText);
      genBody.dataset.fullText=genText;
      current.messages.push({role:'assistant', content:genText});
      saveChat();
      text=genText;
      title=req.topic.slice(0,60);
      segments=replySegments();
    }else{
      text=prev;
      title=req.topic.slice(0,60);
      segments=replySegments();
    }
  }
  if(!text || !text.trim()){
    appendMessage('assistant', 'Nothing to put in the PDF yet \u2014 ask something first, then ask for the PDF.', false);
    current.messages.push({role:'assistant', content:'Nothing to put in the PDF yet \u2014 ask something first, then ask for the PDF.'});
    saveChat(); return;
  }
  const aiBody=appendMessage('assistant', 'Creating PDF\u2026', false);
  try{
    const res=await fetch('/api/pdf/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:title, text:text, segments:segments})});
    if(!res.ok){ throw new Error(await res.text()||('server error '+res.status)); }
    const j=await res.json();
    aiBody.innerHTML='';
    const nImg=j.images_embedded||0, nFail=j.images_failed||0;
    const cap=document.createElement('div'); cap.textContent='PDF: '+j.name+(nImg>0 ? ' ('+nImg+' image'+(nImg===1?'':'s')+')' : ''); cap.style.cssText='margin-bottom:8px;color:var(--muted);font-size:13px;';
    const dl=document.createElement('a'); dl.href=j.url; dl.download=j.name; dl.textContent='Download '+j.name; dl.style.cssText='color:var(--green);font-weight:600;';
    aiBody.appendChild(cap); aiBody.appendChild(dl);
    if(nFail>0){ const fn=document.createElement('div'); fn.textContent=nFail+' image'+(nFail===1?'':'s')+' could not be loaded \u2014 their links are in the PDF.'; fn.style.cssText='color:var(--muted);font-size:13px;margin-top:6px;'; aiBody.appendChild(fn); }
    current.messages.push({role:'assistant', content:'PDF created: '+j.name+' \u2014 Download it here: ['+j.name+']('+j.url+')'});
    await loadPdfList();
    showToast(nImg>0 ? ('PDF saved ('+nImg+' image'+(nImg===1?'':'s')+')') : 'PDF saved');
  }catch(err){
    const msg2='PDF failed: '+err.message;
    aiBody.textContent=msg2;
    current.messages.push({role:'assistant', content:msg2});
  }
  log.scrollTop=log.scrollHeight;
  saveChat();
}

function imagePromptFromText(msg){
  var m=(msg||'').trim();
  if(!m) return null;
  var low=m.toLowerCase();
  var pre=low.match(/^(image|img|picture|photo|pic)\s*:\s*/);
  if(pre){
    return m.slice(pre[0].length).trim();
  }
  var patterns=[
    /^(?:please\s+)?(?:can you\s+|could you\s+)?(?:make|generate|create|draw|paint|sketch|render|design|produce)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:image|picture|photo|pic|drawing|illustration|artwork|poster|logo)(?:\s+of)?\s*/i,
    /^(?:please\s+)?(?:show|give)\s+me\s+(?:a\s+|an\s+|the\s+)?(?:image|picture|photo|pic)(?:\s+of)?\s*/i,
    /^(?:i\s+(?:want|need)|i'?d\s+like)\s+(?:a\s+|an\s+|the\s+)?(?:image|picture|photo|pic)(?:\s+of)?\s*/i,
    /^(?:a\s+|an\s+|the\s+)?(?:image|picture|photo)\s+of\s+/i,
    /^(?:draw|paint|sketch)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?/i
  ];
  for(var i=0;i<patterns.length;i++){
    var mm=m.match(patterns[i]);
    if(mm){
      var rest=m.slice(mm[0].length).trim().replace(/^of\s+/i,'');
      return rest||null;
    }
  }
  return null;
}

async function generateImageFromInput(promptArg, skipUser){
  let prompt=(promptArg!==undefined && promptArg!==null) ? String(promptArg).trim() : q.value.trim();
  if(prompt.toLowerCase().indexOf('image:')===0) prompt=prompt.slice(6).trim();
  if(!prompt){ showToast('Type an image description first'); q.focus(); return; }
  if(promptArg===undefined || promptArg===null){ q.value=''; }
  if(!current) current=newChatObj();
  if(!skipUser){
    if(!current.messages.length){current.title=prompt.slice(0,40);}
    current.messages.push({role:'user', content:prompt});
    appendMessage('user', prompt, true, null, null, null, current.messages.length-1);
    log.scrollTop=log.scrollHeight;
  }
  const aiBody=appendMessage('assistant', 'Generating image...', false);
  let url='';
  try{
    const res=await fetch('/api/image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:prompt, small:dataSaver})});
    if(!res.ok){ throw new Error('server error '+res.status); }
    const j=await res.json(); url=j.url;
  }catch(err){
    const msg='Image generation failed: '+err.message+'. Check your connection and try again.';
    aiBody.textContent=msg;
    current.messages.push({role:'assistant', content:msg});
    saveChat(); return;
  }
  aiBody.innerHTML='';
  const cap=document.createElement('div'); cap.textContent='Image: '+prompt; cap.style.cssText='margin-bottom:8px;color:var(--muted);font-size:13px;';
  const img=document.createElement('img'); img.alt=prompt;
  img.style.cssText='max-width:100%;border-radius:12px;display:block;';
  const status=document.createElement('div'); status.style.cssText='color:var(--muted);font-size:13px;margin-bottom:8px;';
  let attempts=0; const maxAttempts=4;
  const delays=[0,10000,25000,45000];
  function tryLoad(){
    status.textContent = attempts===0 ? 'Generating image\u2026' : 'Pollinations free limit (1 image per 45s) \u2014 retrying\u2026 (attempt '+(attempts+1)+' of '+maxAttempts+')';
    img.src=url;
    attempts++;
  }
  img.addEventListener('load', function(){ status.textContent=''; });
  img.addEventListener('error', function(){
    if(attempts<maxAttempts){ setTimeout(tryLoad, delays[attempts]); }
    else{
      status.textContent='';
      aiBody.textContent='Image failed to load \u2014 Pollinations may be unreachable or its free limit was hit. Wait a minute and try again.';
    }
  });
  aiBody.appendChild(cap); aiBody.appendChild(status);
  if(dataSaver){
    // Data Saver: the image is not downloaded until the user taps.
    aiBody.appendChild(tapToLoadBox('Image: '+prompt, function(box){ box.replaceWith(img); tryLoad(); }));
  }else{
    aiBody.appendChild(img);
    tryLoad();
  }
  current.messages.push({role:'assistant', content:'Image: '+prompt, image:url});
  log.scrollTop=log.scrollHeight;
  saveChat();
}

// Detect a plain-language video request and return the video prompt,
// or null when the message is not a video request. Mirrors the image
// detector: "make a video of a red bicycle" just works, no prefix.
// Questions about videos ("what is a video", "describe this video")
// and "video call" still go to chat.
function videoPromptFromText(msg){
  var m=(msg||'').trim();
  if(!m) return null;
  var low=m.toLowerCase();
  var pre=low.match(/^(video|vid|clip|movie)\s*:\s*/);
  if(pre){
    return m.slice(pre[0].length).trim();
  }
  var patterns=[
    /^(?:please\s+)?(?:can you\s+|could you\s+)?(?:make|generate|create|produce|render|film|shoot)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:short\s+)?(?:video|clip|movie|animation|short)(?:\s+of)?\s*/i,
    /^(?:please\s+)?(?:show|give)\s+me\s+(?:a\s+|an\s+|the\s+)?(?:short\s+)?(?:video|clip|movie)(?:\s+of)?\s*/i,
    /^(?:i\s+(?:want|need)|i'?d\s+like)\s+(?:a\s+|an\s+|the\s+)?(?:short\s+)?(?:video|clip|movie)(?:\s+of)?\s*/i,
    /^(?:a\s+|an\s+|the\s+)?(?:short\s+)?(?:video|clip|movie)\s+of\s+/i,
    /^(?:animate)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?/i
  ];
  for(var i=0;i<patterns.length;i++){
    var mm=m.match(patterns[i]);
    if(mm){
      var rest=m.slice(mm[0].length).trim().replace(/^of\s+/i,'');
      if(!rest) return null;
      if(/^(call|chat|meeting)\b/i.test(rest)) return null;
      return rest;
    }
  }
  return null;
}

async function generateVideoFromInput(promptArg, skipUser){
  let prompt=(promptArg!==undefined && promptArg!==null) ? String(promptArg).trim() : q.value.trim();
  if(prompt.toLowerCase().indexOf('video:')===0) prompt=prompt.slice(6).trim();
  if(!prompt){ showToast('Type a video description first'); q.focus(); return; }
  if(promptArg===undefined || promptArg===null){ q.value=''; }
  if(!current) current=newChatObj();
  if(!skipUser){
    if(!current.messages.length){current.title=prompt.slice(0,40);}
    current.messages.push({role:'user', content:prompt});
    appendMessage('user', prompt, true, null, null, null, current.messages.length-1);
    log.scrollTop=log.scrollHeight;
  }
  const aiBody=appendMessage('assistant', 'Generating video...', false);
  const t0=Date.now();
  const timer=setInterval(function(){
    const s=Math.floor((Date.now()-t0)/1000);
    aiBody.textContent='Generating video... '+Math.floor(s/60)+':'+String(s%60).padStart(2,'0')+' \u2014 clips take about a minute.';
  },1000);
  let j=null;
  try{
    const res=await fetch('/api/video',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:prompt, small:dataSaver})});
    if(!res.ok){ const t=await res.text(); throw new Error(t||('server error '+res.status)); }
    j=await res.json();
  }catch(err){
    clearInterval(timer);
    const msg='Video generation failed: '+err.message;
    aiBody.textContent=msg;
    current.messages.push({role:'assistant', content:msg});
    saveChat(); return;
  }
  clearInterval(timer);
  aiBody.innerHTML='';
  const cap=document.createElement('div'); cap.textContent='Video: '+prompt; cap.style.cssText='margin-bottom:8px;color:var(--muted);font-size:13px;';
  aiBody.appendChild(cap);
  function showVideo(){
    const vid=document.createElement('video'); vid.src=j.url; vid.controls=true; vid.autoplay=true; vid.loop=true; vid.muted=true; vid.playsInline=true;
    vid.style.cssText='max-width:100%;border-radius:12px;display:block;background:#000;';
    const dl=document.createElement('a'); dl.href=j.url; dl.download=''; dl.textContent='Download video'; dl.style.cssText='color:var(--muted);font-size:13px;display:inline-block;margin-top:8px;';
    aiBody.appendChild(vid); aiBody.appendChild(dl);
  }
  if(dataSaver){
    // Data Saver: the clip is not downloaded until the user taps.
    aiBody.appendChild(tapToLoadBox('Video: '+prompt, function(box){ box.remove(); showVideo(); }));
  }else{
    showVideo();
  }
  current.messages.push({role:'assistant', content:'Video: '+prompt, video:j.url});
  log.scrollTop=log.scrollHeight;
  saveChat();
}

function fmtSize(n){
  if(n<1024) return n+' B';
  if(n<1024*1024) return (n/1024).toFixed(1)+' KB';
  return (n/1024/1024).toFixed(1)+' MB';
}
let termBody=null;
function termBubble(){
  const d=document.createElement('div'); d.className='msg term';
  const b=document.createElement('div'); b.className='body';
  d.appendChild(b); log.appendChild(d);
  termBody=b; log.scrollTop=log.scrollHeight;
  return b;
}
function termLine(text, cls){
  if(!termBody) termBubble();
  const d=document.createElement('div');
  if(cls) d.className=cls;
  d.textContent=text;
  termBody.appendChild(d);
  while(termBody.children.length>200){ termBody.removeChild(termBody.firstChild); }
  log.scrollTop=log.scrollHeight;
}
function termDiff(diffText){
  if(!termBody) termBubble();
  const lines=diffText.split('\n');
  for(const ln of lines){
    if(!ln) continue;
    let cls='ctx', txt=ln;
    if(ln.startsWith('+ ')){ cls='add'; txt=ln.slice(2); }
    else if(ln.startsWith('- ')){ cls='del'; txt=ln.slice(2); }
    else if(ln.startsWith('  ')){ cls='ctx'; txt=ln.slice(2); }
    const d=document.createElement('div');
    d.className=cls; d.textContent=txt;
    termBody.appendChild(d);
  }
  while(termBody.children.length>200){ termBody.removeChild(termBody.firstChild); }
  log.scrollTop=log.scrollHeight;
}
// Agent step timeline (Odysseus-style): one row per tool call, with a
// status mark, the tool name, done/failed, and an expandable detail body.
let agentStepsEl=null, agentStepsDone=0, curStep=null, agentStepLog=[];
const AGENT_MAX_STEPS=10;
function stepIcon(tool){
  if(tool==='run_shell') return '<span class="astep-tico">&gt;_</span>';
  if(tool==='write_file'||tool==='edit_file'||tool==='create_pdf') return '<span class="astep-tico">&#9998;</span>';
  return '<span class="astep-tico">&#9679;</span>';
}
function agentProgress(){
  if(!agentStepsEl) return;
  const bar=agentStepsEl.querySelector('.aprog .fill');
  const lbl=agentStepsEl.querySelector('.aprog .lbl');
  if(bar) bar.style.width=Math.min(100, agentStepsDone/AGENT_MAX_STEPS*100)+'%';
  if(lbl) lbl.textContent='Agent \u00b7 step '+agentStepsDone+'/'+AGENT_MAX_STEPS;
}
function stepStart(tool, args){
  agentStepLog.push({tool:tool, args:args||'', status:'running'});
  if(!agentStepsEl){
    agentStepsEl=document.createElement('div');
    agentStepsEl.className='asteps';
    agentStepsEl.innerHTML='<div class="aprog"><span class="lbl">Agent \u00b7 step 0/'+AGENT_MAX_STEPS+'</span><span class="bar"><span class="fill"></span></span></div>';
    log.appendChild(agentStepsEl);
  }
  const node=document.createElement('div');
  node.className='astep';
  const head=document.createElement('div');
  head.className='astep-head';
  head.innerHTML='<span class="astep-dot"></span><span class="astep-ico">&#9679;</span>'+stepIcon(tool)+'<span class="astep-tool">'+esc(String(tool).toUpperCase())+'</span><span class="astep-status">running</span><span class="astep-chev">&#8250;</span>';
  head.onclick=function(){ node.classList.toggle('open'); };
  const body=document.createElement('div');
  body.className='astep-body';
  if(args){ const c=document.createElement('pre'); c.className='astep-cmd'; c.textContent=args; body.appendChild(c); }
  node.appendChild(head); node.appendChild(body);
  agentStepsEl.appendChild(node);
  agentStepsDone++;
  agentProgress();
  log.scrollTop=log.scrollHeight;
  curStep=node;
  return node;
}
function stepDiffBox(body, diffText, file){
  const lines=diffText.split('\n');
  let added=0, removed=0, ctx=0;
  for(const ln of lines){
    if(ln.startsWith('+ ')) added++;
    else if(ln.startsWith('- ')) removed++;
    else if(ln.startsWith('  ')) ctx++;
  }
  const isNew=added>0 && removed===0 && ctx===0;
  const det=document.createElement('details'); det.className='astep-out astep-diff';
  const sum=document.createElement('summary');
  const f=document.createElement('span'); f.className='diff-file'; f.textContent=file; sum.appendChild(f);
  const st=document.createElement('span'); st.className='diff-stats';
  if(isNew){ const n=document.createElement('span'); n.className='diff-stat-new'; n.textContent='new'; st.appendChild(n); }
  if(added){ const a=document.createElement('span'); a.className='diff-stat-add'; a.textContent='+'+added; st.appendChild(a); }
  if(removed){ const d=document.createElement('span'); d.className='diff-stat-del'; d.textContent='\u2212'+removed; st.appendChild(d); }
  sum.appendChild(st);
  det.appendChild(sum);
  const pre=document.createElement('pre'); pre.className='diff-pre';
  for(const ln of lines){
    if(!ln) continue;
    let cls='diff-ctx', txt=ln;
    if(ln.startsWith('+ ')){ cls='diff-add'; txt=ln.slice(2); }
    else if(ln.startsWith('- ')){ cls='diff-del'; txt=ln.slice(2); }
    else if(ln.startsWith('  ')){ cls='diff-ctx'; txt=ln.slice(2); }
    else if(ln==='...'){ cls='diff-meta'; }
    const sp=document.createElement('span'); sp.className=cls; sp.textContent=txt||'\u00a0'; pre.appendChild(sp);
  }
  det.appendChild(pre);
  body.appendChild(det);
}
function stepFinish(ev){
  const node=curStep; if(!node) return;
  const ok=!(ev.tool==='run_shell' && ev.exit_code!==0 && ev.exit_code!==undefined && ev.exit_code!==null);
  const lastLog=agentStepLog[agentStepLog.length-1]; if(lastLog) lastLog.status=ok?'done':'failed';
  const body=node.querySelector('.astep-body');
  if(ev.diff){ const cb=body.querySelector('.astep-cmd'); if(cb) cb.remove(); }
  if(ev.output && ev.output.trim()){
    const det=document.createElement('details'); det.className='astep-out';
    const sum=document.createElement('summary'); sum.textContent='Output'; det.appendChild(sum);
    const pre=document.createElement('pre'); pre.textContent=ev.output; det.appendChild(pre);
    body.appendChild(det);
  }
  if(ev.diff){
    const file=(lastLog && lastLog.args)? (lastLog.args.split('\n')[0].split('/').pop() || 'diff') : 'diff';
    stepDiffBox(body, ev.diff, file);
  }
  if(!ok){ const e=document.createElement('div'); e.className='err'; e.textContent='\u2014 exit '+ev.exit_code+' \u2014'; body.appendChild(e); }
  const ico=node.querySelector('.astep-ico');
  const st=node.querySelector('.astep-status');
  if(ok){ node.classList.add('done'); if(ico) ico.textContent='\u2713'; if(st) st.textContent='done'; }
  else { node.classList.add('err'); if(ico) ico.textContent='\u2717'; if(st) st.textContent='failed'; }
  log.scrollTop=log.scrollHeight;
}
async function refreshWorkspace(){
  try{
    const rs=await fetch('/api/settings');
    const sj=await rs.json();
    if(wsPathLabel) wsPathLabel.textContent=sj.workspace||'';
  }catch(e){}
  try{
    const r=await fetch('/api/workspace');
    const items=await r.json();
    wsList.innerHTML='';
    if(!items.length){ const p=document.createElement('div'); p.className='note'; p.textContent='Workspace is empty.'; wsList.appendChild(p); return; }
    for(const it of items){
      const row=document.createElement('div'); row.className='wsrow';
      const nm=document.createElement('span'); nm.className='nm'; nm.textContent=it.path+(it.is_dir?' /':'');
      const sz=document.createElement('span'); sz.className='sz'; sz.textContent=it.is_dir?'':fmtSize(it.size);
      row.appendChild(nm); row.appendChild(sz); wsList.appendChild(row);
    }
  }catch(e){}
}
async function stopAgent(){
  if(!currentRunId) return;
  try{ await fetch('/api/agent/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id:currentRunId})}); }catch(e){}
  if(agentReader){ try{ await agentReader.cancel(); }catch(e){} }
  agentRunning=false;
  stopAgentBtn.style.display='none';
  updateResendButtons();
  termLine('— stopped by user —', 'note');
}
// Trigger phrases for auto-activating a skill. The document skills are
// listed explicitly (their trigger words are distinctive); every other
// skill is matched by its id/name or by a file extension in its text.
const AUTO_TRIGGERS={
  docx:["word document","word doc","word letter","word report","word memo","word template","word file","microsoft word",".docx",".dotx","letterhead"],
  pdf:[".pdf","pdf form"],
  pptx:["presentation","slide deck","slides","pitch deck",".pptx",".potx","powerpoint","slide"],
  xlsx:["spreadsheet",".xlsx",".xlsm",".csv",".tsv","excel","workbook"]
};
// Auto-activate a skill when the user's message clearly needs one.
// Only used when no skill is manually selected (manual selection always
// wins). Per-request only: it does not change the chat's selected skill,
// so each message is judged on its own. Returns a skill id, or null.
function autoSkillForMessage(text){
  if(!text || !skills || !skills.length) return null;
  const norm=' '+String(text).toLowerCase().replace(/[^a-z0-9.]+/g,' ').replace(/\s+/g,' ').trim()+' ';
  let best=null, bestScore=0;
  for(const s of skills){
    let score=0;
    const id=String(s.id||'').toLowerCase();
    const name=String(s.name||'').toLowerCase();
    const instr=String(s.instruction||'').toLowerCase();
    if(id && (norm.indexOf(' '+id+' ')>=0 || norm.indexOf(' '+id+'s ')>=0)) score+=5;
    else if(name && name!==id && (norm.indexOf(' '+name+' ')>=0 || norm.indexOf(' '+name+'s ')>=0)) score+=5;
    const exts=instr.slice(0,3000).match(/\.[a-z0-9]{2,5}\b/g)||[];
    const seen={};
    for(const ext of exts){
      if(!seen[ext]){ seen[ext]=1; if(norm.indexOf(ext)>=0) score+=4; }
    }
    const trig=AUTO_TRIGGERS[id]||AUTO_TRIGGERS[name]||[];
    for(const ph of trig){
      if(norm.indexOf(ph)>=0) score+=3;
    }
    if(score>bestScore){ bestScore=score; best=s; }
  }
  return bestScore>=3 ? best.id : null;
}
function lastUserText(){
  if(!current || !current.messages) return '';
  for(let i=current.messages.length-1;i>=0;i--){
    if(current.messages[i].role==='user') return current.messages[i].content||'';
  }
  return '';
}
// The skill for this request: the manually selected one, or the
// auto-detected one. Shows a toast when a skill auto-activates.
function effectiveSkillId(){
  if(current && current.skill_id) return current.skill_id;
  const auto=autoSkillForMessage(lastUserText());
  if(auto){
    const s=skills.find(function(x){ return x.id===auto; });
    if(s) showToast('Auto-activated skill: '+s.name);
  }
  return auto;
}

function agentSummaryText(){
  if(!agentStepLog.length) return '';
  const failN=agentStepLog.filter(function(s){ return s.status!=='done'; }).length;
  const lines=['Summary - '+agentStepLog.length+' step'+(agentStepLog.length===1?'':'s')+(failN?(', '+failN+' failed'):'')];
  agentStepLog.forEach(function(s){
    let a=(s.args||'').split('\n')[0];
    if(a.length>90) a=a.slice(0,90)+'…';
    let what='';
    if(s.tool==='write_file') what='Wrote '+a;
    else if(s.tool==='read_file') what='Read '+a;
    else if(s.tool==='run_shell') what='Ran: '+a;
    else if(s.tool==='list_files') what='Listed '+(a||'workspace');
    else if(s.tool==='create_pdf') what='Made PDF '+a;
    else if(s.tool==='search_images') what='Searched images: '+a;
    else what=s.tool+' '+a;
    if(s.status==='failed') what+=' - failed';
    lines.push('- '+what);
  });
  return lines.join('\n');
}
async function runAgent(){
  const runId=newId();
  currentRunId=runId; agentRunning=true;
  stopAgentBtn.style.display='inline-block';
  updateResendButtons();
  const aiBody=appendMessage('assistant', '', false);
  termBody=null;
  agentStepsEl=null; agentStepsDone=0; curStep=null; agentStepLog=[];
  let aiText=''; let done=false;
  const sendMessages=current.messages.map(function(m, idx){
    const copy={role:m.role, content:m.content};
    if(m.pdfText && idx===current.messages.length-1){ copy.content+='\n\n[Attached PDF: '+(m.pdfName||'document')+']\n'+m.pdfText; }
    return copy;
  });
  try{
    const res=await fetch('/api/agent',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:current.model||currentModel, messages:sendMessages, endpoint_id: activeEndpointId||null, skill_id: effectiveSkillId()||null, run_id: runId, persona: currentPersona})});
    if(!res.ok){ throw new Error('server error '+res.status); }
    agentReader=res.body.getReader();
    const dec=new TextDecoder();
    let buf='';
    while(true){
      const r2=await agentReader.read(); if(r2.done) break;
      buf+=dec.decode(r2.value,{stream:true});
      let idx;
      while((idx=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,idx); buf=buf.slice(idx+1);
        if(!line.startsWith('data:')) continue;
        let d=line.slice(5); if(d.startsWith(' ')) d=d.slice(1);
        if(!d || d==='[DONE]') continue;
        let ev; try{ ev=JSON.parse(d); }catch(e){ continue; }
        if(ev.type==='tool'){
          let a=ev.args||'';
          if(ev.tool==='write_file'||ev.tool==='create_pdf'){ a=a.split('\n')[0]; }
          stepStart(ev.tool, a);
        } else if(ev.type==='result'){
          stepFinish(ev);
          refreshWorkspace();
        } else if(ev.type==='text'){
          aiText+=ev.content||'';
          aiBody.innerHTML=md(aiText);
          aiBody.dataset.fullText=aiText;
          log.scrollTop=log.scrollHeight;
        } else if(ev.type==='done'){
          done=true;
          agentStepsDone=ev.steps||agentStepsDone;
          agentProgress();
          if(agentStepsEl){ const lbl=agentStepsEl.querySelector('.aprog .lbl'); if(lbl) lbl.textContent='Agent \u00b7 done ('+agentStepsDone+' step'+(agentStepsDone===1?'':'s')+')'; }
          const sum=agentSummaryText();
          if(sum){ aiText+=(aiText?'\n\n':'')+sum; aiBody.innerHTML=md(aiText); aiBody.dataset.fullText=aiText; log.scrollTop=log.scrollHeight; }
        } else if(ev.type==='error'){
          aiText+=(aiText?'\n\n':'')+'Error: '+ev.message;
          aiBody.innerHTML=md(aiText); aiBody.dataset.fullText=aiText;
          if(curStep){ curStep.classList.add('err'); const ico=curStep.querySelector('.astep-ico'); const st=curStep.querySelector('.astep-status'); if(ico) ico.textContent='\u2717'; if(st) st.textContent='failed'; }
        }
      }
    }
  }catch(err){
    if(agentRunning){
      aiText+=(aiText?'\n\n':'')+'(Agent stopped: '+err.message+')';
      aiBody.innerHTML=md(aiText); aiBody.dataset.fullText=aiText;
    }
  }finally{
    agentRunning=false; currentRunId=null; agentReader=null;
    stopAgentBtn.style.display='none';
    updateResendButtons();
    refreshWorkspace();
  }
  current.messages.push({role:'assistant', content:aiText});
  log.scrollTop=log.scrollHeight;
  saveChat();
}

async function runPuterChat(aiBody){
  // Puter (puter.js) runs in the browser: the user signs in to Puter (free
  // account) and Puter covers the usage. The Rust server is not involved and
  // never sees a Puter token. Agent mode stays on the selected endpoint /
  // local model because Puter offers no supported server-side API for the
  // Rust server to call.
  let aiText='';
  function puterFail(msg){
    aiText=msg; aiBody.textContent=aiText; aiBody.dataset.fullText=aiText;
    current.messages.push({role:'assistant', content:aiText});
    log.scrollTop=log.scrollHeight; saveChat();
  }
  if(!puterReady()){
    puterFail('Puter is unavailable — the Puter script could not load. Check your connection and reload the page, or pick a local or endpoint model.');
    return;
  }
  try{
    if(puter.auth && typeof puter.auth.isSignedIn==='function'){
      let signed=false;
      try{ signed=puter.auth.isSignedIn(); }catch(e){ signed=false; }
      if(!signed){
        aiBody.textContent='Signing in to Puter… (free account)';
        try{ await puter.auth.signIn(); }catch(se){ puterFail('Puter sign-in was cancelled. Sign in to Puter with a free account and send again.'); return; }
        try{ signed=puter.auth.isSignedIn(); }catch(e){ signed=false; }
        if(!signed){ puterFail('Puter sign-in did not complete. Sign in to Puter with a free account and send again.'); return; }
      }
    }
  }catch(e){ /* fall through and try the chat call */ }
  try{
    const cleanMsgs=current.messages.map(function(m){ return {role:m.role, content:m.content}; });
    const resp=await puter.ai.chat(cleanMsgs, {model: activePuterModel, stream: true});
    for await (const part of resp){
      if(part && part.text){ aiText+=part.text; aiBody.textContent=aiText; log.scrollTop=log.scrollHeight; }
    }
    if(!aiText){ aiText='(Puter returned an empty reply. Try again.)'; aiBody.textContent=aiText; }
    else{ aiBody.innerHTML=md(aiText); }
  }catch(err){
    const m='(Puter error: '+(err && err.message ? err.message : err)+'. If this is a sign-in problem, sign in to Puter and try again.)';
    aiText=m; aiBody.textContent=aiText;
  }
  aiBody.dataset.fullText=aiText;
  current.messages.push({role:'assistant', content:aiText});
  log.scrollTop=log.scrollHeight;
  saveChat();
}

// Resend: re-send an earlier user message exactly as if the user had
// typed it again, with the currently selected endpoint/model. The old
// assistant reply directly after that message is removed (display and
// saved history) and replaced by the fresh reply.
function updateResendButtons(){
  const busy=chatBusy||agentRunning;
  document.querySelectorAll('.resendbtn').forEach(function(btn){ btn.disabled=busy; });
}
async function resendMessage(idx){
  if(chatBusy || agentRunning){ showToast('Wait for the current reply to finish'); return; }
  if(!current || !current.messages[idx] || current.messages[idx].role!=='user') return;
  const msg=current.messages[idx].content;
  let end=idx+1;
  while(end<current.messages.length && current.messages[end].role==='assistant'){ end++; }
  current.messages.splice(idx+1, end-(idx+1));
  renderMessages();
  log.scrollTop=log.scrollHeight;
  const vidPrompt=videoPromptFromText(msg);
  if(vidPrompt!==null){ await generateVideoFromInput(vidPrompt, true); return; }
  const imgPrompt=imagePromptFromText(msg);
  if(imgPrompt!==null){ await generateImageFromInput(imgPrompt, true); return; }
  const pdfReq=pdfRequestFromText(msg);
  if(pdfReq!==null){ await generatePdfFromInput(pdfReq, true); return; }
  if(mode==='agent'){ await runAgent(); return; }
  await runPlainChat(idx);
}
document.addEventListener('click',function(e){
  const b=e.target && e.target.closest ? e.target.closest('.resendbtn') : null;
  if(!b || b.disabled) return;
  const msgEl=b.closest('.msg');
  const idx=msgEl && msgEl.dataset.msgIndex!==undefined ? parseInt(msgEl.dataset.msgIndex,10) : -1;
  if(idx<0 || isNaN(idx)) return;
  resendMessage(idx);
});

// Plain-chat reply (streamed /api/chat). Extracted from the submit
// handler so Resend can re-run the same flow for an earlier message.
// resendIdx: index of the user message being (re)sent, so its attached
// images are included even when it is not the last message.
async function runPlainChat(resendIdx){
  const aiBody=appendMessage('assistant', '', false);
  let aiText=''; let errMsg='';
  if(activePuterModel){ await runPuterChat(aiBody); return; }
  chatBusy=true; updateResendButtons();
  // Only the resent (or latest) message keeps its images, so older
  // images are not re-sent on every turn.
  const sendIdx=(resendIdx===undefined || resendIdx===null) ? current.messages.length-1 : resendIdx;
  const sendMessages=current.messages.map(function(m, idx){
    const copy={role:m.role, content:m.content};
    if(m.images && idx===sendIdx){ copy.images=m.images; }
    if(m.pdfText && idx===sendIdx){ copy.content+='\n\n[Attached PDF: '+(m.pdfName||'document')+']\n'+m.pdfText; }
    return copy;
  });
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:current.model||currentModel, messages:sendMessages, endpoint_id: activeEndpointId||null, skill_id: effectiveSkillId()||null, persona: currentPersona})});
    if(!res.ok){errMsg='(error '+res.status+')';}
    else{
      const reader=res.body.getReader(); const dec=new TextDecoder();
      while(true){const r2=await reader.read(); if(r2.done) break;
        const chunk=dec.decode(r2.value,{stream:true});
        for(const line of chunk.split('\n')){if(!line.startsWith('data:')) continue; let d=line.slice(5); if(d.startsWith(' ')) d=d.slice(1); if(d==='[DONE]'||d==='') continue; aiText+=d; aiBody.textContent=stripWebImageMarkerLive(stripMarkerLive(aiText));}
        log.scrollTop=log.scrollHeight;
      }
    }
  }catch(err){errMsg='(error: '+err.message+')';}
  chatBusy=false; updateResendButtons();
  if(errMsg){aiText=errMsg; aiBody.textContent=aiText;}
  else{
    // Safety net: strip a stray [[IMAGE: ...]] marker if the model
    // emitted one on its own. No image is generated in normal chat
    // (auto-illustration removed 2026-11-21 at Steven's word).
    const ex=extractImageMarker(aiText);
    // Real web photos: a [[WEBIMAGE: ...]] marker means the reply
    // should show real photos from the web (never AI-generated).
    const wx=extractWebImageMarker(ex.text);
    aiText=wx.text;
    aiBody.innerHTML=md(aiText);
    aiBody.dataset.fullText=aiText;
    const textMsg={role:'assistant', content:aiText};
    current.messages.push(textMsg);
    log.scrollTop=log.scrollHeight;
    saveChat();
    if(wx.query){ attachWebImages(wx.query); }
    return;
  }
  aiBody.dataset.fullText=aiText;
  current.messages.push({role:'assistant', content:aiText});
  log.scrollTop=log.scrollHeight;
  saveChat();
}

f.addEventListener('submit', async function(e){
  e.preventDefault();
  const msg=q.value.trim(); if(!msg && !pendingImage && !pendingPdf) return;
  const vidPrompt=videoPromptFromText(msg);
  if(vidPrompt!==null){ generateVideoFromInput(vidPrompt); return; }
  const imgPrompt=imagePromptFromText(msg);
  if(imgPrompt!==null){ generateImageFromInput(imgPrompt); return; }
  const pdfReq=pdfRequestFromText(msg);
  if(pdfReq!==null){ generatePdfFromInput(pdfReq); return; }
  q.value='';
  if(!current) current=newChatObj();
  if(!current.messages.length){current.title=(msg||(pendingPdf?pendingPdf.name:'Image')).slice(0,40);}
  const userMsg={role:'user', content:msg||(pendingPdf?'Please read this PDF.':'Describe this image.')};
  let attachedImages=null;
  if(pendingImage){
    attachedImages=[pendingImage.b64];
    userMsg.images=attachedImages;
  }
  if(pendingPdf){
    userMsg.pdfName=pendingPdf.name;
    let pt=pendingPdf.text||'';
    if(!pt) pt='[No readable text found in this PDF - it may be a scanned image.]';
    if(pt.length>12000) pt=pt.slice(0,12000)+'\n[Text truncated - showing the first 12,000 characters.]';
    userMsg.pdfText=pt;
    userMsg.content+='\n[PDF attached: '+pendingPdf.name+']';
    pendingPdf=null; attachPreview.style.display='none';
  }
  current.messages.push(userMsg);
  appendMessage('user', userMsg.content, true, null, attachedImages, null, current.messages.length-1);
  log.scrollTop=log.scrollHeight;
  if(pendingImage){
    const eff=effectiveModel();
    let note='';
    if(!eff.vision){ note='This model can\u2019t see images. Switch to a Gemini model (in the model picker) to describe a photo.'; }
    pendingImage=null; attachPreview.style.display='none';
    if(note){
      current.messages.push({role:'assistant', content:note});
      appendMessage('assistant', note, true);
      log.scrollTop=log.scrollHeight;
      saveChat();
      return;
    }
  }
  if(mode==='agent'){ await runAgent(); return; }
  await runPlainChat();
});

(async function(){
  await loadPresets();
  await loadEndpoints();
  await loadMemories();
  await loadSkills();
  await loadChats();
  if(chats.length){await openChat(chats[0].id);}
  else{current=newChatObj(); renderMessages(); renderChatList();}
  updateSkillChip();
})();
</script>
</body>
</html>
"##;
