#!/usr/bin/env python3
"""Add Puter (puter.js) free higher-end models to Athena AI (chat mode, browser-side).
Run on the phone inside /root/odysseus-rust. Backs up src/main.rs first is done by caller.
Each replacement must match exactly once, else the script aborts loudly.
"""
import sys

PATH = "/root/odysseus-rust/src/main.rs"

def rep(s, old, new, name):
    n = s.count(old)
    if n != 1:
        print(f"FAIL {name}: expected 1 occurrence, found {n}")
        sys.exit(1)
    return s.replace(old, new)

s = open(PATH, encoding="utf-8").read()
orig_len = len(s)

# 1. Load puter.js in the page head.
s = rep(s,
'<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Athena AI</title>',
'<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Athena AI</title>\n<script src="https://js.puter.com/v2/"></script>',
"head-puter-script")

# 2. Declare activePuterModel + PUTER_MODELS + helpers next to activeEndpointId.
s = rep(s,
"""let activeEndpointId='';
try{ activeEndpointId=localStorage.getItem('athena_active_endpoint')||''; }catch(e){}""",
"""let activeEndpointId='';
try{ activeEndpointId=localStorage.getItem('athena_active_endpoint')||''; }catch(e){}
let activePuterModel='';
try{ activePuterModel=localStorage.getItem('athena_active_puter')||''; }catch(e){}
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
    if(activePuterModel) localStorage.setItem('athena_active_puter', activePuterModel);
    else localStorage.removeItem('athena_active_puter');
  }catch(e){}
  refreshModelSelect();
}""",
"puter-state")

# 3a. Change handler: handle 'puter:' picks first.
s = rep(s,
"""modelSel.addEventListener('change', async function(){
  const v=modelSel.value;
  if(v.indexOf('endpoint:')===0){""",
"""modelSel.addEventListener('change', async function(){
  const v=modelSel.value;
  if(v.indexOf('puter:')===0){
    const mid=v.slice('puter:'.length);
    setActivePuter(mid);
    if(activeEndpointId) setActiveEndpoint('');
    currentModel=mid;
    if(current){current.model=currentModel; saveChat();}
    return;
  }
  if(v.indexOf('endpoint:')===0){""",
"picker-change-puter")

# 3b. Endpoint branch: clear Puter when an endpoint is picked.
s = rep(s,
"""      setActiveEndpoint(id);
      currentModel=model;""",
"""      setActivePuter('');
      setActiveEndpoint(id);
      currentModel=model;""",
"picker-change-endpoint-clears-puter")

# 3c. Local-model fallthrough: clear Puter too.
s = rep(s,
"""  currentModel=v;
  if(activeEndpointId) setActiveEndpoint('');
  if(current){current.model=currentModel; saveChat();}
});""",
"""  currentModel=v;
  if(activeEndpointId) setActiveEndpoint('');
  setActivePuter('');
  if(current){current.model=currentModel; saveChat();}
});""",
"picker-change-local-clears-puter")

# 4. refreshModelSelect: add the Puter optgroup and restore a Puter selection.
s = rep(s,
"""    modelSel.appendChild(g);
  }
  if(activeEndpointId){
    const ep=endpoints.find(function(e){return e.id===activeEndpointId;});""",
"""    modelSel.appendChild(g);
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
    try{ localStorage.removeItem('athena_active_puter'); }catch(e){}
  }
  if(activeEndpointId){
    const ep=endpoints.find(function(e){return e.id===activeEndpointId;});""",
"picker-puter-group")

# 5. effectiveModel: report Puter (non-vision, so image attach shows the switch note).
s = rep(s,
"""function effectiveModel(){
  if(activeEndpointId){""",
"""function effectiveModel(){
  if(activePuterModel){
    return {name: activePuterModel, vision: false, via: 'puter'};
  }
  if(activeEndpointId){""",
"effective-model-puter")

# 6. openChat: restore a Puter model when reopening a chat that used one.
s = rep(s,
"""  current=await r.json();
  currentModel=current.model||DEFAULT_MODEL;
  if(activeEndpointId){""",
"""  current=await r.json();
  currentModel=current.model||DEFAULT_MODEL;
  if(currentModel.indexOf('puter:')===0){ setActivePuter(currentModel.slice(6)); currentModel=activePuterModel; if(activeEndpointId) setActiveEndpoint(''); }
  else if(PUTER_MODELS.some(function(p){return p.id===currentModel;})){ setActivePuter(currentModel); if(activeEndpointId) setActiveEndpoint(''); }
  else if(activeEndpointId){""",
"openchat-puter-restore")

# 7. New chat: normalise a 'puter:' picker value to the bare model id.
s = rep(s,
"""  current=newChatObj();
  currentModel=modelSel.value||DEFAULT_MODEL; current.model=currentModel;""",
"""  current=newChatObj();
  currentModel=modelSel.value||DEFAULT_MODEL; current.model=currentModel;
  if(currentModel.indexOf('puter:')===0){ setActivePuter(currentModel.slice(6)); currentModel=activePuterModel; current.model=currentModel; }""",
"newchat-puter-normalise")

# 8. Chat send: branch to the browser-side Puter call when a Puter model is active.
s = rep(s,
"""  if(mode==='agent'){ await runAgent(); return; }
  const aiBody=appendMessage('assistant', '', false);
  let aiText=''; let errMsg='';""",
"""  if(mode==='agent'){ await runAgent(); return; }
  const aiBody=appendMessage('assistant', '', false);
  let aiText=''; let errMsg='';
  if(activePuterModel){ await runPuterChat(aiBody); return; }""",
"send-branch-puter")

# 9. runPuterChat: browser-side Puter chat (streaming), rendered + saved like a normal reply.
s = rep(s,
"""f.addEventListener('submit', async function(e){""",
"""async function runPuterChat(aiBody){
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

f.addEventListener('submit', async function(e){""",
"run-puter-chat")

open(PATH, "w", encoding="utf-8").write(s)
print(f"OK patched {PATH}: {orig_len} -> {len(s)} bytes, 9 replacements applied")
