import itertools
import os
import threading
import time
from pathlib import Path
import streamlit as st
import torch
import pandas as pd

from src.model.rtu import Model
from src.model.droid import DroidEngine
from src.model.droid_manager import DroidManager
from src.model.teacher import OpenAITeacher

# ponytail: Unified Streamlit WebUI supporting Droid Lifelong Engine, Raw RTU, and LLM Provider.

st.set_page_config(page_title="TMT Droid Studio", page_icon="🤖", layout="wide")

# Initialize global managers
if "droid_mgr" not in st.session_state:
    st.session_state.droid_mgr = DroidManager(base_dir="droids")

droid_mgr: DroidManager = st.session_state.droid_mgr

# Ensure default droid exists
available_droids = droid_mgr.list_droids()
if not available_droids:
    droid_mgr.create_droid("droid-alpha")
    available_droids = droid_mgr.list_droids()

if "selected_droid_name" not in st.session_state:
    st.session_state.selected_droid_name = available_droids[0]

# Active droid
active_droid: DroidEngine = droid_mgr.get_droid(st.session_state.selected_droid_name)

# LLM Provider configuration state
if "llm_enabled" not in st.session_state:
    st.session_state.llm_enabled = False
if "llm_base_url" not in st.session_state:
    st.session_state.llm_base_url = "http://localhost:11434/v1"
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = "EMPTY"
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "llama3:latest"
if "llm_preset" not in st.session_state:
    st.session_state.llm_preset = "Ollama (localhost:11434)"

# Session state for chat & training
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "model_lock" not in st.session_state:
    st.session_state.model_lock = threading.Lock()

if "raw_model" not in st.session_state:
    st.session_state.raw_model = Model(dim=256, layers=8, spread=32, rate=5e-4)

raw_model: Model = st.session_state.raw_model
lock: threading.Lock = st.session_state.model_lock

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🤖 Droid Studio")
    
    engine_mode = st.radio(
        "Engine Selection",
        ["Droid Lifelong Engine (Anchored)", "Raw Byte-Level RTU (Experimental)"],
        index=0
    )
    
    st.divider()

    if engine_mode.startswith("Droid"):
        st.subheader("Droid Profile")
        
        col_sel, col_new = st.columns([2, 1])
        selected_name = col_sel.selectbox(
            "Active Droid",
            available_droids,
            index=available_droids.index(st.session_state.selected_droid_name) if st.session_state.selected_droid_name in available_droids else 0
        )
        if selected_name != st.session_state.selected_droid_name:
            st.session_state.selected_droid_name = selected_name
            st.session_state.chat_history = []
            st.rerun()

        new_name = st.text_input("New Droid Name", placeholder="e.g. droid-geospatial")
        if st.button("➕ Create Droid"):
            if new_name.strip():
                clean = new_name.strip().lower().replace(" ", "-")
                droid_mgr.create_droid(clean)
                st.session_state.selected_droid_name = clean
                st.session_state.chat_history = []
                st.rerun()

        st.markdown(f"**Facts in Memory**: {len(active_droid.knowledge_bank)}")
        st.markdown(f"**Experience Steps**: {active_droid.step_count}")
        mem_norm = float(torch.norm(active_droid.memory.states).item())
        st.markdown(f"**Memory State Norm**: `{mem_norm:.2f}`")

        st.divider()
        st.subheader("Base Model & Anchor")
        st.success("🟢 ONNX MiniLM (INT8, 22MB) Active")
        st.caption("Cached in `~/.cache/huggingface` (Zero GPU RAM, sub-15ms CPU).")

        st.divider()
        st.subheader("🌐 LLM Provider (Teacher & Synthesis)")
        llm_on = st.checkbox("Enable External LLM Provider", value=st.session_state.llm_enabled)
        st.session_state.llm_enabled = llm_on

        if llm_on:
            presets = {
                "Ollama (localhost:11434)": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "llama3:latest",
                    "api_key": "EMPTY"
                },
                "LM Studio (localhost:1234)": {
                    "base_url": "http://localhost:1234/v1",
                    "model": "local-model",
                    "api_key": "EMPTY"
                },
                "Groq": {
                    "base_url": "https://api.groq.com/openai/v1",
                    "model": "llama-3.3-70b-versatile",
                    "api_key": ""
                },
                "OpenRouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "meta-llama/llama-3-8b-instruct",
                    "api_key": ""
                },
                "OpenAI": {
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "api_key": ""
                },
                "Custom": {
                    "base_url": st.session_state.llm_base_url,
                    "model": st.session_state.llm_model,
                    "api_key": st.session_state.llm_api_key
                }
            }

            preset_choice = st.selectbox(
                "Provider Preset",
                list(presets.keys()),
                index=list(presets.keys()).index(st.session_state.llm_preset) if st.session_state.llm_preset in presets else 0
            )

            if preset_choice != st.session_state.llm_preset:
                st.session_state.llm_preset = preset_choice
                if preset_choice != "Custom":
                    st.session_state.llm_base_url = presets[preset_choice]["base_url"]
                    st.session_state.llm_model = presets[preset_choice]["model"]
                    if presets[preset_choice]["api_key"]:
                        st.session_state.llm_api_key = presets[preset_choice]["api_key"]
                st.rerun()

            st.session_state.llm_base_url = st.text_input("Base URL", value=st.session_state.llm_base_url)
            st.session_state.llm_model = st.text_input("Model Name", value=st.session_state.llm_model)
            st.session_state.llm_api_key = st.text_input("API Key", value=st.session_state.llm_api_key, type="password")

            if st.button("🧪 Test LLM Connection"):
                with st.spinner("Pinging LLM provider..."):
                    t = OpenAITeacher(
                        base_url=st.session_state.llm_base_url,
                        api_key=st.session_state.llm_api_key,
                        model=st.session_state.llm_model
                    )
                    ok, msg = t.test_connection(timeout=5.0)
                    if ok:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")

        st.divider()
        col_save, col_reset = st.columns(2)
        if col_save.button("💾 Save Droid"):
            droid_mgr.save_droid(st.session_state.selected_droid_name)
            st.success(f"Saved {st.session_state.selected_droid_name}")
        if col_reset.button("🔄 Reset Memory"):
            active_droid.memory.reset()
            active_droid.knowledge_bank.clear()
            active_droid.knowledge_vectors = None
            active_droid.step_count = 0
            droid_mgr.save_droid(st.session_state.selected_droid_name)
            st.warning("Memory reset to zero.")
            st.rerun()

    else:
        st.subheader("Raw RTU Config")
        col_dim, col_layers = st.columns(2)
        dim = col_dim.number_input("Dim", min_value=32, max_value=1024, value=raw_model.dim, step=32)
        layers = col_layers.number_input("Layers", min_value=1, max_value=32, value=raw_model.layers, step=1)
        if dim != raw_model.dim or layers != raw_model.layers:
            if st.button("Reinitialize Model"):
                st.session_state.raw_model = Model(dim=int(dim), layers=int(layers), rate=5e-4)
                st.rerun()
        st.markdown(f"**Parameters**: {raw_model.count():,}")

# --- MAIN INTERFACE TABS ---
tab_chat, tab_teach, tab_logs, tab_weights = st.tabs([
    "💬 Chat & Conversational Teach",
    "📚 Domain Teaching & Ingestion",
    "📜 Transparent Audit Logs",
    "🔬 Weights & Memory Inspector"
])

# --- TAB 1: CHAT & IN-CHAT TEACHING ---
with tab_chat:
    st.subheader(f"Chat with {st.session_state.selected_droid_name if engine_mode.startswith('Droid') else 'Raw RTU'}")

    col_opt1, col_opt2 = st.columns([2, 1])
    auto_teach = col_opt1.checkbox("Auto-learn new facts directly from chat", value=True)
    use_llm_synth = False
    if st.session_state.llm_enabled:
        use_llm_synth = col_opt2.checkbox("Synthesize reply with LLM Provider", value=True)

    if engine_mode.startswith("Droid"):
        st.caption("Tip: You can talk normally, teach concepts (e.g. 'learn: Remote sensing is...'), or ask questions.")

    chat_container = st.container()

    # Render conversation above chat input
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("source_label"):
                    st.caption(f"🏷️ **Source**: {msg['source_label']}")
                if msg.get("facts_recalled"):
                    with st.expander(f"🔍 View {len(msg['facts_recalled'])} Recalled Facts from Plastic Memory"):
                        for idx, f in enumerate(msg["facts_recalled"]):
                            score = f.get("similarity", 0.0)
                            st.markdown(f"**{idx+1}.** {f['text']} *(match score: `{score:.3f}`)*")

    user_query = st.chat_input("Say something or teach your Droid...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        if engine_mode.startswith("Droid"):
            source_label = ""
            facts_recalled = []

            # If auto-teach enabled and message looks like educational text/definition
            if auto_teach and len(user_query) > 50 and not user_query.strip().endswith("?"):
                with st.spinner("Absorbing knowledge into plastic memory..."):
                    res = active_droid.teach(user_query, source="chat")
                    response = f"✅ Absorbed {res['propositions']} facts into plastic memory (state norm: {res['memory_norm']:.2f}). You can now ask me about this domain!"
                    source_label = f"🧠 Plastic Memory (Live conversational absorption of {res['propositions']} facts)"
            else:
                with st.spinner("Searching plastic memory..."):
                    hits = active_droid.recall(user_query, top_k=4, threshold=0.45)
                    # Contextual follow-up fallback: e.g. "explain complete", "what about RS"
                    if not hits and active_droid.last_query:
                        contextual_query = f"{active_droid.last_query} {user_query}"
                        hits = active_droid.recall(contextual_query, top_k=4, threshold=0.40)

                    facts_recalled = hits
                    context = " ".join([h["text"] for h in hits]) if hits else None

                    if st.session_state.llm_enabled and use_llm_synth:
                        with st.spinner(f"Synthesizing response via {st.session_state.llm_model}..."):
                            teacher = OpenAITeacher(
                                base_url=st.session_state.llm_base_url,
                                api_key=st.session_state.llm_api_key,
                                model=st.session_state.llm_model
                            )
                            response, meta = teacher.synthesize_answer(
                                user_query,
                                context=context,
                                droid_name=st.session_state.selected_droid_name
                            )
                            if meta.get("source") == "llm_grounded_memory":
                                max_score = hits[0]["similarity"] if hits else 0.0
                                source_label = f"🌐 LLM ({st.session_state.llm_model}) + 🧠 Grounded in {len(hits)} Droid Facts (top match: {max_score:.2f})"
                            elif meta.get("source") == "llm_general_knowledge":
                                source_label = f"🌐 LLM ({st.session_state.llm_model}) [General Knowledge — No Droid facts matched]"
                            else:
                                source_label = "🧠 Plastic Memory (Direct recall — LLM call failed)"
                    else:
                        response = active_droid.chat(user_query)
                        if hits:
                            source_label = f"🧠 Plastic Droid Memory ({len(hits)} facts recalled, top match: {hits[0]['similarity']:.2f})"
                        else:
                            source_label = "🧠 Plastic Droid Memory [No relevant facts found in memory]"

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "source_label": source_label,
                "facts_recalled": facts_recalled
            })

        else:
            # Raw byte RTU inference
            data = (user_query + "\n").encode("utf-8")
            out_bytes = []
            with lock:
                for i, (c, n) in enumerate(itertools.pairwise(data)):
                    raw_model(c, nextb=n, end=(i == len(data) - 2), frozen=True)
                b = data[-1]
                for _ in range(300):
                    b, stop = raw_model(b, nextb=None, end=False, frozen=True)
                    out_bytes.append(b)
                    if stop > 0.35:
                        break
            final_text = bytes(out_bytes).decode("utf-8", errors="replace")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_text,
                "source_label": f"⚙️ Raw Byte RTU Model (dim={raw_model.dim}, layers={raw_model.layers})"
            })

        st.rerun()

# --- TAB 2: DOMAIN TEACHING & INGESTION ---
with tab_teach:
    st.subheader("Teach Specific Domain Knowledge")
    st.markdown("Paste any domain text, article, or definition. Parameters and learning rate are **automatically selected** to guarantee zero collapse.")

    uploaded_file = st.file_uploader("Upload Markdown Document (.md)", type=["md"])
    file_content = uploaded_file.read().decode("utf-8", errors="ignore") if uploaded_file else ""

    sample_remote_sensing = (
        "Remote sensing is the acquisition of information about an object or phenomenon "
        "without making physical contact with the object, in contrast to on-site observation. "
        "The term is applied especially to acquiring information about Earth and other planets. "
        "Remote sensing is used in numerous fields, including geophysics, geography, land surveying "
        "and most Earth science disciplines (e.g. exploration geophysics, hydrology, ecology, meteorology, "
        "oceanography, glaciology, geology). It also has military, intelligence, commercial, economic, "
        "planning, and humanitarian applications."
    )

    teach_text = st.text_area(
        "Domain Text / Paragraph:",
        value=file_content if file_content else sample_remote_sensing,
        height=200
    )

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        start_teach = st.button("🚀 Teach Droid Instantly", type="primary")
    with col_info:
        use_llm_distill = False
        if st.session_state.llm_enabled:
            use_llm_distill = st.checkbox("Distill text with LLM Teacher before learning", value=True)

    if start_teach and teach_text.strip():
        if engine_mode.startswith("Droid"):
            with st.spinner("Processing propositions & updating plastic RTU memory..."):
                if st.session_state.llm_enabled and use_llm_distill:
                    teacher = OpenAITeacher(
                        base_url=st.session_state.llm_base_url,
                        api_key=st.session_state.llm_api_key,
                        model=st.session_state.llm_model
                    )
                    distilled = teacher.distill_propositions(teach_text)
                    text_to_teach = "\n".join(distilled)
                else:
                    text_to_teach = teach_text

                result = active_droid.teach(text_to_teach, source="webui_ingest", auto_tune=True)
                droid_mgr.save_droid(st.session_state.selected_droid_name)

            st.success(
                f"🎉 Knowledge successfully absorbed! "
                f"Learned **{result['propositions']} propositions** in **{result['elapsed_ms']:.1f} ms**. "
                f"Auto-selected LR: `{result['effective_lr']:.4f}` | Memory Norm: `{result['memory_norm']:.2f}`."
            )
            
            # Show propositions learned
            with st.expander("View Learned Propositions", expanded=True):
                recent_facts = active_droid.knowledge_bank[-result['propositions']:]
                for idx, fact in enumerate(recent_facts):
                    st.markdown(f"**{idx+1}.** {fact['text']}")
        else:
            st.info("Raw byte-level training requires character iterations.")

# --- TAB 3: TRANSPARENT AUDIT LOGS ---
with tab_logs:
    st.subheader(f"Audit & Training Logs for {st.session_state.selected_droid_name}")
    
    if engine_mode.startswith("Droid"):
        logs = active_droid.logs
        if logs:
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs)
            if st.button("Clear Logs"):
                active_droid.logs.clear()
                st.rerun()
        else:
            st.info("No logs recorded yet. Start chatting or teaching to generate audit trails.")
    else:
        st.info("Logs only available in Droid Lifelong Engine mode.")

# --- TAB 4: WEIGHTS & MEMORY INSPECTOR ---
with tab_weights:
    st.subheader("Weights, Plastic Decay & Memory State Buffers")

    if engine_mode.startswith("Droid"):
        mem = active_droid.memory
        decay_vals = torch.sigmoid(mem.decay).detach().cpu().numpy()
        states_vals = mem.states.detach().cpu().numpy()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Memory Dimension", f"{active_droid.dim}")
        col_m2.metric("Mean Decay Rate", f"{decay_vals.mean():.4f}")
        col_m3.metric("Max Memory State", f"{states_vals.max():.4f}")
        col_m4.metric("State Norm L2", f"{torch.norm(mem.states).item():.3f}")

        st.markdown("**Plastic Decay Retention Curve Across Hidden Channels**")
        df_decay = pd.DataFrame({"Channel": range(len(decay_vals)), "Decay Retention": decay_vals})
        st.line_chart(df_decay.set_index("Channel"))

        st.markdown("**RTU Hidden Memory State Activations (h_t)**")
        df_states = pd.DataFrame({"Channel": range(len(states_vals)), "State Value": states_vals})
        st.bar_chart(df_states.set_index("Channel"))
    else:
        st.info("Switch to Droid mode or use parameter inspector for raw weights.")
