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
from src.model.propositions import extract_propositions

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

# Feedback state for inline LLM response feedback
if "feedback_state" not in st.session_state:
    st.session_state.feedback_state = {}

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
            st.session_state.feedback_state = {}
            st.rerun()

        new_name = st.text_input("New Droid Name", placeholder="e.g. droid-geospatial")
        if st.button("➕ Create Droid"):
            if new_name.strip():
                clean = new_name.strip().lower().replace(" ", "-")
                droid_mgr.create_droid(clean)
                st.session_state.selected_droid_name = clean
                st.session_state.chat_history = []
                st.session_state.feedback_state = {}
                st.rerun()

        st.markdown(f"**Facts in Memory**: {len(active_droid.knowledge_bank)}")
        st.markdown(f"**Experience Steps**: {active_droid.step_count}")
        try:
            mem_norm = float(torch.norm(active_droid.memory.states).item())
            st.markdown(f"**Memory State Norm**: `{mem_norm:.2f}`")
        except:
            st.markdown(f"**Memory State Norm**: `N/A`")

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
            try:
                active_droid.memory.reset()
            except:
                pass
            active_droid.clear_knowledge()
            active_droid.step_count = 0
            droid_mgr.save_droid(st.session_state.selected_droid_name)
            st.session_state.chat_history = []
            st.session_state.feedback_state = {}
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
tab_chat, tab_teach, tab_memory, tab_logs, tab_weights = st.tabs([
    "💬 Chat & Conversational Teach",
    "📚 Domain Teaching & Ingestion",
    "🧠 Memory Console",
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
        st.caption("Tip: You can talk normally, teach concepts (e.g. 'learn: Remote sensing is...'), or ask questions. After LLM responses, expand Memory Console to select facts.")

    chat_container = st.container()

    # Render conversation above chat input
    with chat_container:
        for msg_idx, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("source_label"):
                    st.caption(f"🏷️ **Source**: {msg['source_label']}")
                if msg.get("facts_recalled"):
                    with st.expander(f"🔍 View {len(msg['facts_recalled'])} Recalled Facts from Plastic Memory"):
                        for idx, f in enumerate(msg["facts_recalled"]):
                            score = f.get("similarity", 0.0)
                            st.markdown(f"**{idx+1}.** {f['text']} *(match score: `{score:.3f}`)*")
                
                # Inline LLM Response Feedback - only for assistant messages with LLM source
                if msg["role"] == "assistant" and msg.get("source_label") and "LLM" in msg.get("source_label", ""):
                    # Extract propositions for this message if not already done
                    if msg_idx not in st.session_state.feedback_state:
                        props = extract_propositions(msg["content"])
                        st.session_state.feedback_state[msg_idx] = {
                            "propositions": props,
                            "selected": set(),
                            "expanded": False
                        }
                    
                    feedback = st.session_state.feedback_state.get(msg_idx, {})
                    propositions = feedback.get("propositions", [])
                    
                    if propositions:
                        with st.expander(f"🧠 Memory Console: Select propositions to learn ({len(propositions)} found)", expanded=feedback.get("expanded", False)):
                            st.caption("Select sentences to promote into persistent Droid memory:")
                            
                            selected_props = []
                            for i, prop in enumerate(propositions):
                                key = f"fb_{msg_idx}_{i}"
                                checked = st.checkbox(prop, key=key)
                                if checked:
                                    selected_props.append(prop)
                            
                            # Custom proposition input
                            custom_prop = st.text_input("Add custom proposition", key=f"custom_fb_{msg_idx}", placeholder="Enter additional fact to learn...")
                            
                            col_accept, col_clear = st.columns([1, 1])
                            with col_accept:
                                if st.button(f"✅ Accept Selected ({len(selected_props) + (1 if custom_prop.strip() else 0)})", key=f"accept_{msg_idx}"):
                                    to_teach = selected_props.copy()
                                    if custom_prop.strip():
                                        to_teach.append(custom_prop.strip())
                                    
                                    if to_teach:
                                        combined = "\n".join(to_teach)
                                        with st.spinner("Teaching selected propositions..."):
                                            res = active_droid.teach(combined, source="chat_feedback")
                                            droid_mgr.save_droid(st.session_state.selected_droid_name)
                                        st.success(f"✅ Absorbed {res['propositions']} facts (norm: {res['memory_norm']:.2f})")
                                        # Collapse after accept
                                        st.session_state.feedback_state[msg_idx]["expanded"] = False
                                        st.rerun()
                                    else:
                                        st.warning("No propositions selected")
                            
                            with col_clear:
                                if st.button("Clear", key=f"clear_{msg_idx}"):
                                    st.session_state.feedback_state[msg_idx]["selected"] = set()
                                    st.rerun()

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
                    # Contextual follow-up fallback
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

# --- TAB 3: MEMORY CONSOLE ---
with tab_memory:
    st.subheader("🧠 Memory Console - Human-in-the-Loop Management")
    st.markdown("Browse, edit, delete facts and teach via LLM distillation workflow.")
    
    if not engine_mode.startswith("Droid"):
        st.info("Memory Console only available in Droid Lifelong Engine mode.")
    else:
        sub_tab_browse, sub_tab_teach, sub_tab_stats = st.tabs(["📖 Browse & Manage", "🤖 LLM Teach", "📊 Stats"])
        
        # Sub-tab 1: Browse & Manage
        with sub_tab_browse:
            st.markdown("### Fact Browser")
            
            # Search
            search_query = st.text_input("Search facts (case-insensitive substring)", key="mem_search", placeholder="e.g. remote sensing, lidar, hydrology...")
            
            # Get facts
            try:
                if search_query.strip():
                    # Simple substring search
                    all_facts = active_droid.knowledge.get_active_facts()
                    filtered = [f for f in all_facts if search_query.lower() in f['text'].lower()]
                else:
                    filtered = active_droid.knowledge.get_active_facts()
                
                # Pagination
                page_size = 50
                total = len(filtered)
                st.caption(f"Showing {min(page_size, total)} of {total} active facts")
                
                # Display as dataframe for overview
                if filtered:
                    # Prepare display data
                    display_data = []
                    for f in filtered[:page_size]:
                        display_data.append({
                            "ID": f['id'],
                            "Text": f['text'][:100] + ("..." if len(f['text']) > 100 else ""),
                            "Source": f.get('source', 'unknown'),
                            "Timestamp": time.strftime('%Y-%m-%d %H:%M', time.localtime(f.get('timestamp', 0))) if f.get('timestamp') else "N/A",
                            "Novelty": f"{f.get('novelty', 0):.3f}",
                            "Access": f.get('access_count', 1)
                        })
                    
                    df = pd.DataFrame(display_data)
                    st.dataframe(df, use_container_width=True, height=300)
                    
                    # Detailed fact management
                    st.markdown("#### Manage Individual Fact")
                    fact_id_input = st.number_input("Fact ID to manage", min_value=1, step=1, value=filtered[0]['id'] if filtered else 1)
                    
                    # Get fact details
                    fact = active_droid.knowledge.get_fact(fact_id_input)
                    if fact:
                        st.json({
                            "id": fact['id'],
                            "text": fact['text'],
                            "source": fact.get('source'),
                            "timestamp": fact.get('timestamp'),
                            "step": fact.get('step'),
                            "novelty": fact.get('novelty'),
                            "superseded": bool(fact.get('superseded')),
                            "valid_until": fact.get('valid_until'),
                            "superseded_by": fact.get('superseded_by'),
                            "access_count": fact.get('access_count'),
                            "last_retrieved": fact.get('last_retrieved'),
                        })
                        
                        # Edit
                        with st.expander("✏️ Edit Fact"):
                            new_text = st.text_area("New fact text", value=fact['text'], key=f"edit_{fact_id_input}", height=100)
                            if st.button("Save Edit", key=f"save_edit_{fact_id_input}"):
                                try:
                                    active_droid.knowledge.update_fact(fact_id_input, new_text)
                                    droid_mgr.save_droid(st.session_state.selected_droid_name)
                                    st.success(f"Fact {fact_id_input} updated and re-embedded")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to update: {e}")
                        
                        # Supersede and Delete in columns
                        col_sup, col_del = st.columns(2)
                        with col_sup:
                            if st.button(f"🚫 Supersede Fact {fact_id_input}", key=f"sup_{fact_id_input}"):
                                try:
                                    active_droid.knowledge.supersede(fact_id_input, 0)
                                    droid_mgr.save_droid(st.session_state.selected_droid_name)
                                    st.success(f"Fact {fact_id_input} marked as superseded")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to supersede: {e}")
                        
                        with col_del:
                            if st.button(f"🗑️ Delete Fact {fact_id_input}", key=f"del_{fact_id_input}"):
                                # Simple confirmation via checkbox
                                confirm_key = f"confirm_del_{fact_id_input}"
                                if confirm_key not in st.session_state:
                                    st.session_state[confirm_key] = False
                                
                                if st.checkbox(f"Confirm delete fact {fact_id_input}?", key=confirm_key):
                                    try:
                                        active_droid.knowledge.delete_fact(fact_id_input)
                                        droid_mgr.save_droid(st.session_state.selected_droid_name)
                                        st.success(f"Fact {fact_id_input} deleted")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to delete: {e}")
                                else:
                                    st.warning("Check confirmation box to delete")
                    else:
                        st.warning(f"Fact ID {fact_id_input} not found")
                else:
                    st.info("No facts found matching search")
                    
            except Exception as e:
                st.error(f"Error browsing facts: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Sub-tab 2: LLM Teach
        with sub_tab_teach:
            st.markdown("### LLM Teach Workflow")
            st.caption("Paste raw text → Distill into propositions → Review → Teach selected")
            
            teach_input = st.text_area("Paste text to distill and teach", height=200, key="llm_teach_input", placeholder="Paste article, definition, or any domain text...")
            
            col_distill, col_local = st.columns([1, 1])
            with col_distill:
                distill_btn = st.button("🤖 Distill with LLM", type="primary", key="distill_llm")
            with col_local:
                local_btn = st.button("✂️ Split Locally", key="split_local")
            
            # Session state for distillation results
            if "llm_teach_propositions" not in st.session_state:
                st.session_state.llm_teach_propositions = []
            
            if distill_btn and teach_input.strip():
                if st.session_state.llm_enabled:
                    with st.spinner(f"Distilling via {st.session_state.llm_model}..."):
                        try:
                            teacher = OpenAITeacher(
                                base_url=st.session_state.llm_base_url,
                                api_key=st.session_state.llm_api_key,
                                model=st.session_state.llm_model
                            )
                            distilled = teacher.distill_propositions(teach_input)
                            st.session_state.llm_teach_propositions = distilled
                            st.success(f"Distilled {len(distilled)} propositions")
                        except Exception as e:
                            st.error(f"LLM distillation failed: {e}, falling back to local split")
                            st.session_state.llm_teach_propositions = extract_propositions(teach_input)
                else:
                    st.warning("LLM Provider not enabled, using local split")
                    st.session_state.llm_teach_propositions = extract_propositions(teach_input)
            
            if local_btn and teach_input.strip():
                st.session_state.llm_teach_propositions = extract_propositions(teach_input)
                st.success(f"Split into {len(st.session_state.llm_teach_propositions)} propositions")
            
            # Display results as checkboxes
            if st.session_state.llm_teach_propositions:
                st.markdown(f"#### Review {len(st.session_state.llm_teach_propositions)} Propositions")
                
                selected = []
                for i, prop in enumerate(st.session_state.llm_teach_propositions):
                    if st.checkbox(prop, key=f"teach_{i}", value=True):
                        selected.append(prop)
                
                if st.button(f"🚀 Teach Selected ({len(selected)})", type="primary", key="teach_selected"):
                    if selected:
                        combined = "\n".join(selected)
                        with st.spinner("Teaching selected propositions..."):
                            res = active_droid.teach(combined, source="memory_console_llm")
                            droid_mgr.save_droid(st.session_state.selected_droid_name)
                        st.success(f"🎉 Absorbed {res['propositions']} facts in {res['elapsed_ms']:.1f}ms (norm: {res['memory_norm']:.2f})")
                        st.session_state.llm_teach_propositions = []
                        st.rerun()
                    else:
                        st.warning("No propositions selected")
        
        # Sub-tab 3: Stats
        with sub_tab_stats:
            st.markdown("### Knowledge Store Statistics")
            
            try:
                active_count = active_droid.knowledge.active_count()
                total_count = active_droid.knowledge.total_count()
                superseded_count = total_count - active_count
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Active Facts", active_count)
                col2.metric("Superseded Facts", superseded_count)
                col3.metric("Total Facts", total_count)
                
                # Sources breakdown
                all_facts = active_droid.knowledge.get_active_facts()
                if all_facts:
                    sources = {}
                    for f in all_facts:
                        src = f.get('source', 'unknown')
                        sources[src] = sources.get(src, 0) + 1
                    
                    st.markdown("#### Sources Breakdown")
                    df_sources = pd.DataFrame([{"Source": k, "Count": v} for k, v in sources.items()])
                    st.bar_chart(df_sources.set_index("Source"))
                    
                    # Date range
                    timestamps = [f.get('timestamp', 0) for f in all_facts if f.get('timestamp')]
                    if timestamps:
                        min_ts = min(timestamps)
                        max_ts = max(timestamps)
                        st.markdown(f"**Date Range**: {time.strftime('%Y-%m-%d %H:%M', time.localtime(min_ts))} to {time.strftime('%Y-%m-%d %H:%M', time.localtime(max_ts))}")
                    
                    # Memory metrics
                    st.markdown("#### Memory Metrics")
                    col_m1, col_m2 = st.columns(2)
                    try:
                        mem_norm = float(torch.norm(active_droid.memory.states).item())
                        col_m1.metric("Memory Norm", f"{mem_norm:.2f}")
                    except:
                        col_m1.metric("Memory Norm", "N/A")
                    col_m2.metric("Step Count", active_droid.step_count)
                    
                    # Access counts
                    access_counts = [f.get('access_count', 1) for f in all_facts]
                    if access_counts:
                        st.markdown(f"**Avg Access Count**: {sum(access_counts)/len(access_counts):.1f}, Max: {max(access_counts)}")
                else:
                    st.info("No facts in knowledge store")
                    
            except Exception as e:
                st.error(f"Error loading stats: {e}")
                import traceback
                st.code(traceback.format_exc())

# --- TAB 4: TRANSPARENT AUDIT LOGS ---
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

# --- TAB 5: WEIGHTS & MEMORY INSPECTOR ---
with tab_weights:
    st.subheader("Weights, Plastic Decay & Memory State Buffers")

    if engine_mode.startswith("Droid"):
        mem = active_droid.memory
        try:
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
        except Exception as e:
            st.error(f"Error inspecting memory: {e}")
    else:
        st.info("Switch to Droid mode or use parameter inspector for raw weights.")
