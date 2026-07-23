# Gradio UI for Enhanced RAG Chatbot
import gradio as gr
from main import EnhancedRAGChatbot
import os
import re
from config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_USE_EXPANSION,
    DEFAULT_ENABLE_RERANKING,
    DEFAULT_SHOW_DEBUG,
    GRADIO_SERVER_PORT,
    SUPPORTED_LANGUAGES,
)

def _apply_gradio_api_info_hotfix():
    """
    Work around Gradio schema parsing crashes seen on Spaces for some
    component schemas (e.g., additionalProperties=true).
    """
    try:
        from gradio.blocks import Blocks
    except Exception:
        return

    if getattr(Blocks, "_reviewer_api_info_patched", False):
        return

    original_get_api_info = Blocks.get_api_info

    def _safe_get_api_info(self):
        try:
            return original_get_api_info(self)
        except Exception as e:
            print(f"Gradio API info generation failed; continuing without API metadata: {e}")
            return {"named_endpoints": {}, "unnamed_endpoints": {}}

    Blocks.get_api_info = _safe_get_api_info
    Blocks._reviewer_api_info_patched = True

def _apply_gradio_schema_hotfix():
    """
    Guard gradio_client schema conversion against malformed schema nodes such
    as `additionalProperties: true` that can raise TypeError in some versions.
    """
    try:
        import gradio_client.utils as gc_utils
    except Exception:
        return

    if getattr(gc_utils, "_reviewer_schema_patched", False):
        return

    original_json_schema_to_python_type = gc_utils.json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema):
        try:
            return original_json_schema_to_python_type(schema)
        except TypeError as e:
            if "bool" in str(e):
                return "Any"
            raise
        except Exception:
            return "Any"

    gc_utils.json_schema_to_python_type = _safe_json_schema_to_python_type
    gc_utils._reviewer_schema_patched = True

def launch():
    _apply_gradio_api_info_hotfix()
    _apply_gradio_schema_hotfix()
    rag = EnhancedRAGChatbot()
    language_name_to_code = {name: code for code, name in SUPPORTED_LANGUAGES.items()}
    language_choices = list(language_name_to_code.keys())
    
    # Initial status from pre-defined keys
    initial_gemini_status = f"**Gemini Ready!**\nModel: {rag.gen.model_name}\nStatus: Pre-configured" if rag.gen.api_key_set else "*API key not set*"
    initial_pinecone_status = "**Vector store**\nConnected to Pinecone (cloud)." if getattr(rag.vs, "pinecone_index", None) is not None else "**Vector store**\nUsing local storage (FAISS)."

    def ui_clear():
        cleared_status, cleared_chat, cleared_stats = rag.clear()
        return (cleared_status, cleared_chat, cleared_stats, gr.update(value=None), gr.update(value=""),
                gr.update(value=""), gr.update(value="English"),
                gr.update(value="*API key not set*"),
                gr.update(value="*Using local storage*"))

    def split_answer_and_debug(answer_text):
        if not answer_text:
            return "", ""
        # Primary split: debug block is always appended after a markdown divider.
        divider = re.search(r"\n\s*---\s*\n", answer_text)
        if divider:
            idx = divider.start()
            clean_answer = answer_text[:idx].strip()
            details = answer_text[idx:].strip()
            return clean_answer, details

        # Fallback for historical formats where divider is missing.
        m = re.search(
            r"(\n\s*---\s*\n\s*(?:\*\*)?Method Used:?(?:\*\*)?|\n\s*(?:\*\*)?Method Used:?(?:\*\*))",
            answer_text,
            flags=re.IGNORECASE,
        )
        if not m:
            return answer_text, ""
        idx = m.start()
        clean_answer = answer_text[:idx].strip()
        details = answer_text[idx:].strip()
        return clean_answer, details

    def attach_inline_details(answer_text, details_text):
        if not details_text:
            return answer_text
        details_clean = re.sub(r"^\s*---\s*", "", details_text.strip(), flags=re.S)
        # Per-reply info toggle shown at right side of each assistant response.
        return (
            f"{answer_text}\n\n"
            f"<div class='msg-info-holder'>"
            f"<details class='msg-info-details'>"
            f"<summary class='msg-info-btn'><b><i>i</i></b></summary>"
            f"<div class='msg-info-content'>{details_clean}</div>"
            f"</details>"
            f"</div>"
        )

    custom_css = """
    .gradio-container .share-btn,
    .gradio-container button[title="Share"],
    .gradio-container [aria-label="Share"] {
        display: none !important;
    }
    .msg-info-holder {
        display: flex;
        justify-content: flex-end;
        margin-top: 4px;
    }
    .msg-info-details {
        position: relative;
    }
    .msg-info-btn {
        list-style: none;
        cursor: pointer;
        width: 30px;
        height: 30px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.45);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.08);
        color: #fff;
        font-size: 18px;
        line-height: 1;
        user-select: none;
    }
    .msg-info-btn::marker,
    .msg-info-btn::-webkit-details-marker {
        display: none;
        content: "";
    }
    .msg-info-content {
        margin-top: 8px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        background: rgba(0,0,0,0.18);
        white-space: pre-wrap;
        font-size: 0.9rem;
    }
    /* Make chat avatars easier to see */
    .gradio-container .avatar-container img,
    .gradio-container img.avatar-image {
        width: 42px !important;
        height: 42px !important;
    }
    """
    with gr.Blocks(title="Document Q&A Assistant", theme=gr.themes.Soft(), css=custom_css) as demo:
        # Header with title
        gr.HTML("<div style='text-align: center; padding: 20px;'><h1>📄 Document Q&A Assistant</h1><p style='font-size: 16px; color: #666;'>Ask questions about your PDF documents and get instant answers</p></div>")

        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                # API Configuration Section
                with gr.Accordion("🔑 API Configuration", open=True):
                    api_key = gr.Textbox(
                        label="Google Gemini API Key", 
                        type="password", 
                        placeholder="Pre-defined key in use. Enter to override...",
                        info="Pre-configured key used by default. Override if needed."
                    )
                    set_key_btn = gr.Button("Set API Key", variant="primary", size="sm")
                    gemini_status = gr.Markdown(initial_gemini_status, elem_classes=["status-text"])
                    
                    gr.Markdown("---")
                    
                    pinecone_api_key = gr.Textbox(
                        label="Pinecone API Key (Optional)", 
                        type="password", 
                        placeholder="Pre-defined key in use. Enter to override...",
                        info="Uses cloud vector database. Pre-configured key used by default."
                    )
                    set_pinecone_btn = gr.Button("Set Pinecone Key", variant="secondary", size="sm")
                    pinecone_status = gr.Markdown(initial_pinecone_status, elem_classes=["status-text"])

                # Document Upload Section
                with gr.Accordion("📤 Upload Documents", open=True):
                    files = gr.File(
                        label="Upload PDF Files", 
                        file_count="multiple", 
                        file_types=[".pdf"], 
                        type="filepath"
                    )
                    process_btn = gr.Button("Process Documents", variant="primary", size="lg")
                    status = gr.Textbox(
                        label="Status", 
                        lines=8, 
                        interactive=False, 
                        value="Ready to process documents...",
                        show_label=False
                    )

                # Document Preview Section
                with gr.Accordion("📄 Document Preview", open=False):
                    doc_preview_select = gr.Dropdown(
                        choices=["all"],
                        value="all",
                        label="Select Document",
                        info="View full text or browse chunks"
                    )
                    preview_mode = gr.Radio(
                        choices=["Full Text", "Browse Chunks", "Segmentation Info"],
                        value="Full Text",
                        label="Preview Mode"
                    )
                    preview_btn = gr.Button("View Preview", variant="secondary", size="sm")
                    preview_output = gr.Markdown(
                        label="Preview",
                        show_label=False
                    )

                # System info (advanced knobs removed; app uses tuned defaults)
                with gr.Accordion("⚙️ System Info", open=False):
                    with gr.Row():
                        stats_btn = gr.Button("📊 System Info", variant="secondary", size="sm")
                        metrics_btn = gr.Button("📈 Performance", variant="secondary", size="sm")
                    stats_box = gr.Textbox(label="Information", lines=15, interactive=False, visible=False)
                
                # Export Functionality
                with gr.Accordion("💾 Export Conversation", open=False):
                    export_format = gr.Dropdown(
                        choices=["pdf", "docx", "txt"], 
                        value="pdf", 
                        label="Format"
                    )
                    export_filename = gr.Textbox(
                        label="Filename (optional)", 
                        placeholder="Leave empty for auto-generated name"
                    )
                    export_conversation_btn = gr.Button("Export", variant="secondary")
                    export_status = gr.Textbox(
                        label="Status", 
                        lines=2, 
                        interactive=False
                    )
                    export_download_btn = gr.DownloadButton("Download", visible=False)

                clear_btn = gr.Button("🗑️ Clear All", variant="stop", size="lg")

            with gr.Column(scale=2):
                chat = gr.Chatbot(
                    label="Chat",
                    height=650,
                    show_label=False,
                    avatar_images=("https://i.pravatar.cc/100?img=68", "https://robohash.org/contract-bot.png?size=100x100&set=set3"),
                    sanitize_html=False,
                )
                
                # Question Input Section
                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Ask any question about your documents...", 
                        lines=2, 
                        scale=5,
                        show_label=False,
                        container=False
                    )
                    ask_btn = gr.Button("Ask ➤", variant="primary", scale=1, size="lg")
                
                # Document Comparison
                with gr.Accordion("🔀 Compare Documents", open=False):
                    compare_doc1 = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Document 1",
                        info="Select first document"
                    )
                    compare_doc2 = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Document 2",
                        info="Select second document"
                    )
                    compare_question = gr.Textbox(
                        label="Comparison Question (optional)",
                        placeholder="Leave empty for general comparison",
                        lines=2
                    )
                    compare_btn = gr.Button("Compare", variant="secondary", size="sm")
                    compare_output = gr.Markdown(
                        label="Comparison Result",
                    )

                # Optional Controls (collapsed by default)
                with gr.Accordion("🌐 Language & Options", open=False):
                    target_language = gr.Dropdown(
                        choices=language_choices,
                        value="English",
                        label="Answer Language",
                    )

        # Bind actions
        set_key_btn.click(fn=lambda k: rag.set_gemini(k), inputs=[api_key], outputs=[status, gemini_status], show_progress="hidden")
        set_pinecone_btn.click(fn=lambda k: rag.set_pinecone(k), inputs=[pinecone_api_key], outputs=[status, pinecone_status], show_progress="hidden")
        
        # Process documents and then update document selector
        def process_and_update(files):
            # First process documents
            result = rag.process_documents(files)
            preview_update = update_preview_selector()
            comp1_update, comp2_update = update_comparison_selectors()
            print(f"Processing complete. Document sections available: {rag.document_sections}")
            return result, preview_update, comp1_update, comp2_update
        
        process_btn.click(fn=process_and_update, inputs=[files], outputs=[status, doc_preview_select, compare_doc1, compare_doc2], show_progress="full")
        
        # Update document preview selector
        def update_preview_selector():
            if rag.processed:
                choices = ["all"] + [d['file'] for d in rag.processed]
                return gr.update(choices=choices, value="all")
            return gr.update(choices=["all"], value="all")
        
        # Update comparison selectors (no "all" option)
        def update_comparison_selectors():
            if rag.processed and len(rag.processed) >= 2:
                choices = [d['file'] for d in rag.processed]
                return gr.update(choices=choices, value=choices[0] if len(choices) > 0 else None), \
                       gr.update(choices=choices, value=choices[1] if len(choices) > 1 else choices[0] if len(choices) > 0 else None)
            return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
        
        # Document preview handler
        def show_preview(doc_name, mode):
            if mode == "Full Text":
                return rag.get_document_preview(doc_name if doc_name != "all" else None)
            elif mode == "Browse Chunks":
                return rag.browse_chunks(doc_name if doc_name != "all" else None)
            else:  # Segmentation Info
                return rag.get_document_segmentation(doc_name if doc_name != "all" else None)
        
        # Handle queries with document and language support
        def handle_query(question, history, target_lang_name):
            # Advanced settings removed from UI; keep tuned defaults in config.
            k = DEFAULT_TOP_K
            conf = DEFAULT_CONFIDENCE_THRESHOLD
            exp = DEFAULT_USE_EXPANSION
            rerank = DEFAULT_ENABLE_RERANKING
            # Always generate debug internally, but hide it from chat unless user clicks.
            debug = True
            max_tokens = 0
            target_lang = language_name_to_code.get((target_lang_name or "").strip(), "en")
            if target_lang != "en":
                # Use bilingual answer method
                empty, new_history = rag.answer_with_translation(question, history, target_lang, k, conf, exp, rerank, max_tokens, debug)
            else:
                # Use regular answer method for all documents
                empty, new_history = rag.answer(question, history, k, conf, exp, max_tokens, rerank, debug)
            
            details_payload = ""
            if len(new_history) > len(history):
                last_q, last_a = new_history[-1]
                clean_answer, details_payload = split_answer_and_debug(last_a)
                new_history[-1] = (last_q, clean_answer)

            if len(new_history) > len(history):
                q, a = new_history[-1]
                new_history[-1] = (q, attach_inline_details(a, details_payload))

            return (
                empty,
                new_history,
            )
        
        ask_btn.click(fn=handle_query,
                      inputs=[question_input, chat, target_language],
                      outputs=[question_input, chat], show_progress="full")
        question_input.submit(fn=handle_query,
                              inputs=[question_input, chat, target_language],
                              outputs=[question_input, chat], show_progress="full")
        
        # Document preview
        process_btn.click(fn=update_preview_selector, outputs=[doc_preview_select], show_progress="hidden").then(
            fn=update_preview_selector, outputs=[compare_doc1]
        ).then(
            fn=update_preview_selector, outputs=[compare_doc2]
        )
        preview_btn.click(fn=show_preview, inputs=[doc_preview_select, preview_mode], outputs=[preview_output], show_progress="full")
        
        # Document comparison
        def handle_compare(doc1, doc2, question):
            if not doc1 or not doc2:
                return "Please select two documents to compare."
            if doc1 == doc2:
                return "Please select two different documents."
            return rag.compare_documents(doc1, doc2, question if question and question.strip() else None)
        
        compare_btn.click(fn=handle_compare, inputs=[compare_doc1, compare_doc2, compare_question], outputs=[compare_output], show_progress="full")
        
        # Stats box visibility toggle
        stats_btn.click(fn=lambda: rag.stats(), outputs=[stats_box], show_progress="hidden").then(
            fn=lambda: gr.update(visible=True), outputs=[stats_box]
        )
        metrics_btn.click(fn=lambda: rag.get_performance_metrics(), outputs=[stats_box], show_progress="hidden").then(
            fn=lambda: gr.update(visible=True), outputs=[stats_box]
        )

        clear_btn.click(fn=ui_clear, outputs=[status, chat, stats_box, files, question_input,
                                              api_key, target_language, gemini_status, pinecone_status],
                        show_progress="hidden").then(
            fn=lambda: gr.update(visible=False), outputs=[stats_box]
        )
        
        # Export functionality
        def handle_conversation_export(fmt, filename, chat_history):
            result = rag.export_conversation(chat_history, fmt, filename)
            if result[0] and os.path.exists(result[0]):  # If file was created
                return result[1], gr.update(value=result[0], visible=True)
            else:
                return result[1], gr.update(value=None, visible=False)
        
        export_conversation_btn.click(
            fn=handle_conversation_export,
            inputs=[export_format, export_filename, chat],
            outputs=[export_status, export_download_btn],
            show_progress="full"
        )

    # Simple launch - just open in browser
    print("Launching Enhanced RAG Chatbot...")
    print(f"Using Gemini model: {rag.gen.model_name}")
    print("SINGLE MODEL CONFIGURATION - No model selection needed!")
    print("Simple browser launch - ready for cloud deployment.")
    
    # Cloud platforms can provide PORT dynamically.
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", GRADIO_SERVER_PORT)))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    share = os.environ.get("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")
    is_hf_space = bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))
    if is_hf_space:
        # Spaces requires external access; share=False can fail startup.
        share = True
        print("Detected Hugging Face Spaces runtime; forcing share=True.")
    print(f"Opening in browser (port {port})...")
    # Try to launch on specified port, but allow Gradio to find an available port if needed
    import socket
    def is_port_available(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
    
    def _launch_with_compat(**kwargs):
        try:
            demo.launch(**kwargs)
        except TypeError as e:
            # Older Gradio builds may not support show_api.
            if "show_api" in str(e):
                kwargs.pop("show_api", None)
                demo.launch(**kwargs)
            else:
                raise

    # If port is not available, let Gradio find one automatically
    if not is_port_available(port):
        print(f"Port {port} is in use, Gradio will find an available port automatically...")
        _launch_with_compat(
            debug=True,
            show_error=True,
            quiet=False,
            show_api=False,
            share=share,
            server_name=server_name
        )
    else:
        _launch_with_compat(
            debug=True,
            show_error=True,
            quiet=False,
            show_api=False,
            server_port=port,
            share=share,
            server_name=server_name
        )

if __name__ == "__main__":
    launch()
