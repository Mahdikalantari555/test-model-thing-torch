
import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional

# ponytail: unified headless CLI - tmt-droid entrypoint

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def cmd_chat(args):
    """Interactive terminal chat loop."""
    from src.model.droid_manager import DroidManager
    mgr = DroidManager(base_dir=args.base_dir)
    droid = mgr.get_droid(args.name)
    
    mode = getattr(args, 'mode', 'auto')
    print(f"Chatting with {args.name} (mode={mode}). Type 'exit' or Ctrl+C to quit.", file=sys.stderr)
    
    if args.query:
        # Single query mode
        query = args.query
        # Handle /fast /slow overrides
        if query.startswith('/fast '):
            mode = 'fast'
            query = query[5:].strip()
        elif query.startswith('/slow '):
            mode = 'slow'
            query = query[6:].strip()
        
        if mode == 'fast' or (mode == 'auto' and hasattr(droid, 'decide_path')):
            try:
                if mode == 'fast':
                    hits = droid.recall(query, top_k=4)
                    if hits:
                        # Synthesize without LLM
                        parts = []
                        for h in hits:
                            t = h["text"].strip()
                            if t not in parts:
                                parts.append(t)
                        print(" ".join(parts))
                    else:
                        print(droid.chat(query))
                elif hasattr(droid, 'decide_path'):
                    decision = droid.decide_path(query)
                    if decision.get('action') == 'fast_recall' or decision.get('path') == 'fast':
                        print(f"[fast path, conf={decision.get('confidence',0):.2f}]")
                        hits = droid.recall(query, top_k=4)
                        if hits:
                            print(" ".join([h["text"] for h in hits]))
                        else:
                            print(droid.chat(query))
                    else:
                        print(f"[slow path, conf={decision.get('confidence',0):.2f}]")
                        if hasattr(droid, 'generate_with_memory'):
                            result = droid.generate_with_memory(query)
                            if isinstance(result, dict):
                                print(result.get('text', str(result)))
                            else:
                                print(result)
                        else:
                            print(droid.chat(query))
                else:
                    print(droid.chat(query))
            except Exception as e:
                print(droid.chat(query))
                print(f"[error in mode handling: {e}]", file=sys.stderr)
        else:
            # Check for explicit mode
            if mode == 'slow' and hasattr(droid, 'generate_with_memory'):
                result = droid.generate_with_memory(query)
                if isinstance(result, dict):
                    print(result.get('text', str(result)))
                else:
                    print(result)
            else:
                print(droid.chat(query))
        return
    
    # Interactive loop
    try:
        while True:
            try:
                user_input = input(f"{args.name}> ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ('exit', 'quit', 'q'):
                break
            
            # Handle /fast /slow
            cur_mode = mode
            query = user_input
            if user_input.startswith('/fast '):
                cur_mode = 'fast'
                query = user_input[5:].strip()
            elif user_input.startswith('/slow '):
                cur_mode = 'slow'
                query = user_input[6:].strip()
            
            try:
                if cur_mode == 'fast':
                    hits = droid.recall(query, top_k=4)
                    if hits:
                        parts = []
                        for h in hits:
                            t = h["text"].strip()
                            if not any(t in p or p in t for p in parts):
                                parts.append(t)
                        print(" ".join(parts))
                    else:
                        print(droid.chat(query))
                elif cur_mode == 'slow' and hasattr(droid, 'generate_with_memory'):
                    result = droid.generate_with_memory(query)
                    if isinstance(result, dict):
                        print(result.get('text', str(result)))
                    else:
                        print(result)
                else:
                    # auto
                    if hasattr(droid, 'decide_path'):
                        decision = droid.decide_path(query)
                        if decision.get('action') in ('fast_recall', 'recall') or decision.get('path') == 'fast':
                            hits = droid.recall(query, top_k=4)
                            if hits:
                                print(" ".join([h["text"] for h in hits]))
                            else:
                                print(droid.chat(query))
                        else:
                            if hasattr(droid, 'generate_with_memory'):
                                result = droid.generate_with_memory(query)
                                if isinstance(result, dict):
                                    print(result.get('text', str(result)))
                                else:
                                    print(result)
                            else:
                                print(droid.chat(query))
                    else:
                        print(droid.chat(query))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
    except KeyboardInterrupt:
        print("\nExiting.", file=sys.stderr)

def cmd_teach(args):
    """Teach from stdin markdown ingestion."""
    from src.model.droid_manager import DroidManager
    mgr = DroidManager(base_dir=args.base_dir)
    droid = mgr.get_droid(args.name)
    
    # Read from stdin or file
    if args.file:
        text = Path(args.file).read_text(encoding='utf-8')
    else:
        # Read stdin
        if sys.stdin.isatty():
            print("Reading from stdin (Ctrl+D to finish)...", file=sys.stderr)
        text = sys.stdin.read()
    
    if not text.strip():
        print("No input text provided.", file=sys.stderr)
        return 1
    
    result = droid.teach(text, source=args.source or "cli")
    mgr.save_droid(args.name)
    
    print(f"Absorbed {result.get('propositions',0)} facts (episodic: {result.get('episodic_absorbed', result.get('propositions',0))})")
    print(f"Memory norm: {result.get('memory_norm',0):.2f}, avg surprise: {result.get('avg_surprise',0):.3f}")
    print(f"Total knowledge: {result.get('total_knowledge',0)}")
    return 0

def cmd_decide(args):
    """Score options using System-1 decision head."""
    from src.model.decision_head import DecisionHead
    from src.model.droid_manager import DroidManager
    import torch
    
    # Try to get embedding from droid anchor
    mgr = DroidManager(base_dir=args.base_dir)
    try:
        droid = mgr.get_droid(args.name)
        q_emb = droid.anchor.embed(args.query)
        if q_emb.dim() > 1:
            q_emb = q_emb.squeeze(0)
    except Exception as e:
        print(f"Warning: could not get anchor embedding, using random: {e}", file=sys.stderr)
        torch.manual_seed(hash(args.query) % 10000)
        q_emb = torch.randn(384)
    
    # If options provided, create decision head with options as prototypes
    if args.options:
        options = [o.strip() for o in args.options.split(',')]
        # Try to embed options
        option_embs = {}
        try:
            mgr = DroidManager(base_dir=args.base_dir)
            droid = mgr.get_droid(args.name)
            for opt in options:
                try:
                    emb = droid.anchor.embed(opt)
                    if emb.dim() > 1:
                        emb = emb.squeeze(0)
                    import torch.nn.functional as F
                    import torch
                    emb = emb / (torch.linalg.vector_norm(emb).clamp(min=1e-8))
                    option_embs[opt] = emb.tolist()
                except:
                    pass
        except:
            pass
        
        if option_embs:
            dh = DecisionHead(prototypes=option_embs)
        else:
            dh = DecisionHead(prototype_names=options)
    else:
        dh = DecisionHead()
    
    decision = dh.decide(q_emb)
    
    print(f"Query: {args.query}")
    print(f"Selected action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.3f}")
    print(f"Latency: {decision['latency_ms']:.2f}ms")
    print(f"Probabilities:")
    for name, prob, sim in zip(decision['prototype_names'], decision['probabilities'], decision['similarities']):
        print(f"  {name}: prob={prob:.3f} sim={sim:.3f}")
    
    if args.json:
        print(json.dumps(decision, indent=2))
    
    return 0

def cmd_serve(args):
    """Serve MCP or REST."""
    if args.mcp:
        from src.mcp_server import MCPServer
        server = MCPServer(base_dir=args.base_dir)
        server.run_stdio()
    elif args.rest:
        # Lightweight REST microdaemon
        from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
        import json as json_lib
        
        base_dir = args.base_dir
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json_lib.dumps({"status": "ok"}).encode())
                elif self.path == '/droids':
                    from pathlib import Path
                    base = Path(base_dir)
                    droids = [d.name for d in base.iterdir() if d.is_dir()] if base.exists() else []
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json_lib.dumps({"droids": droids}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                try:
                    data = json_lib.loads(body.decode('utf-8'))
                except:
                    self.send_response(400)
                    self.end_headers()
                    return
                
                path = self.path
                try:
                    from src.model.droid_manager import DroidManager
                    mgr = DroidManager(base_dir=base_dir)
                    
                    if path == '/recall':
                        query = data.get('query', '')
                        droid_name = data.get('droid_name', 'droid-alpha')
                        top_k = data.get('top_k', 3)
                        droid = mgr.get_droid(droid_name)
                        hits = droid.recall(query, top_k=top_k)
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json_lib.dumps({"hits": hits}).encode())
                    elif path == '/teach':
                        text = data.get('text', '')
                        droid_name = data.get('droid_name', 'droid-alpha')
                        droid = mgr.get_droid(droid_name)
                        result = droid.teach(text)
                        mgr.save_droid(droid_name)
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json_lib.dumps(result).encode())
                    elif path == '/chat':
                        query = data.get('query', '')
                        droid_name = data.get('droid_name', 'droid-alpha')
                        droid = mgr.get_droid(droid_name)
                        reply = droid.chat(query)
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json_lib.dumps({"reply": reply}).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json_lib.dumps({"error": str(e)}).encode())
            
            def log_message(self, format, *args):
                # Log to stderr
                sys.stderr.write("%s - - [%s] %s\n" %
                                 (self.client_address[0],
                                  self.log_date_time_string(),
                                  format%args))
        
        port = args.port or 8000
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        print(f"REST microdaemon serving on http://127.0.0.1:{port}", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.", file=sys.stderr)
    else:
        print("Must specify --mcp or --rest", file=sys.stderr)
        return 1

def cmd_package(args):
    """Package export/import."""
    from src.model.droid_package import DroidPackage
    
    if args.package_cmd == 'export':
        profile_dir = Path(args.base_dir) / args.name
        if not profile_dir.exists():
            print(f"Droid profile not found: {profile_dir}", file=sys.stderr)
            return 1
        result = DroidPackage.export_droid(str(profile_dir), args.out)
        print(f"Exported {args.name} to {args.out}")
        print(f"Files: {result['files']}")
    elif args.package_cmd == 'import':
        target_dir = Path(args.base_dir) / args.name
        result = DroidPackage.import_droid(args.archive, str(target_dir))
        print(f"Imported {args.archive} to {target_dir}")
        print(f"Verified: {result['verified']}")
    elif args.package_cmd == 'list':
        from src.model.droid_package import DroidPackage
        result = DroidPackage.list_contents(args.archive)
        print(f"Archive: {args.archive}")
        print(f"Files: {result['files']}")
        print(json.dumps(result['manifest'], indent=2))
    else:
        print(f"Unknown package subcommand: {args.package_cmd}", file=sys.stderr)
        return 1
    return 0

def cmd_eval(args):
    """Evaluation subcommand."""
    print("Eval subcommand not yet implemented", file=sys.stderr)
    return 0

def cmd_download_model(args):
    """Download GGUF model."""
    try:
        from src.model.gguf_backend import GgufBackend
        backend = GgufBackend()
        path = backend.download_model(args.repo_id, args.filename, args.local_dir)
        print(f"Downloaded model to {path}")
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        # Fallback to huggingface_hub direct
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, local_dir=args.local_dir)
            print(f"Downloaded via huggingface_hub to {path}")
        except Exception as e2:
            print(f"Fallback also failed: {e2}", file=sys.stderr)
            return 1
    return 0

def cmd_generate(args):
    """Direct LLM generation with memory."""
    from src.model.droid_manager import DroidManager
    mgr = DroidManager(base_dir=args.base_dir)
    droid = mgr.get_droid(args.name)
    
    if hasattr(droid, 'generate_with_memory'):
        result = droid.generate_with_memory(args.query, strategy=args.strategy)
        if isinstance(result, dict):
            print(result.get('text', json.dumps(result, indent=2)))
            if args.verbose:
                print(json.dumps(result, indent=2), file=sys.stderr)
        else:
            print(result)
    else:
        print("GGUF backend not available, falling back to chat", file=sys.stderr)
        print(droid.chat(args.query))

def main():
    parser = argparse.ArgumentParser(prog='tmt-droid', description='Unified headless CLI for Droid')
    parser.add_argument('--base-dir', default='droids', help='Droids base directory')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # chat
    p_chat = subparsers.add_parser('chat', help='Interactive terminal chat')
    p_chat.add_argument('--name', default='droid-alpha', help='Droid name')
    p_chat.add_argument('--mode', choices=['auto', 'fast', 'slow'], default='auto', help='Generation mode')
    p_chat.add_argument('query', nargs='?', help='Single query (if omitted, interactive loop)')
    p_chat.set_defaults(func=cmd_chat)
    
    # teach
    p_teach = subparsers.add_parser('teach', help='Teach from stdin or file')
    p_teach.add_argument('--name', default='droid-alpha', help='Droid name')
    p_teach.add_argument('--file', help='File to read instead of stdin')
    p_teach.add_argument('--source', help='Source tag')
    p_teach.set_defaults(func=cmd_teach)
    
    # decide
    p_decide = subparsers.add_parser('decide', help='System-1 option scoring')
    p_decide.add_argument('query', help='Query text')
    p_decide.add_argument('--options', help='Comma-separated options')
    p_decide.add_argument('--name', default='droid-alpha', help='Droid name')
    p_decide.add_argument('--json', action='store_true', help='Output JSON')
    p_decide.set_defaults(func=cmd_decide)
    
    # serve
    p_serve = subparsers.add_parser('serve', help='Serve MCP or REST')
    p_serve.add_argument('--mcp', action='store_true', help='Serve MCP stdio')
    p_serve.add_argument('--rest', action='store_true', help='Serve REST microdaemon')
    p_serve.add_argument('--port', type=int, default=8000, help='REST port')
    p_serve.set_defaults(func=cmd_serve)
    
    # package
    p_pkg = subparsers.add_parser('package', help='Portable brain packages')
    p_pkg_sub = p_pkg.add_subparsers(dest='package_cmd', required=True)
    p_export = p_pkg_sub.add_parser('export', help='Export droid to .droid archive')
    p_export.add_argument('name', help='Droid name')
    p_export.add_argument('--out', required=True, help='Output .droid file')
    p_import = p_pkg_sub.add_parser('import', help='Import .droid archive')
    p_import.add_argument('archive', help='Archive path')
    p_import.add_argument('--name', required=True, help='Target droid name')
    p_list = p_pkg_sub.add_parser('list', help='List archive contents')
    p_list.add_argument('archive', help='Archive path')
    p_pkg.set_defaults(func=cmd_package)
    
    # eval
    p_eval = subparsers.add_parser('eval', help='Evaluation')
    p_eval.set_defaults(func=cmd_eval)
    
    # download-model
    p_dl = subparsers.add_parser('download-model', help='Download GGUF model')
    p_dl.add_argument('repo_id', help='HF repo id')
    p_dl.add_argument('filename', help='Filename or pattern')
    p_dl.add_argument('--local-dir', default='models', help='Local dir')
    p_dl.set_defaults(func=cmd_download_model)
    
    # generate
    p_gen = subparsers.add_parser('generate', help='Generate with memory')
    p_gen.add_argument('query', help='Query')
    p_gen.add_argument('--name', default='droid-alpha', help='Droid name')
    p_gen.add_argument('--strategy', default='auto', choices=['auto', 'prompt', 'hidden', 'kv_cache'])
    p_gen.add_argument('--verbose', action='store_true')
    p_gen.set_defaults(func=cmd_generate)
    
    args = parser.parse_args()
    return args.func(args)

if __name__ == '__main__':
    sys.exit(main() or 0)
