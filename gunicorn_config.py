"""Gunicorn configuration to ensure state is seeded in each worker."""
import os
import sys

# Gunicorn config variables
bind = "0.0.0.0:8080"
workers = 2
timeout = 120
worker_class = "sync"
preload_app = False  # Don't preload - let each worker import fresh

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    try:
        # Import here to avoid circular imports
        from app import seed_state
        
        # Get the app from the worker
        app = worker.app
        if app and hasattr(app, 'config'):
            state = app.config.get('dt_state')
            if state:
                # Ensure nodes are present - re-seed if needed
                nodes = state.list_nodes()
                if not nodes:
                    seed_state(state)
                    node_count = len(state.list_nodes())
                    print(f"[Worker {worker.pid}] State seeded with {node_count} nodes", file=sys.stderr, flush=True)
                else:
                    print(f"[Worker {worker.pid}] State already has {len(nodes)} nodes", file=sys.stderr, flush=True)
            else:
                print(f"[Worker {worker.pid}] WARNING: No state found in app.config", file=sys.stderr, flush=True)
        else:
            print(f"[Worker {worker.pid}] WARNING: App or config not found", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[Worker {worker.pid}] ERROR in post_worker_init: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
