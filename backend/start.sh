#!/bin/bash
# Entrypoint for HF Spaces dev mode compatibility.
# Dev mode may spawn the CMD multiple times; kill any prior
# uvicorn instance so the new one can bind the port.
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 0.3
exec uvicorn main:app --host 0.0.0.0 --port 7860
