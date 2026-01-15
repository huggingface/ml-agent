"""Agent API routes - WebSocket and REST endpoints."""

import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from models import (
    ApprovalRequest,
    HealthResponse,
    SessionInfo,
    SessionResponse,
    SubmitRequest,
)
from session_manager import session_manager
from websocket import manager as ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok", active_sessions=session_manager.active_session_count
    )


@router.post("/session", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    """Create a new agent session."""
    session_id = await session_manager.create_session()
    return SessionResponse(session_id=session_id, ready=True)


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    """Get session information."""
    info = session_manager.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(**info)


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions() -> list[SessionInfo]:
    """List all sessions."""
    sessions = session_manager.list_sessions()
    return [SessionInfo(**s) for s in sessions]


@router.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session."""
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/submit")
async def submit_input(request: SubmitRequest) -> dict:
    """Submit user input to a session."""
    success = await session_manager.submit_user_input(request.session_id, request.text)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/approve")
async def submit_approval(request: ApprovalRequest) -> dict:
    """Submit tool approvals to a session."""
    approvals = [
        {
            "tool_call_id": a.tool_call_id,
            "approved": a.approved,
            "feedback": a.feedback,
        }
        for a in request.approvals
    ]
    success = await session_manager.submit_approval(request.session_id, approvals)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/interrupt/{session_id}")
async def interrupt_session(session_id: str) -> dict:
    """Interrupt the current operation in a session."""
    success = await session_manager.interrupt(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "interrupted", "session_id": session_id}


@router.post("/undo/{session_id}")
async def undo_session(session_id: str) -> dict:
    """Undo the last turn in a session."""
    success = await session_manager.undo(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "undo_requested", "session_id": session_id}


@router.post("/compact/{session_id}")
async def compact_session(session_id: str) -> dict:
    """Compact the context in a session."""
    success = await session_manager.compact(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "compact_requested", "session_id": session_id}


@router.post("/shutdown/{session_id}")
async def shutdown_session(session_id: str) -> dict:
    """Shutdown a session."""
    success = await session_manager.shutdown_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "shutdown_requested", "session_id": session_id}


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time events."""
    # Verify session exists
    info = session_manager.get_session_info(session_id)
    if not info:
        await websocket.close(code=4004, reason="Session not found")
        return

    await ws_manager.connect(websocket, session_id)

    try:
        while True:
            # Keep connection alive, handle ping/pong
            data = await websocket.receive_json()

            # Handle client messages (e.g., ping)
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        ws_manager.disconnect(session_id)
