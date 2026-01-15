"""Session manager for handling multiple concurrent agent sessions."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.config import load_config
from agent.core.agent_loop import process_submission
from agent.core.session import Event, Operation, OpType, Session, Submission
from agent.core.tools import ToolRouter
from websocket import manager as ws_manager

logger = logging.getLogger(__name__)


@dataclass
class AgentSession:
    """Wrapper for an agent session with its associated resources."""

    session_id: str
    session: Session
    tool_router: ToolRouter
    submission_queue: asyncio.Queue
    task: asyncio.Task | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


class SessionManager:
    """Manages multiple concurrent agent sessions."""

    def __init__(self, config_path: str = "configs/main_agent_config.json") -> None:
        self.config = load_config(config_path)
        self.sessions: dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> str:
        """Create a new agent session and return its ID."""
        session_id = str(uuid.uuid4())

        # Create queues for this session
        submission_queue: asyncio.Queue = asyncio.Queue()
        event_queue: asyncio.Queue = asyncio.Queue()

        # Create tool router
        tool_router = ToolRouter(config=self.config)

        # Create the agent session
        session = Session(event_queue, config=self.config, tool_router=tool_router)

        # Create wrapper
        agent_session = AgentSession(
            session_id=session_id,
            session=session,
            tool_router=tool_router,
            submission_queue=submission_queue,
        )

        async with self._lock:
            self.sessions[session_id] = agent_session

        # Start the agent loop task
        task = asyncio.create_task(
            self._run_session(session_id, submission_queue, event_queue, tool_router)
        )
        agent_session.task = task

        logger.info(f"Created session {session_id}")
        return session_id

    async def _run_session(
        self,
        session_id: str,
        submission_queue: asyncio.Queue,
        event_queue: asyncio.Queue,
        tool_router: ToolRouter,
    ) -> None:
        """Run the agent loop for a session and forward events to WebSocket."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            logger.error(f"Session {session_id} not found")
            return

        session = agent_session.session

        # Start event forwarder task
        event_forwarder = asyncio.create_task(
            self._forward_events(session_id, event_queue)
        )

        try:
            async with tool_router:
                # Send ready event
                await session.send_event(
                    Event(event_type="ready", data={"message": "Agent initialized"})
                )

                while session.is_running:
                    try:
                        # Wait for submission with timeout to allow checking is_running
                        submission = await asyncio.wait_for(
                            submission_queue.get(), timeout=1.0
                        )
                        should_continue = await process_submission(session, submission)
                        if not should_continue:
                            break
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        logger.info(f"Session {session_id} cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Error in session {session_id}: {e}")
                        await session.send_event(
                            Event(event_type="error", data={"error": str(e)})
                        )

        finally:
            event_forwarder.cancel()
            try:
                await event_forwarder
            except asyncio.CancelledError:
                pass

            async with self._lock:
                if session_id in self.sessions:
                    self.sessions[session_id].is_active = False

            logger.info(f"Session {session_id} ended")

    async def _forward_events(
        self, session_id: str, event_queue: asyncio.Queue
    ) -> None:
        """Forward events from the agent to the WebSocket."""
        while True:
            try:
                event: Event = await event_queue.get()
                await ws_manager.send_event(session_id, event.event_type, event.data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error forwarding event for {session_id}: {e}")

    async def submit(self, session_id: str, operation: Operation) -> bool:
        """Submit an operation to a session."""
        async with self._lock:
            agent_session = self.sessions.get(session_id)

        if not agent_session or not agent_session.is_active:
            logger.warning(f"Session {session_id} not found or inactive")
            return False

        submission = Submission(id=f"sub_{uuid.uuid4().hex[:8]}", operation=operation)
        await agent_session.submission_queue.put(submission)
        return True

    async def submit_user_input(self, session_id: str, text: str) -> bool:
        """Submit user input to a session."""
        operation = Operation(op_type=OpType.USER_INPUT, data={"text": text})
        return await self.submit(session_id, operation)

    async def submit_approval(
        self, session_id: str, approvals: list[dict[str, Any]]
    ) -> bool:
        """Submit tool approvals to a session."""
        operation = Operation(
            op_type=OpType.EXEC_APPROVAL, data={"approvals": approvals}
        )
        return await self.submit(session_id, operation)

    async def interrupt(self, session_id: str) -> bool:
        """Interrupt a session."""
        operation = Operation(op_type=OpType.INTERRUPT)
        return await self.submit(session_id, operation)

    async def undo(self, session_id: str) -> bool:
        """Undo last turn in a session."""
        operation = Operation(op_type=OpType.UNDO)
        return await self.submit(session_id, operation)

    async def compact(self, session_id: str) -> bool:
        """Compact context in a session."""
        operation = Operation(op_type=OpType.COMPACT)
        return await self.submit(session_id, operation)

    async def shutdown_session(self, session_id: str) -> bool:
        """Shutdown a specific session."""
        operation = Operation(op_type=OpType.SHUTDOWN)
        success = await self.submit(session_id, operation)

        if success:
            async with self._lock:
                agent_session = self.sessions.get(session_id)
                if agent_session and agent_session.task:
                    # Wait for task to complete
                    try:
                        await asyncio.wait_for(agent_session.task, timeout=5.0)
                    except asyncio.TimeoutError:
                        agent_session.task.cancel()

        return success

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely."""
        async with self._lock:
            agent_session = self.sessions.pop(session_id, None)

        if not agent_session:
            return False

        # Cancel the task if running
        if agent_session.task and not agent_session.task.done():
            agent_session.task.cancel()
            try:
                await agent_session.task
            except asyncio.CancelledError:
                pass

        return True

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get information about a session."""
        agent_session = self.sessions.get(session_id)
        if not agent_session:
            return None

        return {
            "session_id": session_id,
            "created_at": agent_session.created_at.isoformat(),
            "is_active": agent_session.is_active,
            "message_count": len(agent_session.session.context_manager.messages),
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions."""
        return [
            self.get_session_info(sid)
            for sid in self.sessions
            if self.get_session_info(sid)
        ]

    @property
    def active_session_count(self) -> int:
        """Get count of active sessions."""
        return sum(1 for s in self.sessions.values() if s.is_active)


# Global session manager instance
session_manager = SessionManager()
