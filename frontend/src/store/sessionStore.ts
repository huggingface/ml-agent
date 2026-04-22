import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SessionMeta } from '@/types/agent';
import { deleteMessages, moveMessages } from '@/lib/chat-message-store';
import { moveBackendMessages, deleteBackendMessages } from '@/lib/backend-message-store';

interface SessionStore {
  sessions: SessionMeta[];
  activeSessionId: string | null;

  // Actions
  createSession: (id: string) => void;
  deleteSession: (id: string) => void;
  switchSession: (id: string) => void;
  setSessionActive: (id: string, isActive: boolean) => void;
  updateSessionTitle: (id: string, title: string) => void;
  setNeedsAttention: (id: string, needs: boolean) => void;
  /** Mark a session as expired (backend no longer has it). The UI shows a
   *  recovery banner and disables input. */
  markExpired: (id: string) => void;
  /** Clear the expired flag (used after restore-with-summary succeeds). */
  clearExpired: (id: string) => void;
  /** Atomically swap a session's id in the list + both localStorage caches.
   *  Used when we rehydrate an expired session into a freshly-created backend
   *  session — preserves title, timestamps, and messages. */
  renameSession: (oldId: string, newId: string) => void;
}

export const useSessionStore = create<SessionStore>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,

      createSession: (id: string) => {
        const newSession: SessionMeta = {
          id,
          title: `Chat ${get().sessions.length + 1}`,
          createdAt: new Date().toISOString(),
          isActive: true,
          needsAttention: false,
        };
        set((state) => ({
          sessions: [...state.sessions, newSession],
          activeSessionId: id,
        }));
      },

      deleteSession: (id: string) => {
        deleteMessages(id);
        deleteBackendMessages(id);
        set((state) => {
          const newSessions = state.sessions.filter((s) => s.id !== id);
          const newActiveId =
            state.activeSessionId === id
              ? newSessions[newSessions.length - 1]?.id || null
              : state.activeSessionId;
          return {
            sessions: newSessions,
            activeSessionId: newActiveId,
          };
        });
      },

      markExpired: (id: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) => (s.id === id ? { ...s, expired: true } : s)),
        }));
      },

      clearExpired: (id: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, expired: false } : s,
          ),
        }));
      },

      renameSession: (oldId: string, newId: string) => {
        if (oldId === newId) return;
        moveMessages(oldId, newId);
        moveBackendMessages(oldId, newId);
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === oldId ? { ...s, id: newId, expired: false } : s,
          ),
          activeSessionId: state.activeSessionId === oldId ? newId : state.activeSessionId,
        }));
      },

      switchSession: (id: string) => {
        set((state) => ({
          activeSessionId: id,
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, needsAttention: false } : s
          ),
        }));
      },

      setSessionActive: (id: string, isActive: boolean) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, isActive } : s
          ),
        }));
      },

      updateSessionTitle: (id: string, title: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, title } : s
          ),
        }));
      },

      setNeedsAttention: (id: string, needs: boolean) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, needsAttention: needs } : s
          ),
        }));
      },
    }),
    {
      name: 'hf-agent-sessions',
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
      }),
    }
  )
);
