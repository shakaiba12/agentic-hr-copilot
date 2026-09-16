import { ChatResponse, HealthStatus, PolicyDocument, ConversationSummary, ConversationDetail } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function fetchHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchDocuments(): Promise<PolicyDocument[]> {
  const response = await fetch(`${API_BASE_URL}/api/docs/list`);
  if (!response.ok) {
    throw new Error(`Failed to list documents: ${response.statusText}`);
  }
  const data = await response.json();
  return data.documents || [];
}

export async function fetchConversations(limit: number = 50): Promise<ConversationSummary[]> {
  const response = await fetch(`${API_BASE_URL}/api/conversations?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`Failed to list conversations: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchConversation(id: string): Promise<ConversationDetail> {
  const response = await fetch(`${API_BASE_URL}/api/conversations/${encodeURIComponent(id)}`);
  if (!response.ok) {
    throw new Error(`Failed to get conversation: ${response.statusText}`);
  }
  return response.json();
}

export async function createConversation(title?: string): Promise<ConversationSummary> {
  const response = await fetch(`${API_BASE_URL}/api/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!response.ok) {
    throw new Error(`Failed to create conversation: ${response.statusText}`);
  }
  return response.json();
}

export async function deleteConversation(id: string): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/api/conversations/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  return response.ok;
}

export async function sendChatMessage(
  query: string,
  conversationId?: string | null,
  history: Array<{ role: string; content: string }> = [],
  signal?: AbortSignal
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      conversation_id: conversationId || undefined,
      history: history.length > 0 ? history : undefined,
    }),
    signal,
  });

  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const errJson = await response.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // ignore parse error
    }
    throw new Error(errorDetail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export interface StreamCallbacks {
  onMeta?: (meta: Record<string, unknown>) => void;
  onToken?: (delta: string) => void;
  onDone?: (data: Record<string, unknown>) => void;
  onError?: (error: string) => void;
}

export function streamChatMessage(
  query: string,
  conversationId: string | null | undefined,
  history: Array<{ role: string; content: string }> = [],
  callbacks: StreamCallbacks
): { abort: () => void } {
  const controller = new AbortController();

  (async () => {
    let gotDone = false;
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({
          query,
          conversation_id: conversationId || undefined,
          history: history.length > 0 ? history : undefined,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        let errText = response.statusText;
        try {
          const errData = await response.json();
          if (errData.detail) errText = errData.detail;
        } catch {
          // ignore
        }
        throw new Error(errText || `Stream failed: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('Streaming response body is empty.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          const lines = part.split('\n').filter(Boolean);
          let eventName = 'message';
          let dataStr = '';

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventName = line.slice('event:'.length).trim();
            } else if (line.startsWith('data:')) {
              dataStr += line.slice('data:'.length).trim();
            }
          }

          if (!dataStr) continue;

          let data: Record<string, unknown>;
          try {
            data = JSON.parse(dataStr);
          } catch {
            data = { raw: dataStr };
          }

          if (eventName === 'meta') {
            callbacks.onMeta?.(data);
          } else if (eventName === 'token') {
            callbacks.onToken?.((data.delta as string) || '');
          } else if (eventName === 'done') {
            gotDone = true;
            callbacks.onDone?.(data);
          } else if (eventName === 'error') {
            callbacks.onError?.((data.message as string) || 'Streaming error');
          }
        }
      }

      if (!controller.signal.aborted && !gotDone) {
        callbacks.onDone?.({ status: 'completed' });
      }
    } catch (err: unknown) {
      if (controller.signal.aborted) return;
      const message = err instanceof Error ? err.message : String(err);
      callbacks.onError?.(message);
    }
  })();

  return {
    abort: () => controller.abort(),
  };
}
