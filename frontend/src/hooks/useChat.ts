import { useState, useCallback, useRef } from 'react';
import { ChatMessage, RAGResult, SQLResult } from '../types/api';
import { sendChatMessage, streamChatMessage } from '../services/api';

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<(() => void) | null>(null);

  const clearChat = useCallback(() => {
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    setMessages([]);
    setError(null);
    setLoading(false);
    setStreaming(false);
  }, []);

  const stopGeneration = useCallback(() => {
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    setLoading(false);
    setStreaming(false);
  }, []);

  const sendMessage = useCallback(
    async (queryText: string, useStreamingMode: boolean = true) => {
      const cleanText = queryText.trim();
      if (!cleanText || loading || streaming) return;

      setError(null);
      const userMsgId = `user-${Date.now()}`;
      const assistantMsgId = `assistant-${Date.now()}`;

      const userMessage: ChatMessage = {
        id: userMsgId,
        role: 'user',
        content: cleanText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      const historyPayload = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Optimistically append user message
      setMessages((prev) => [...prev, userMessage]);
      setLoading(true);

      if (useStreamingMode) {
        setStreaming(true);

        // Placeholder assistant message
        const initialAssistantMsg: ChatMessage = {
          id: assistantMsgId,
          role: 'assistant',
          content: '',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isStreaming: true,
        };

        setMessages((prev) => [...prev, initialAssistantMsg]);

        const streamHandle = streamChatMessage(cleanText, historyPayload, {
          onMeta: (meta) => {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id !== assistantMsgId) return msg;
                return {
                  ...msg,
                  source: (meta.source as string) || 'master',
                  allowed: meta.allowed !== false,
                  decision: {
                    category: (meta.category as string) || 'GENERAL',
                    target: (meta.target as string) || null,
                    allowed: meta.allowed !== false,
                    confidence: typeof meta.confidence === 'number' ? meta.confidence : 1.0,
                    reason: (meta.reason as string) || '',
                  },
                  rag_result: meta.sources
                    ? ({
                        success: true,
                        sources: (meta.sources as any[]) || [],
                        chunks_count: (meta.chunks_count as number) || 0,
                        grounded: meta.grounded !== false,
                      } as RAGResult)
                    : null,
                  sql_result: meta.rows
                    ? ({
                        success: true,
                        generated_sql: (meta.generated_sql as string) || null,
                        rows: (meta.rows as Array<Record<string, unknown>>) || [],
                        row_count: (meta.row_count as number) || 0,
                        message: '',
                        was_truncated: meta.was_truncated === true,
                      } as SQLResult)
                    : null,
                };
              })
            );
          },
          onToken: (token) => {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id !== assistantMsgId) return msg;
                return {
                  ...msg,
                  content: msg.content + token,
                };
              })
            );
          },
          onDone: () => {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id !== assistantMsgId) return msg;
                return {
                  ...msg,
                  isStreaming: false,
                };
              })
            );
            setLoading(false);
            setStreaming(false);
            abortRef.current = null;
          },
          onError: (errMsg) => {
            setError(errMsg);
            setLoading(false);
            setStreaming(false);
            abortRef.current = null;
          },
        });

        abortRef.current = streamHandle.abort;
      } else {
        // Non-streaming fallback
        try {
          const res = await sendChatMessage(cleanText, historyPayload);
          const assistantMsg: ChatMessage = {
            id: assistantMsgId,
            role: 'assistant',
            content: res.response,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            source: res.source,
            allowed: res.allowed,
            decision: res.decision,
            rag_result: res.rag_result,
            sql_result: res.sql_result,
            error: res.error,
            isStreaming: false,
          };
          setMessages((prev) => [...prev, assistantMsg]);
        } catch (err: unknown) {
          const errMsg = err instanceof Error ? err.message : String(err);
          setError(errMsg);
        } finally {
          setLoading(false);
        }
      }
    },
    [loading, streaming, messages]
  );

  return {
    messages,
    loading,
    streaming,
    error,
    sendMessage,
    clearChat,
    stopGeneration,
  };
}
