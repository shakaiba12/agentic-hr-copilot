import { useState, useCallback, useRef, useEffect } from 'react';
import { ChatMessage, RAGResult, SQLResult, ConversationSummary } from '../types/api';
import {
  sendChatMessage,
  streamChatMessage,
  fetchConversations,
  fetchConversation,
  deleteConversation as apiDeleteConversation,
} from '../services/api';

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<(() => void) | null>(null);

  const refreshConversations = useCallback(async () => {
    try {
      const list = await fetchConversations();
      setConversations(list);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  const startNewChat = useCallback(() => {
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    setMessages([]);
    setCurrentConversationId(null);
    setError(null);
    setLoading(false);
    setStreaming(false);
  }, []);

  const loadConversation = useCallback(
    async (convId: string) => {
      if (loading || streaming) return;
      try {
        setLoading(true);
        setError(null);
        const detail = await fetchConversation(convId);
        setCurrentConversationId(detail.id);

        const loadedMessages: ChatMessage[] = detail.messages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          timestamp: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          source: m.source || undefined,
          category: m.category || undefined,
          allowed: m.metadata?.allowed,
          decision: m.metadata?.decision,
          rag_result: m.metadata?.rag_result,
          sql_result: m.metadata?.sql_result,
        }));

        setMessages(loadedMessages);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setError(`Failed to load conversation: ${msg}`);
      } finally {
        setLoading(false);
      }
    },
    [loading, streaming]
  );

  const deleteConversation = useCallback(
    async (convId: string) => {
      try {
        await apiDeleteConversation(convId);
        setConversations((prev) => prev.filter((c) => c.id !== convId));
        if (currentConversationId === convId) {
          startNewChat();
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setError(`Failed to delete conversation: ${msg}`);
      }
    },
    [currentConversationId, startNewChat]
  );

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

      // Optimistically append user message
      setMessages((prev) => [...prev, userMessage]);
      setLoading(true);

      const convIdToUse = currentConversationId;

      if (useStreamingMode) {
        setStreaming(true);

        const initialAssistantMsg: ChatMessage = {
          id: assistantMsgId,
          role: 'assistant',
          content: '',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isStreaming: true,
        };

        setMessages((prev) => [...prev, initialAssistantMsg]);

        const streamHandle = streamChatMessage(cleanText, convIdToUse, [], {
          onMeta: (meta) => {
            if (meta.conversation_id && typeof meta.conversation_id === 'string') {
              setCurrentConversationId(meta.conversation_id);
            }

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
          onDone: (data) => {
            if (data?.conversation_id && typeof data.conversation_id === 'string') {
              setCurrentConversationId(data.conversation_id);
            }
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
            refreshConversations();
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
        try {
          const res = await sendChatMessage(cleanText, convIdToUse);
          if (res.conversation_id) {
            setCurrentConversationId(res.conversation_id);
          }
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
          refreshConversations();
        } catch (err: unknown) {
          const errMsg = err instanceof Error ? err.message : String(err);
          setError(errMsg);
        } finally {
          setLoading(false);
        }
      }
    },
    [loading, streaming, currentConversationId, refreshConversations]
  );

  return {
    messages,
    currentConversationId,
    conversations,
    loading,
    streaming,
    error,
    sendMessage,
    startNewChat,
    loadConversation,
    deleteConversation,
    refreshConversations,
    stopGeneration,
  };
}
