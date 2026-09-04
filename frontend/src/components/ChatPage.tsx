import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Send,
  Square,
  Copy,
  Check,
  Sparkles,
  Database,
  FileText,
  AlertTriangle,
  RotateCcw,
} from 'lucide-react';
import { ChatMessage, HealthStatus, PolicyDocument } from '../types/api';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { DataTable } from './DataTable';
import { KpiCard } from './KpiCard';
import { SqlViewer } from './SqlViewer';
import { SourceCard } from './SourceCard';

interface ChatPageProps {
  messages: ChatMessage[];
  loading: boolean;
  streaming: boolean;
  error: string | null;
  health: HealthStatus | null;
  documents: PolicyDocument[];
  onSendMessage: (query: string) => void;
  onClearChat: () => void;
  onStopGeneration: () => void;
}

const SAMPLE_SQL_CARDS = [
  {
    title: "Employees in Austin",
    query: "How many employees work in Austin?",
    type: "SQL Count",
  },
  {
    title: "Recent Hires",
    query: "Which employees were hired after January 1, 2024?",
    type: "SQL Records",
  },
  {
    title: "Engineering Team",
    query: "Show employees in the Engineering department",
    type: "SQL Department",
  },
];

const SAMPLE_RAG_CARDS = [
  {
    title: "Parental Leave Policy",
    query: "What is the parental leave policy?",
    type: "HR Policy",
  },
  {
    title: "Insurance Coverage",
    query: "What percentage of insurance premiums does the company pay?",
    type: "Benefits",
  },
  {
    title: "Equipment & Stipends",
    query: "What happens to company equipment when an employee leaves?",
    type: "Hardware",
  },
];

export const ChatPage: React.FC<ChatPageProps> = ({
  messages,
  loading,
  streaming,
  error,
  health,
  documents,
  onSendMessage,
  onClearChat,
  onStopGeneration,
}) => {
  const [inputQuery, setInputQuery] = useState('');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll on new messages or stream chunks
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: streaming ? 'auto' : 'smooth',
      block: 'end',
    });
  }, [messages, streaming]);

  // Focus input when streaming ends
  useEffect(() => {
    if (!streaming && !loading) {
      textareaRef.current?.focus();
    }
  }, [streaming, loading]);

  const handleCopyText = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const clean = inputQuery.trim();
    if (!clean || loading || streaming) return;
    onSendMessage(clean);
    setInputQuery('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputQuery(e.target.value);
    // Auto-expand textarea height
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`;
  };

  const handleCardClick = (query: string) => {
    if (loading || streaming) return;
    onSendMessage(query);
  };

  // Helper to decide if an SQL result is a single scalar metric suitable for a KPI card
  const isKpiCandidate = (rows?: Array<Record<string, unknown>>) => {
    if (!rows || rows.length !== 1) return false;
    const row = rows[0];
    const keys = Object.keys(row);
    return keys.length <= 3 && keys.some((k) => /count|total|avg|sum|max|min|percentage/i.test(k) || typeof row[k] === 'number');
  };

  return (
    <div className="h-screen w-screen overflow-hidden bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-[#132237] via-[#070b12] to-[#05070b] text-slate-100 flex flex-col">
      <div className="h-full p-2.5 sm:p-3 md:p-4">
        <div className="flex h-full gap-2.5 md:gap-3">
          {/* SIDEBAR */}
          <Sidebar
            collapsed={sidebarCollapsed}
            onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
            onNewChat={onClearChat}
            onClearChat={onClearChat}
            documents={documents}
            onSelectQuestion={handleCardClick}
            health={health}
            messageCount={messages.length}
            loading={loading || streaming}
          />

          {/* MAIN CHAT AREA */}
          <main className="min-w-0 flex-1 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-xl shadow-[0_20px_80px_rgba(0,0,0,0.4)] flex flex-col overflow-hidden">
            {/* TOP HEADER */}
            <Header
              health={health}
              onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
              sidebarCollapsed={sidebarCollapsed}
            />

            {/* CONVERSATION AREA */}
            <div className="min-h-0 flex-1 overflow-y-auto scroll-on-hover px-3 py-4 md:px-6">
              {messages.length === 0 ? (
                /* EMPTY / WELCOME SCREEN */
                <div className="mx-auto my-auto max-w-3xl py-6 md:py-10 text-center flex flex-col items-center justify-center">
                  <div className="grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-cyan-400/20 via-violet-400/20 to-cyan-400/10 border border-cyan-400/30 text-cyan-300 shadow-[0_0_30px_rgba(103,232,249,0.2)] mb-4">
                    <Sparkles className="w-8 h-8" />
                  </div>

                  <h2 className="text-xl sm:text-2xl md:text-3xl font-bold tracking-tight text-white">
                    How can I help with your workforce data?
                  </h2>
                  <p className="mt-2 text-xs sm:text-sm text-slate-400 max-w-lg">
                    Ask questions across operational employee databases or query company HR policies and benefits.
                  </p>

                  {/* Suggestion Cards Container */}
                  <div className="mt-8 w-full space-y-4 text-left">
                    {/* SQL Queries Section */}
                    <div>
                      <div className="flex items-center gap-2 mb-2 px-1">
                        <Database className="w-3.5 h-3.5 text-cyan-400" />
                        <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-300/90">
                          Workforce Data & SQL Analytics
                        </span>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                        {SAMPLE_SQL_CARDS.map((card, idx) => (
                          <button
                            key={`sql-card-${idx}`}
                            type="button"
                            onClick={() => handleCardClick(card.query)}
                            disabled={loading || streaming}
                            className="group relative flex flex-col justify-between rounded-xl border border-white/10 bg-white/[0.02] p-3 text-left transition-all duration-200 hover:border-cyan-400/40 hover:bg-cyan-950/20 hover:shadow-lg hover:shadow-cyan-950/30 disabled:opacity-50"
                          >
                            <span className="text-xs font-medium text-slate-200 group-hover:text-cyan-200 transition-colors">
                              "{card.query}"
                            </span>
                            <span className="mt-3 inline-block text-[10px] font-semibold uppercase tracking-wider text-cyan-400/70">
                              {card.type}
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* RAG Policy Section */}
                    <div className="pt-2">
                      <div className="flex items-center gap-2 mb-2 px-1">
                        <FileText className="w-3.5 h-3.5 text-violet-400" />
                        <span className="text-[11px] font-semibold uppercase tracking-wider text-violet-300/90">
                          HR Policies & Benefit Documents
                        </span>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                        {SAMPLE_RAG_CARDS.map((card, idx) => (
                          <button
                            key={`rag-card-${idx}`}
                            type="button"
                            onClick={() => handleCardClick(card.query)}
                            disabled={loading || streaming}
                            className="group relative flex flex-col justify-between rounded-xl border border-white/10 bg-white/[0.02] p-3 text-left transition-all duration-200 hover:border-violet-400/40 hover:bg-violet-950/20 hover:shadow-lg hover:shadow-violet-950/30 disabled:opacity-50"
                          >
                            <span className="text-xs font-medium text-slate-200 group-hover:text-violet-200 transition-colors">
                              "{card.query}"
                            </span>
                            <span className="mt-3 inline-block text-[10px] font-semibold uppercase tracking-wider text-violet-400/70">
                              {card.type}
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* CHAT MESSAGE BUBBLES */
                <div className="space-y-5 pb-4 max-w-4xl mx-auto">
                  {messages.map((m, idx) => {
                    const isUser = m.role === 'user';
                    const isLast = idx === messages.length - 1;
                    const showCursor = streaming && isLast && !isUser;

                    return (
                      <div
                        key={m.id || idx}
                        className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={[
                            "relative max-w-[92%] sm:max-w-[85%] rounded-2xl p-4 shadow-xl transition-all",
                            isUser
                              ? "border border-cyan-400/30 bg-gradient-to-br from-cyan-950/50 via-slate-900/80 to-violet-950/40 text-white"
                              : "border border-white/10 bg-[#090d16]/80 text-slate-100",
                          ].join(" ")}
                        >
                          {/* Message Header Bar */}
                          <div className="mb-2 flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                              <span
                                className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded ${
                                  isUser
                                    ? 'bg-cyan-400/20 text-cyan-200 border border-cyan-400/30'
                                    : m.source === 'sql'
                                    ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-400/40'
                                    : m.source === 'rag'
                                    ? 'bg-violet-950/80 text-violet-300 border border-violet-400/40'
                                    : 'bg-white/[0.05] text-slate-300 border border-white/10'
                                }`}
                              >
                                {isUser
                                  ? 'You'
                                  : m.source === 'sql'
                                  ? 'DATA ANALYSIS'
                                  : m.source === 'rag'
                                  ? 'POLICY KNOWLEDGE'
                                  : 'HR ASSISTANT'}
                              </span>

                              {/* Target / Allowed Badge */}
                              {!isUser && m.decision && (
                                <span className="text-[10px] text-slate-500 font-mono hidden sm:inline">
                                  {m.decision.category}
                                </span>
                              )}
                            </div>

                            <span className="text-[10px] text-slate-500">{m.timestamp}</span>
                          </div>

                          {/* Message Body */}
                          <div className="prose-dark text-xs sm:text-sm leading-relaxed overflow-hidden">
                            {m.content ? (
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {m.content}
                              </ReactMarkdown>
                            ) : isUser ? null : (
                              <div className="flex items-center gap-2 py-1 text-slate-400 text-xs italic">
                                <span className="inline-block h-2 w-2 rounded-full bg-cyan-400 animate-ping" />
                                Processing with Enterprise Orchestrator...
                              </div>
                            )}
                          </div>

                          {/* Pulsing cursor during streaming */}
                          {showCursor && (
                            <span className="ml-1 inline-block h-3.5 w-1 animate-pulse rounded bg-cyan-300 align-middle" />
                          )}

                          {/* Structured SQL Results: KPI or Table */}
                          {!isUser && m.sql_result && m.sql_result.rows && m.sql_result.rows.length > 0 && (
                            <div className="mt-3">
                              {isKpiCandidate(m.sql_result.rows) ? (
                                <KpiCard data={m.sql_result.rows[0]} />
                              ) : (
                                <DataTable
                                  rows={m.sql_result.rows}
                                  rowCount={m.sql_result.row_count}
                                  wasTruncated={m.sql_result.was_truncated}
                                />
                              )}
                            </div>
                          )}

                          {/* Generated SQL Collapsible */}
                          {!isUser && m.sql_result?.generated_sql && (
                            <SqlViewer sql={m.sql_result.generated_sql} />
                          )}

                          {/* RAG Citations Collapsible */}
                          {!isUser && m.rag_result?.sources && m.rag_result.sources.length > 0 && (
                            <SourceCard sources={m.rag_result.sources} />
                          )}

                          {/* Action Toolbar */}
                          {!isUser && m.content && (
                            <div className="mt-3 flex items-center justify-end gap-2 border-t border-white/5 pt-2">
                              <button
                                type="button"
                                onClick={() => handleCopyText(m.content, m.id)}
                                className="flex items-center gap-1 rounded-md border border-white/10 bg-white/[0.04] px-2.5 py-1 text-[11px] text-slate-300 hover:bg-white/[0.08] transition"
                              >
                                {copiedId === m.id ? (
                                  <>
                                    <Check className="w-3 h-3 text-emerald-400" />
                                    <span className="text-emerald-400">Copied</span>
                                  </>
                                ) : (
                                  <>
                                    <Copy className="w-3 h-3 text-slate-400" />
                                    <span>Copy Response</span>
                                  </>
                                )}
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* INPUT COMPOSER */}
            <div className="shrink-0 border-t border-white/10 bg-[#060a12]/95 px-3 py-3 md:px-6 md:py-4 backdrop-blur-lg">
              {error && (
                <div className="mx-auto mb-3 max-w-4xl flex items-center justify-between rounded-xl border border-red-500/30 bg-red-950/30 p-2.5 text-xs text-red-300">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                    <span>{error}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleSubmit()}
                    className="flex items-center gap-1 rounded bg-red-500/20 px-2 py-0.5 text-[11px] hover:bg-red-500/30 transition"
                  >
                    <RotateCcw className="w-3 h-3" />
                    Retry
                  </button>
                </div>
              )}

              <form onSubmit={handleSubmit} className="mx-auto max-w-4xl flex items-end gap-2">
                <div className="relative flex-1 rounded-xl border border-white/15 bg-black/40 focus-within:border-cyan-400/50 focus-within:ring-1 focus-within:ring-cyan-400/30 transition-all shadow-inner">
                  <textarea
                    ref={textareaRef}
                    rows={1}
                    value={inputQuery}
                    onChange={handleTextareaInput}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about your employees, workforce data, or company policies..."
                    disabled={loading || streaming}
                    className="w-full resize-none bg-transparent px-4 py-3 text-xs sm:text-sm text-slate-100 placeholder:text-slate-500 outline-none scroll-on-hover disabled:opacity-50"
                  />
                </div>

                <button
                  type={streaming ? 'button' : 'submit'}
                  onClick={streaming ? onStopGeneration : undefined}
                  disabled={!streaming && (!inputQuery.trim() || loading)}
                  title={streaming ? 'Stop Generation' : 'Send Message'}
                  className={[
                    "grid h-11 w-11 shrink-0 place-items-center rounded-xl font-bold text-slate-950 transition-all shadow-lg",
                    streaming
                      ? "bg-amber-400 hover:bg-amber-300 shadow-amber-950/40 cursor-pointer"
                      : "bg-gradient-to-r from-cyan-300 via-cyan-400 to-violet-400 hover:brightness-110 shadow-cyan-950/40 disabled:cursor-not-allowed disabled:opacity-40",
                  ].join(" ")}
                >
                  {streaming ? (
                    <Square className="w-4 h-4 fill-slate-950" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </button>
              </form>

              <div className="mx-auto max-w-4xl mt-2 flex items-center justify-between text-[10px] text-slate-500 px-1">
                <span>Shift + Enter for new line • Enter to submit</span>
                <span className="hidden sm:inline">Protected by Deterministic Master Guardrails</span>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};
