import React from 'react';
import {
  MessageSquarePlus,
  Trash2,
  Database,
  FileText,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  HelpCircle,
  Building2,
  BrainCircuit,
  User,
  MessagesSquare,
  MessageSquare,
} from 'lucide-react';
import { HealthStatus, PolicyDocument, ConversationSummary } from '../types/api';

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  onNewChat: () => void;
  onClearChat: () => void;
  documents: PolicyDocument[];
  onSelectQuestion: (question: string) => void;
  health: HealthStatus | null;
  messageCount: number;
  loading: boolean;
  conversations?: ConversationSummary[];
  currentConversationId?: string | null;
  onSelectConversation?: (id: string) => void;
  onDeleteConversation?: (id: string) => void;
}

const SQL_EXAMPLES = [
  "How many employees work in Austin?",
  "Which employees were hired after January 1, 2024?",
  "Show employees in the Engineering department",
  "What is the average employee salary by department?",
];

const RAG_EXAMPLES = [
  "What is the parental leave policy?",
  "What percentage of insurance premiums does the company pay?",
  "What is the home office stipend procedure?",
  "What is the policy for code of conduct?",
];

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onToggleCollapse,
  onNewChat,
  onClearChat,
  documents,
  onSelectQuestion,
  health,
  messageCount,
  loading,
  conversations = [],
  currentConversationId,
  onSelectConversation,
  onDeleteConversation,
}) => {
  return (
    <aside
      className={[
        "h-full shrink-0 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-xl",
        "shadow-[0_20px_70px_rgba(0,0,0,0.35)] transition-all duration-300 flex flex-col overflow-hidden",
        collapsed ? "w-[68px]" : "w-[280px] lg:w-[310px]",
      ].join(" ")}
    >
      <div className="h-full min-h-0 overflow-y-auto scroll-on-hover scroll-smooth p-3 md:p-4 flex flex-col justify-between">
        {/* TOP SECTION */}
        <div className="space-y-4">
          {/* Header Branding */}
          <div className="flex items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2.5">
              <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-cyan-300 via-cyan-400 to-violet-500 font-extrabold text-slate-950 shadow-[0_0_20px_rgba(103,232,249,0.35)]">
                PQ
              </div>

              {!collapsed && (
                <div className="min-w-0 pr-1">
                  <h2 className="truncate text-base font-bold tracking-tight text-white">
                    PeopleQuery AI
                  </h2>
                  <p className="truncate text-[11px] text-slate-400">
                    HR Intelligence Copilot
                  </p>
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={onToggleCollapse}
              className="shrink-0 rounded-lg border border-white/10 bg-white/[0.04] p-1.5 text-xs text-slate-300 hover:bg-white/[0.08] hover:text-white transition-colors"
              title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {collapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-3.5 h-3.5" />}
            </button>
          </div>

          {/* Quick Action Buttons */}
          <div className="flex flex-col gap-2 pt-1">
            <button
              type="button"
              onClick={onNewChat}
              disabled={loading}
              className={[
                "flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-300 to-violet-300 font-semibold text-slate-950 transition hover:brightness-110 disabled:opacity-60 shadow-lg shadow-cyan-950/30",
                collapsed ? "h-10 w-10 p-0" : "w-full py-2.5 px-3 text-xs",
              ].join(" ")}
              title="New Conversation"
            >
              <MessageSquarePlus className="w-4 h-4 shrink-0" />
              {!collapsed && <span>New Conversation</span>}
            </button>

            {!collapsed && messageCount > 0 && (
              <button
                type="button"
                onClick={onClearChat}
                disabled={loading}
                className="flex items-center justify-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.03] py-2 px-3 text-xs text-slate-300 hover:bg-white/[0.08] hover:text-red-300 transition disabled:opacity-50"
              >
                <Trash2 className="w-3.5 h-3.5 shrink-0" />
                <span>Clear History</span>
              </button>
            )}
          </div>

          {!collapsed && (
            <>
              {/* Recent Conversations (Chat Memory) */}
              {conversations.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                      <MessagesSquare className="w-3 h-3 text-cyan-400" />
                      Recent Chats
                    </p>
                    <span className="text-[10px] text-slate-500 font-mono">
                      {conversations.length} saved
                    </span>
                  </div>

                  <div className="space-y-1 max-h-40 overflow-y-auto scroll-on-hover pr-1">
                    {conversations.map((conv) => {
                      const isActive = currentConversationId === conv.id;
                      return (
                        <div
                          key={conv.id}
                          className={[
                            "group flex items-center justify-between gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs transition cursor-pointer",
                            isActive
                              ? "border-cyan-400/50 bg-cyan-950/40 text-cyan-200"
                              : "border-white/5 bg-white/[0.02] text-slate-300 hover:border-white/20 hover:bg-white/[0.05]",
                          ].join(" ")}
                          onClick={() => onSelectConversation?.(conv.id)}
                        >
                          <div className="flex items-center gap-1.5 min-w-0 flex-1">
                            <MessageSquare className="w-3 h-3 shrink-0 text-cyan-300/70" />
                            <span className="truncate text-[11px] font-medium">{conv.title}</span>
                          </div>

                          <div className="flex items-center gap-1 shrink-0">
                            {conv.message_count > 0 && (
                              <span className="text-[9px] px-1.5 py-0.2 rounded-full bg-white/[0.06] text-slate-400 font-mono">
                                {conv.message_count}
                              </span>
                            )}
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                onDeleteConversation?.(conv.id);
                              }}
                              className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-400 hover:text-red-400 hover:bg-white/10 transition"
                              title="Delete conversation"
                            >
                              <Trash2 className="w-2.5 h-2.5" />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* System Capabilities */}
              <div className="rounded-xl border border-white/10 bg-black/20 p-3 space-y-2">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Sparkles className="w-3 h-3 text-cyan-400" />
                  Capabilities
                </p>
                <div className="space-y-1.5 text-xs text-slate-300">
                  <div className="flex items-center gap-2">
                    <Database className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                    <span className="truncate">SQL Data & Headcount Queries</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Building2 className="w-3.5 h-3.5 text-violet-400 shrink-0" />
                    <span className="truncate">HR Policies & Benefit Documents</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <BrainCircuit className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span className="truncate">Persistent Short-Term Memory</span>
                  </div>
                </div>
              </div>

              {/* Indexed Policy Knowledge Base */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <FileText className="w-3 h-3 text-cyan-400" />
                    Knowledge Base
                  </p>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {documents.length} policies
                  </span>
                </div>

                <div className="space-y-1 max-h-32 overflow-y-auto scroll-on-hover pr-1">
                  {documents.length === 0 ? (
                    <p className="text-xs text-slate-500 italic p-1">Loading documents...</p>
                  ) : (
                    documents.map((doc, idx) => (
                      <div
                        key={idx}
                        className="flex items-center gap-2 rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-1.5 text-xs text-slate-300 hover:border-cyan-400/30 hover:bg-white/[0.05] transition group"
                        title={doc.filename}
                      >
                        <FileText className="w-3.5 h-3.5 text-cyan-300/60 group-hover:text-cyan-300 shrink-0" />
                        <span className="truncate text-[11px]">{doc.title}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Sample Queries */}
              <div className="space-y-2">
                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <HelpCircle className="w-3 h-3 text-violet-400" />
                  Sample Questions
                </p>

                <div className="space-y-1.5 max-h-40 overflow-y-auto scroll-on-hover pr-1">
                  <p className="text-[10px] font-semibold text-cyan-300/80 uppercase">SQL / Data</p>
                  {SQL_EXAMPLES.slice(0, 2).map((q, idx) => (
                    <button
                      key={`sql-${idx}`}
                      type="button"
                      onClick={() => onSelectQuestion(q)}
                      disabled={loading}
                      className="w-full text-left rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-1.5 text-[11px] text-slate-300 hover:border-cyan-400/40 hover:bg-cyan-950/30 hover:text-cyan-200 transition"
                    >
                      {q}
                    </button>
                  ))}

                  <p className="text-[10px] font-semibold text-violet-300/80 uppercase pt-1">RAG / Policy</p>
                  {RAG_EXAMPLES.slice(0, 2).map((q, idx) => (
                    <button
                      key={`rag-${idx}`}
                      type="button"
                      onClick={() => onSelectQuestion(q)}
                      disabled={loading}
                      className="w-full text-left rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-1.5 text-[11px] text-slate-300 hover:border-violet-400/40 hover:bg-violet-950/30 hover:text-violet-200 transition"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

        {/* BOTTOM SECTION - User / Environment Status */}
        <div className="pt-4 border-t border-white/10 mt-4 space-y-2">
          {collapsed ? (
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-white/[0.05] border border-white/10 text-slate-300" title="HR Copilot">
              <User className="w-4 h-4" />
            </div>
          ) : (
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-2.5 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 min-w-0">
                <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-slate-800 to-slate-900 border border-white/10 text-cyan-300 font-bold text-xs">
                  HR
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-slate-200 truncate">HR Operations</p>
                  <p className="text-[10px] text-slate-500 truncate">
                    {health ? `${health.environment.toUpperCase()} • ${health.provider}` : 'Enterprise Copilot'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
