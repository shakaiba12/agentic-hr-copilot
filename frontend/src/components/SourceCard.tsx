import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronRight, FileText, Bookmark, Copy, Check } from 'lucide-react';
import { RAGSource } from '../types/api';

interface SourceCardProps {
  sources: RAGSource[];
}

export const SourceCard: React.FC<SourceCardProps> = ({ sources }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  if (!sources || sources.length === 0) return null;

  const handleCopy = (text: string, key: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1500);
  };

  return (
    <div className="my-2 overflow-hidden rounded-xl border border-white/10 bg-black/20 backdrop-blur-sm transition-all duration-200">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between px-3.5 py-2.5 text-left text-xs font-medium text-slate-300 hover:bg-white/[0.04] transition-colors"
      >
        <div className="flex items-center gap-2">
          <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
          <span>Verified Policy Sources ({sources.length})</span>
        </div>
        <div className="flex items-center gap-2">
          {isOpen ? (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </button>

      {isOpen && (
        <div className="border-t border-white/10 p-3 space-y-2 bg-black/40">
          {sources.map((src, idx) => {
            const docName = String(src.document_id || src.document_name || 'Policy Document');
            const section = src.heading_path || src.section || src.section_title || src.title;
            const chunkId = src.chunk_id;
            const snippet = src.text || src.content || '';
            const keyId = `source-${idx}`;

            return (
              <div
                key={idx}
                className="rounded-lg border border-white/10 bg-white/[0.02] p-3 text-xs space-y-1.5 hover:border-cyan-400/20 transition-all"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 text-cyan-200 font-medium truncate">
                    <FileText className="w-3.5 h-3.5 shrink-0 text-cyan-400" />
                    <span className="truncate">{docName}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {chunkId && (
                      <span className="shrink-0 px-1.5 py-0.5 rounded text-[10px] bg-cyan-950/40 border border-cyan-400/20 text-cyan-300 font-mono">
                        {String(chunkId)}
                      </span>
                    )}
                    {src.page_number !== undefined && src.page_number !== null && (
                      <span className="shrink-0 px-2 py-0.5 rounded text-[10px] bg-white/[0.05] border border-white/10 text-slate-400">
                        Page {String(src.page_number)}
                      </span>
                    )}
                  </div>
                </div>

                {section ? (
                  <div className="flex items-center gap-1.5 text-slate-400 text-[11px]">
                    <Bookmark className="w-3 h-3 shrink-0 text-violet-400" />
                    <span className="truncate">{String(section)}</span>
                  </div>
                ) : null}

                {snippet && (
                  <div className="relative mt-2 pt-2 border-t border-white/5">
                    <p className="text-slate-300 text-[11px] leading-relaxed line-clamp-3 hover:line-clamp-none transition-all italic bg-black/30 p-2 rounded">
                      "{String(snippet)}"
                    </p>
                    <div className="flex justify-end mt-1">
                      <button
                        type="button"
                        onClick={(e) => handleCopy(String(snippet), keyId, e)}
                        className="flex items-center gap-1 rounded border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-slate-300 hover:bg-white/[0.08]"
                      >
                        {copiedKey === keyId ? (
                          <>
                            <Check className="w-2.5 h-2.5 text-emerald-400" />
                            <span className="text-emerald-400">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-2.5 h-2.5 text-slate-400" />
                            <span>Copy Snippet</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
