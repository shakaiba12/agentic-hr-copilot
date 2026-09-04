import React, { useState } from 'react';
import { Code2, ChevronDown, ChevronRight, Copy, Check } from 'lucide-react';

interface SqlViewerProps {
  sql: string;
}

export const SqlViewer: React.FC<SqlViewerProps> = ({ sql }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!sql) return null;

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="my-2 overflow-hidden rounded-xl border border-white/10 bg-black/20 backdrop-blur-sm transition-all duration-200">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between px-3.5 py-2.5 text-left text-xs font-medium text-slate-300 hover:bg-white/[0.04] transition-colors"
      >
        <div className="flex items-center gap-2">
          <Code2 className="w-3.5 h-3.5 text-violet-400" />
          <span>Generated SQL Query</span>
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
        <div className="border-t border-white/10 p-3 bg-black/40">
          <div className="relative">
            <pre className="overflow-x-auto rounded-lg border border-white/10 bg-[#090d16] p-3 text-xs font-mono text-cyan-200 scroll-on-hover">
              <code>{sql}</code>
            </pre>
            <button
              type="button"
              onClick={handleCopy}
              className="absolute top-2 right-2 flex items-center gap-1 rounded-md border border-white/15 bg-white/[0.06] px-2 py-1 text-[11px] text-slate-200 hover:bg-white/[0.12] transition-colors shadow"
              title="Copy SQL"
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span className="text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3 text-slate-300" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
