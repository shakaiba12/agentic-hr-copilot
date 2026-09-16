import React from 'react';
import { Menu, Database, Sparkles, ShieldCheck } from 'lucide-react';
import { HealthStatus } from '../types/api';

interface HeaderProps {
  health: HealthStatus | null;
  onToggleSidebar: () => void;
  sidebarCollapsed: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  onToggleSidebar,
}) => {
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="shrink-0 flex items-center justify-between gap-4 border-b border-white/10 px-4 py-3 md:px-6 bg-white/[0.02]">
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="md:hidden grid h-9 w-9 place-items-center rounded-xl border border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08] transition-colors"
          title="Toggle Navigation"
        >
          <Menu className="w-4 h-4" />
        </button>

        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="truncate text-base md:text-xl font-semibold tracking-tight text-white flex items-center gap-1.5">
              PeopleQuery AI
              <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] uppercase font-bold tracking-wider rounded-md bg-gradient-to-r from-cyan-400/20 to-violet-400/20 border border-cyan-400/30 text-cyan-300">
                Copilot
              </span>
            </h1>
          </div>
          <p className="hidden xs:block truncate text-xs text-slate-400">
            Enterprise HR Intelligence • Operational Analytics & Policy Knowledge
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {/* System status pill */}
        <div className="flex items-center gap-1.5 rounded-full border border-white/10 bg-black/40 px-2.5 py-1 text-xs">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              isHealthy ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]' : 'bg-amber-400 animate-pulse'
            }`}
          />
          <span className="text-[11px] font-medium text-slate-300 hidden sm:inline">
            {isHealthy ? 'System Active' : 'Connecting...'}
          </span>
        </div>

        {/* Model info pill */}
        {health && (
          <div className="hidden lg:flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] text-slate-300">
            <Sparkles className="w-3 h-3 text-cyan-400" />
            <span className="font-semibold text-cyan-200">{health.provider.toUpperCase()}</span>
            <span className="text-slate-500">•</span>
            <span className="text-slate-400">{health.model}</span>
          </div>
        )}

        {/* Database status pill */}
        <div className="hidden xl:flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] text-slate-300">
          <Database className="w-3 h-3 text-violet-400" />
          <span>SQLite Database</span>
        </div>

        {/* Guardrails active pill */}
        <div className="hidden md:flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-950/30 px-2.5 py-1 text-[11px] text-emerald-300">
          <ShieldCheck className="w-3 h-3 text-emerald-400" />
          <span>Guardrails Enforced</span>
        </div>
      </div>
    </header>
  );
};
