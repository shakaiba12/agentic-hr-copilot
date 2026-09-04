import React from 'react';
import { BarChart3, Hash } from 'lucide-react';

interface KpiCardProps {
  data: Record<string, unknown>;
}

export const KpiCard: React.FC<KpiCardProps> = ({ data }) => {
  const entries = Object.entries(data);
  if (entries.length === 0) return null;

  // Format key into friendly label
  const formatKey = (key: string) => {
    return key
      .replace(/_/g, ' ')
      .replace(/count/i, 'Count')
      .replace(/total/i, 'Total')
      .replace(/avg/i, 'Average')
      .replace(/\b\w/g, (char) => char.toUpperCase());
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 my-3">
      {entries.map(([key, val], idx) => {
        const isNumeric = typeof val === 'number' || (!isNaN(Number(val)) && val !== '');
        const displayVal = String(val ?? '—');

        return (
          <div
            key={idx}
            className="relative overflow-hidden rounded-xl border border-cyan-400/20 bg-gradient-to-br from-cyan-950/30 via-slate-900/60 to-violet-950/30 p-4 backdrop-blur-md shadow-lg shadow-cyan-950/20 group hover:border-cyan-400/40 transition-all duration-200"
          >
            <div className="absolute top-0 right-0 -mr-4 -mt-4 w-16 h-16 rounded-full bg-cyan-400/5 blur-xl group-hover:bg-cyan-400/10 transition-all"></div>
            
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-300/80">
                {formatKey(key)}
              </span>
              <div className="p-1.5 rounded-lg bg-white/[0.04] border border-white/10 text-cyan-300">
                {isNumeric ? <Hash className="w-3.5 h-3.5" /> : <BarChart3 className="w-3.5 h-3.5" />}
              </div>
            </div>

            <div className="text-2xl font-bold tracking-tight text-white flex items-baseline gap-1">
              <span className="bg-gradient-to-r from-white via-cyan-100 to-cyan-300 bg-clip-text text-transparent">
                {displayVal}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
};
