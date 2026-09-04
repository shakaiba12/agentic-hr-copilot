import React, { useMemo } from 'react';
import { Table, Layers } from 'lucide-react';

interface DataTableProps {
  rows: Array<Record<string, unknown>>;
  rowCount: number;
  wasTruncated?: boolean;
}

export const DataTable: React.FC<DataTableProps> = ({ rows, rowCount, wasTruncated }) => {
  if (!rows || rows.length === 0) return null;

  const columns = useMemo(() => {
    const keys = new Set<string>();
    rows.forEach((row) => {
      Object.keys(row).forEach((k) => keys.add(k));
    });
    return Array.from(keys);
  }, [rows]);

  const formatHeader = (key: string) => {
    return key
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());
  };

  const formatCellValue = (val: unknown): string => {
    if (val === null || val === undefined) return '—';
    if (typeof val === 'boolean') return val ? 'Yes' : 'No';
    if (typeof val === 'object') return JSON.stringify(val);
    return String(val);
  };

  return (
    <div className="my-3 overflow-hidden rounded-xl border border-white/10 bg-black/25 backdrop-blur-md shadow-xl">
      {/* Table Header Controls */}
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-2.5 bg-white/[0.02]">
        <div className="flex items-center gap-2">
          <Table className="w-4 h-4 text-cyan-300" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Workforce Records
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-cyan-950/60 border border-cyan-400/30 text-cyan-300">
            <Layers className="w-3 h-3" />
            {rowCount} {rowCount === 1 ? 'row' : 'rows'}
          </span>
          {wasTruncated && (
            <span className="text-[10px] text-amber-400 bg-amber-950/40 border border-amber-500/30 px-2 py-0.5 rounded-full">
              Truncated
            </span>
          )}
        </div>
      </div>

      {/* Responsive Horizontal Scroll Table */}
      <div className="overflow-x-auto scroll-on-hover max-h-[380px]">
        <table className="w-full text-left text-xs border-collapse">
          <thead className="sticky top-0 z-10 bg-[#0d131f] border-b border-white/10 text-slate-400 uppercase tracking-wider text-[10px]">
            <tr>
              {columns.map((col) => (
                <th key={col} className="px-4 py-2.5 font-semibold whitespace-nowrap">
                  {formatHeader(col)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5 text-slate-200">
            {rows.map((row, rIdx) => (
              <tr
                key={rIdx}
                className="hover:bg-cyan-500/[0.04] transition-colors duration-150 group"
              >
                {columns.map((col) => {
                  const val = row[col];
                  const formatted = formatCellValue(val);
                  return (
                    <td
                      key={col}
                      className="px-4 py-2.5 whitespace-nowrap group-hover:text-white"
                    >
                      {formatted}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
