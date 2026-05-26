'use client';

import { Lightbulb } from 'lucide-react';
import { useConversation } from '@/lib/store';
import { FACT_CHECK_CONFIG } from '@/lib/types';

export function InsightsPanel() {
  const { state } = useConversation();

  const hasContent =
    state.insights.length > 0 || state.factChecks.length > 0 || state.summary;

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* Header */}
      <div className="px-3 py-2 border-b border-[#1d2044] flex items-center gap-1.5 shrink-0">
        <Lightbulb className="w-3.5 h-3.5 text-yellow-400/60" />
        <span className="text-xs font-semibold text-white/30 uppercase tracking-wider">AI Insights</span>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {!hasContent ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-xs text-white/15 text-center">
              Insights and fact-checks<br />appear here after analysis
            </p>
          </div>
        ) : (
          <>
            {/* Key insights */}
            {state.insights.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold text-white/25 uppercase tracking-wider mb-2">
                  Key Insights
                </p>
                <ul className="space-y-1.5">
                  {state.insights.slice(-8).map((insight, i) => (
                    <li key={i} className="flex gap-2 text-xs text-white/60 animate-fade-up">
                      <span className="text-indigo-400 shrink-0">→</span>
                      {insight}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Fact checks */}
            {state.factChecks.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold text-white/25 uppercase tracking-wider mb-2">
                  Fact Checks
                </p>
                <div className="space-y-1.5">
                  {state.factChecks.slice(-6).map((fc, i) => {
                    const cfg = FACT_CHECK_CONFIG[fc.status];
                    const speaker = state.speakers.find((s) => s.id === fc.speakerId);
                    return (
                      <div
                        key={fc.id ?? i}
                        className={`p-2 rounded-lg border border-[#1d2044] ${cfg.bg} animate-fade-up`}
                      >
                        <div className="flex items-center gap-1.5 mb-0.5">
                          <span className={`text-xs font-bold ${cfg.color}`}>{cfg.icon}</span>
                          <span className={`text-[10px] font-semibold ${cfg.color}`}>{cfg.label}</span>
                          {speaker && (
                            <span
                              className="text-[10px] ml-auto"
                              style={{ color: speaker.color }}
                            >
                              {speaker.name}
                            </span>
                          )}
                        </div>
                        <p className="text-[11px] text-white/60 italic">&ldquo;{fc.claim}&rdquo;</p>
                        <p className="text-[10px] text-white/35 mt-0.5">{fc.explanation}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Summary */}
            {state.summary && (
              <div>
                <p className="text-[10px] font-semibold text-white/25 uppercase tracking-wider mb-2">
                  Summary
                </p>
                <p className="text-xs text-white/50 leading-relaxed">{state.summary}</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
