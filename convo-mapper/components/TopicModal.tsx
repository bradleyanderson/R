'use client';

import { X } from 'lucide-react';
import { useConversation } from '@/lib/store';
import { TopicNode, CATEGORY_ICONS, SENTIMENT_CONFIG, FACT_CHECK_CONFIG } from '@/lib/types';

export function TopicModal() {
  const { state, dispatch } = useConversation();
  const topic = state.topics.find((t) => t.id === state.expandedTopicId);

  if (!topic) return null;

  const close = () => dispatch({ type: 'TOGGLE_TOPIC', payload: null });

  const getSpeakerColor = (id: string) =>
    state.speakers.find((s) => s.id === id)?.color ?? '#94a3b8';
  const getSpeakerName = (id: string) =>
    state.speakers.find((s) => s.id === id)?.name ?? id;

  const topicFactChecks = state.factChecks.filter((fc) =>
    topic.keywords.some(
      (kw) =>
        fc.claim.toLowerCase().includes(kw.toLowerCase()) ||
        topic.title.toLowerCase().includes(kw.toLowerCase())
    )
  );

  const sentiment = SENTIMENT_CONFIG[topic.sentiment] ?? SENTIMENT_CONFIG.neutral;
  const icon = CATEGORY_ICONS[topic.category] ?? '📌';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={close}
    >
      <div
        className="bg-[#0c0d1e] border border-[#2d3264] rounded-2xl w-full max-w-2xl max-h-[85vh] overflow-y-auto shadow-2xl shadow-indigo-500/10"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal header */}
        <div className="sticky top-0 bg-[#0c0d1e] border-b border-[#1d2044] px-6 py-4 flex items-start justify-between z-10">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xl">{icon}</span>
              <h2 className="text-lg font-bold text-white">{topic.title}</h2>
              <span className={`text-xs px-2 py-0.5 rounded-full border ${sentiment.bg}`}>
                {sentiment.label}
              </span>
            </div>
            <p className="text-sm text-white/50">{topic.summary}</p>
          </div>
          <button
            onClick={close}
            className="p-1.5 rounded-lg text-white/40 hover:text-white/80 hover:bg-white/5 transition-colors ml-4 shrink-0"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="px-6 py-4 space-y-6">
          {/* Arguments */}
          {topic.arguments.length > 0 && (
            <section>
              <h3 className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-3">
                ⚔️ Arguments by Speaker
              </h3>
              <div className="space-y-3">
                {topic.arguments.map((arg, i) => {
                  const color = getSpeakerColor(arg.speakerId);
                  const name = getSpeakerName(arg.speakerId);
                  const posColor =
                    arg.position === 'for' ? '#22c55e' : arg.position === 'against' ? '#ef4444' : '#94a3b8';
                  return (
                    <div
                      key={i}
                      className="flex gap-3 p-3 rounded-lg bg-[#111228] border border-[#1d2044]"
                    >
                      <div className="w-1 rounded-full shrink-0" style={{ background: color }} />
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-semibold" style={{ color }}>
                            {name}
                          </span>
                          <span
                            className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                            style={{ background: posColor + '20', color: posColor }}
                          >
                            {arg.position}
                          </span>
                        </div>
                        <p className="text-sm text-white/70">{arg.text}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* AI Insights */}
          {topic.aiInsights.length > 0 && (
            <section>
              <h3 className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-3">
                💡 AI Insights
              </h3>
              <div className="space-y-2">
                {topic.aiInsights.map((insight, i) => (
                  <div
                    key={i}
                    className="flex gap-2 p-2.5 rounded-lg bg-indigo-500/5 border border-indigo-500/20"
                  >
                    <span className="text-indigo-400 shrink-0">→</span>
                    <p className="text-sm text-indigo-200/80">{insight}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Related Facts */}
          {topic.relatedFacts.length > 0 && (
            <section>
              <h3 className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-3">
                🔍 Context & Facts
              </h3>
              <ul className="space-y-1.5">
                {topic.relatedFacts.map((fact, i) => (
                  <li key={i} className="flex gap-2 text-sm text-white/60">
                    <span className="text-white/20 shrink-0 mt-0.5">•</span>
                    {fact}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Notable Quotes */}
          {topic.notableQuotes.length > 0 && (
            <section>
              <h3 className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-3">
                💬 Notable Quotes
              </h3>
              <div className="space-y-2">
                {topic.notableQuotes.map((q, i) => (
                  <blockquote
                    key={i}
                    className="text-sm text-white/60 italic border-l-2 border-indigo-500/50 pl-3 py-1"
                  >
                    &ldquo;{q}&rdquo;
                  </blockquote>
                ))}
              </div>
            </section>
          )}

          {/* Fact Checks */}
          {topicFactChecks.length > 0 && (
            <section>
              <h3 className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-3">
                ⚡ Fact Checks
              </h3>
              <div className="space-y-2">
                {topicFactChecks.map((fc, i) => {
                  const cfg = FACT_CHECK_CONFIG[fc.status];
                  return (
                    <div
                      key={i}
                      className={`p-2.5 rounded-lg border border-[#1d2044] ${cfg.bg}`}
                    >
                      <div className="flex items-start gap-2">
                        <span className={`font-bold ${cfg.color} shrink-0`}>{cfg.icon}</span>
                        <div>
                          <p className={`text-xs font-semibold ${cfg.color}`}>{cfg.label}</p>
                          <p className="text-xs text-white/70 mt-0.5">&ldquo;{fc.claim}&rdquo;</p>
                          <p className="text-xs text-white/40 mt-1">{fc.explanation}</p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Translation */}
          {(topic as TopicNode & { translationSummary?: string }).translationSummary && (
            <section>
              <h3 className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-3">
                🌐 Translated Summary
              </h3>
              <p className="text-sm text-white/60 bg-[#111228] rounded-lg p-3 border border-[#1d2044]">
                {(topic as TopicNode & { translationSummary?: string }).translationSummary}
              </p>
            </section>
          )}

          {/* Keywords */}
          {topic.keywords.length > 0 && (
            <section>
              <h3 className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-2">
                🏷️ Keywords
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {topic.keywords.map((kw) => (
                  <span
                    key={kw}
                    className="text-xs text-white/40 bg-white/5 border border-white/10 px-2 py-0.5 rounded-full"
                  >
                    {kw}
                  </span>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
