'use client';

import { ChevronDown, ChevronUp } from 'lucide-react';
import { useConversation } from '@/lib/store';
import { TopicNode, CATEGORY_ICONS, SENTIMENT_CONFIG } from '@/lib/types';

function SpeakerDot({ color }: { color: string }) {
  return (
    <span
      className="inline-block w-2 h-2 rounded-full shrink-0 mt-1"
      style={{ background: color }}
    />
  );
}

export function TopicCard({ topic }: { topic: TopicNode }) {
  const { state, dispatch } = useConversation();
  const isExpanded = state.expandedTopicId === topic.id;

  const toggleExpand = () => {
    dispatch({ type: 'TOGGLE_TOPIC', payload: isExpanded ? null : topic.id });
  };

  const sentiment = SENTIMENT_CONFIG[topic.sentiment] ?? SENTIMENT_CONFIG.neutral;
  const icon = CATEGORY_ICONS[topic.category] ?? '📌';

  const getSpeakerColor = (speakerId: string) =>
    state.speakers.find((s) => s.id === speakerId)?.color ?? '#94a3b8';

  const getSpeakerName = (speakerId: string) =>
    state.speakers.find((s) => s.id === speakerId)?.name ?? speakerId;

  return (
    <div
      className={`rounded-xl border bg-[#0c0d1e] transition-all duration-200 overflow-hidden ${
        topic.isNew ? 'animate-fade-up' : ''
      } ${topic.updated ? 'topic-updated' : ''} ${
        isExpanded ? 'border-indigo-500/50 shadow-lg shadow-indigo-500/10' : 'border-[#1d2044] hover:border-[#2d3264]'
      }`}
    >
      {/* Card header */}
      <div
        className="p-3 cursor-pointer"
        onClick={() => dispatch({ type: 'TOGGLE_TOPIC', payload: topic.id })}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-base shrink-0">{icon}</span>
            <h3 className="font-semibold text-sm text-white/90 truncate">{topic.title}</h3>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full border ${sentiment.bg}`}>
              {sentiment.label}
            </span>
          </div>
        </div>

        <p className="mt-1.5 text-xs text-white/50 line-clamp-2">{topic.summary}</p>

        {/* Speaker argument pills */}
        {topic.arguments.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {topic.arguments.slice(0, 3).map((arg, i) => {
              const color = getSpeakerColor(arg.speakerId);
              const name = getSpeakerName(arg.speakerId);
              return (
                <div
                  key={i}
                  className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full"
                  style={{ background: color + '15', color }}
                >
                  <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: color }} />
                  {name}: {arg.position}
                </div>
              );
            })}
            {topic.arguments.length > 3 && (
              <span className="text-[10px] text-white/30">+{topic.arguments.length - 3}</span>
            )}
          </div>
        )}

        {/* Keywords */}
        {topic.keywords.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {topic.keywords.slice(0, 4).map((kw) => (
              <span key={kw} className="text-[10px] text-white/30 bg-white/5 px-1.5 py-0.5 rounded">
                {kw}
              </span>
            ))}
          </div>
        )}

        {/* Expand/collapse indicator */}
        <div className="flex items-center justify-center mt-2 text-white/20">
          {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </div>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="border-t border-[#1d2044] px-3 pb-3 space-y-3">
          {/* Arguments */}
          {topic.arguments.length > 0 && (
            <div className="pt-3">
              <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider mb-2">Arguments</p>
              <div className="space-y-2">
                {topic.arguments.map((arg, i) => {
                  const color = getSpeakerColor(arg.speakerId);
                  const name = getSpeakerName(arg.speakerId);
                  const posColor =
                    arg.position === 'for'
                      ? '#22c55e'
                      : arg.position === 'against'
                      ? '#ef4444'
                      : '#94a3b8';
                  return (
                    <div key={i} className="flex gap-2">
                      <SpeakerDot color={color} />
                      <div className="min-w-0">
                        <span className="text-xs font-medium" style={{ color }}>
                          {name}
                        </span>
                        <span
                          className="ml-1 text-[10px] px-1 py-0.5 rounded"
                          style={{ background: posColor + '20', color: posColor }}
                        >
                          {arg.position}
                        </span>
                        <p className="text-xs text-white/60 mt-0.5">{arg.text}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* AI Insights */}
          {topic.aiInsights.length > 0 && (
            <div>
              <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider mb-2">
                💡 AI Insights
              </p>
              <ul className="space-y-1">
                {topic.aiInsights.map((insight, i) => (
                  <li key={i} className="text-xs text-indigo-300/80 flex gap-1.5">
                    <span className="text-indigo-400 shrink-0">→</span>
                    {insight}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Related Facts */}
          {topic.relatedFacts.length > 0 && (
            <div>
              <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider mb-2">
                🔍 Context & Facts
              </p>
              <ul className="space-y-1">
                {topic.relatedFacts.map((fact, i) => (
                  <li key={i} className="text-xs text-white/50 flex gap-1.5">
                    <span className="text-white/20 shrink-0">•</span>
                    {fact}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Notable Quotes */}
          {topic.notableQuotes.length > 0 && (
            <div>
              <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider mb-2">
                💬 Notable Quotes
              </p>
              <div className="space-y-1.5">
                {topic.notableQuotes.map((quote, i) => (
                  <blockquote
                    key={i}
                    className="text-xs text-white/60 italic border-l-2 border-indigo-500/40 pl-2"
                  >
                    &ldquo;{quote}&rdquo;
                  </blockquote>
                ))}
              </div>
            </div>
          )}

          {/* Translation */}
          {(topic as TopicNode & { translationSummary?: string }).translationSummary && (
            <div>
              <p className="text-[10px] font-semibold text-white/30 uppercase tracking-wider mb-1.5">
                🌐 Translation
              </p>
              <p className="text-xs text-white/60">
                {(topic as TopicNode & { translationSummary?: string }).translationSummary}
              </p>
            </div>
          )}

          {/* Full expand button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              dispatch({ type: 'TOGGLE_TOPIC', payload: topic.id });
            }}
            className="w-full text-xs text-white/30 hover:text-white/50 pt-1 transition-colors"
          >
            Collapse ↑
          </button>
        </div>
      )}
    </div>
  );
}
