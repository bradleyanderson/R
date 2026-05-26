'use client';

import { useEffect, useRef } from 'react';
import { MessageSquare } from 'lucide-react';
import { useConversation } from '@/lib/store';

function formatTime(ts: number) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function TranscriptFeed() {
  const { state } = useConversation();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.segments.length, state.pendingText]);

  return (
    <div className="flex flex-col border-b border-[#1d2044]" style={{ height: '55%' }}>
      {/* Header */}
      <div className="px-3 py-2 border-b border-[#1d2044] flex items-center justify-between shrink-0">
        <div className="flex items-center gap-1.5">
          <MessageSquare className="w-3.5 h-3.5 text-white/30" />
          <span className="text-xs font-semibold text-white/30 uppercase tracking-wider">Live Transcript</span>
        </div>
        {state.segments.length > 0 && (
          <span className="text-[10px] text-white/20">{state.segments.length} segments</span>
        )}
      </div>

      {/* Transcript list */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5">
        {state.segments.length === 0 && !state.pendingText ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-xs text-white/15 text-center">
              Transcript will appear here<br />as you speak
            </p>
          </div>
        ) : (
          <>
            {state.segments.map((seg, idx) => {
              const speaker = state.speakers.find((s) => s.id === seg.speakerId);
              const isLast = idx === state.segments.length - 1;
              return (
                <div
                  key={seg.id}
                  className={`flex gap-2 text-xs ${isLast ? 'animate-fade-up' : ''}`}
                >
                  {/* Speaker bar */}
                  <div
                    className="w-0.5 rounded-full shrink-0 my-0.5"
                    style={{ background: speaker?.color ?? '#94a3b8' }}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-1.5 mb-0.5">
                      <span
                        className="font-semibold text-[11px] shrink-0"
                        style={{ color: speaker?.color ?? '#94a3b8' }}
                      >
                        {speaker?.name ?? seg.speakerId}
                      </span>
                      <span className="text-[10px] text-white/15 font-mono shrink-0">
                        {formatTime(seg.timestamp)}
                      </span>
                    </div>
                    <p className="text-white/70 leading-relaxed">{seg.text}</p>
                  </div>
                </div>
              );
            })}

            {/* Pending (interim) text */}
            {state.pendingText && (
              <div className="flex gap-2 text-xs opacity-50 animate-pulse">
                <div
                  className="w-0.5 rounded-full shrink-0 my-0.5"
                  style={{
                    background:
                      state.speakers.find((s) => s.id === state.activeSpeakerId)?.color ??
                      '#94a3b8',
                  }}
                />
                <div className="min-w-0">
                  <span
                    className="font-semibold text-[11px] italic"
                    style={{
                      color:
                        state.speakers.find((s) => s.id === state.activeSpeakerId)?.color ??
                        '#94a3b8',
                    }}
                  >
                    {state.speakers.find((s) => s.id === state.activeSpeakerId)?.name ?? '?'}
                  </span>
                  <p className="text-white/50 italic mt-0.5">{state.pendingText}</p>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Analysis checkpoint markers */}
      {state.lastAnalyzedCount > 0 && state.lastAnalyzedCount < state.segments.length && (
        <div className="px-3 py-1 border-t border-[#1d2044] shrink-0">
          <div className="flex items-center gap-1.5">
            <div className="h-px flex-1 bg-indigo-500/20" />
            <span className="text-[10px] text-indigo-400/50">
              {state.segments.length - state.lastAnalyzedCount} unanalyzed segments
            </span>
            <div className="h-px flex-1 bg-indigo-500/20" />
          </div>
        </div>
      )}
    </div>
  );
}
