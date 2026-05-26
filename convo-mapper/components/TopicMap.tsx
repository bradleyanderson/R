'use client';

import { useConversation } from '@/lib/store';
import { TopicCard } from './TopicCard';
import { Mic, Zap } from 'lucide-react';
import { useSession } from '@/hooks/useSession';
import { SENTIMENT_CONFIG } from '@/lib/types';

function EmptyState() {
  const { state, dispatch } = useConversation();
  const { toggleRecording } = useSession();

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 text-center px-8">
      <div className="relative">
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-600/20 to-violet-600/20 border border-indigo-500/20 flex items-center justify-center">
          <Mic className="w-9 h-9 text-indigo-400/60" />
        </div>
        <div className="absolute -bottom-2 -right-2 w-8 h-8 rounded-xl bg-[#0c0d1e] border border-[#1d2044] flex items-center justify-center">
          <Zap className="w-4 h-4 text-yellow-400/60" />
        </div>
      </div>
      <div>
        <h2 className="text-lg font-bold text-white/80 mb-1">AI Conversation Mapper</h2>
        <p className="text-sm text-white/30 max-w-sm">
          Start speaking to map your conversation. Topics, arguments, insights, and fact-checks appear here in real time.
        </p>
      </div>
      <div className="flex flex-col gap-2 text-xs text-white/20 items-center">
        <div className="flex flex-wrap gap-x-4 gap-y-1 justify-center">
          <span>🎙️ Real-time transcription</span>
          <span>🗺️ Topic mapping</span>
          <span>⚔️ Argument tracking</span>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 justify-center">
          <span>💡 AI insights</span>
          <span>🌐 Translation</span>
          <span>⚡ Fact checks</span>
        </div>
      </div>
      {!state.apiKey ? (
        <button
          onClick={() => dispatch({ type: 'SET_SHOW_SETTINGS', payload: true })}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-semibold transition-colors"
        >
          Set API Key to Start
        </button>
      ) : (
        <button
          onClick={toggleRecording}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-semibold transition-colors flex items-center gap-2"
        >
          <Mic className="w-4 h-4" />
          Start Recording
        </button>
      )}
    </div>
  );
}

function SummaryView() {
  const { state } = useConversation();
  if (!state.summary && state.topics.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-white/20 text-sm">
        No summary yet — start speaking and run analysis
      </div>
    );
  }
  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      {state.summary && (
        <div className="p-4 rounded-xl bg-[#0c0d1e] border border-[#1d2044]">
          <h3 className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-2">Summary</h3>
          <p className="text-sm text-white/70">{state.summary}</p>
        </div>
      )}
      {state.topics.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-3">Topics Covered</h3>
          <div className="space-y-2">
            {state.topics.map((t) => (
              <div key={t.id} className="flex items-start gap-3 p-3 rounded-lg bg-[#0c0d1e] border border-[#1d2044]">
                <span className="text-lg shrink-0">
                  {t.category === 'science' ? '🔬' : t.category === 'politics' ? '🏛️' : t.category === 'economic' ? '💰' : '📌'}
                </span>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-white/90">{t.title}</p>
                  <p className="text-xs text-white/40 mt-0.5">{t.summary}</p>
                  <div className="flex flex-wrap gap-1 mt-1.5">
                    {t.keywords.slice(0, 5).map((kw) => (
                      <span key={kw} className="text-[10px] text-white/30 bg-white/5 px-1.5 py-0.5 rounded">
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>
                <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full border shrink-0 ${SENTIMENT_CONFIG[t.sentiment]?.bg ?? ''}`}>
                  {SENTIMENT_CONFIG[t.sentiment]?.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function TimelineView() {
  const { state } = useConversation();
  if (state.segments.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-white/20 text-sm">
        No transcript yet — start recording
      </div>
    );
  }

  const sessionStart = state.sessionStartTime ?? state.segments[0]?.timestamp ?? Date.now();

  return (
    <div className="p-4 space-y-1 overflow-y-auto h-full">
      {state.segments.map((seg) => {
        const speaker = state.speakers.find((s) => s.id === seg.speakerId);
        const elapsed = Math.floor((seg.timestamp - sessionStart) / 1000);
        const mm = Math.floor(elapsed / 60);
        const ss = String(elapsed % 60).padStart(2, '0');
        return (
          <div key={seg.id} className="flex gap-3 py-1.5 animate-fade-up">
            <span className="text-[10px] text-white/20 font-mono shrink-0 mt-1 w-10 text-right">
              {mm}:{ss}
            </span>
            <div
              className="w-0.5 rounded-full shrink-0"
              style={{ background: speaker?.color ?? '#94a3b8' }}
            />
            <div className="min-w-0">
              <span className="text-xs font-semibold" style={{ color: speaker?.color ?? '#94a3b8' }}>
                {speaker?.name ?? seg.speakerId}
              </span>
              <p className="text-sm text-white/70 mt-0.5">{seg.text}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function TopicMap() {
  const { state } = useConversation();

  const isEmpty = state.topics.length === 0;

  if (state.activeView === 'timeline') {
    return (
      <main className="flex-1 overflow-hidden bg-[#08090f] flex flex-col">
        <div className="px-4 py-2 border-b border-[#1d2044] flex items-center justify-between">
          <span className="text-xs font-semibold text-white/30 uppercase tracking-wider">Timeline</span>
          <span className="text-xs text-white/20">{state.segments.length} segments</span>
        </div>
        <TimelineView />
      </main>
    );
  }

  if (state.activeView === 'summary') {
    return (
      <main className="flex-1 overflow-y-auto bg-[#08090f]">
        <div className="px-4 py-2 border-b border-[#1d2044]">
          <span className="text-xs font-semibold text-white/30 uppercase tracking-wider">Summary</span>
        </div>
        <SummaryView />
      </main>
    );
  }

  return (
    <main className="flex-1 overflow-hidden bg-[#08090f] flex flex-col">
      {/* Map header */}
      {!isEmpty && (
        <div className="px-4 py-2 border-b border-[#1d2044] flex items-center justify-between shrink-0">
          <span className="text-xs font-semibold text-white/30 uppercase tracking-wider">
            Topic Map
          </span>
          <div className="flex items-center gap-3 text-xs text-white/20">
            <span>{state.topics.length} topics</span>
            {state.isAnalyzing && (
              <div className="flex items-center gap-1 text-indigo-400">
                <div className="w-3 h-3 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                Analyzing...
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error banner */}
      {state.error && (
        <div className="mx-4 mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400 shrink-0">
          ⚠️ {state.error}
        </div>
      )}

      {/* Content */}
      {isEmpty ? (
        <EmptyState />
      ) : (
        <div className="flex-1 overflow-y-auto p-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3 auto-rows-min">
            {state.topics.map((topic) => (
              <TopicCard key={topic.id} topic={topic} />
            ))}
          </div>
        </div>
      )}
    </main>
  );
}
