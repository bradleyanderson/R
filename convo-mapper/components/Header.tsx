'use client';

import { Mic, MicOff, Settings, RotateCcw, Download, Zap, Map, Clock, AlignLeft } from 'lucide-react';
import { useConversation } from '@/lib/store';
import { useSession } from '@/hooks/useSession';
import { AppMode } from '@/lib/types';

const MODES: { value: AppMode; label: string; emoji: string }[] = [
  { value: 'debate', label: 'Debate', emoji: '⚔️' },
  { value: 'conversation', label: 'Chat', emoji: '💬' },
  { value: 'study', label: 'Study', emoji: '📚' },
  { value: 'meeting', label: 'Meeting', emoji: '👥' },
  { value: 'translate', label: 'Translate', emoji: '🌐' },
];

const VIEWS = [
  { value: 'map' as const, icon: Map, label: 'Map' },
  { value: 'timeline' as const, icon: Clock, label: 'Timeline' },
  { value: 'summary' as const, icon: AlignLeft, label: 'Summary' },
];

function Waveform() {
  return (
    <div className="flex items-center gap-0.5 text-red-400">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className={`waveform-bar wave-bar-${i}`} style={{ height: 8 }} />
      ))}
    </div>
  );
}

export function Header() {
  const { state, dispatch } = useConversation();
  const { toggleRecording, runAnalysis } = useSession();

  const duration = state.sessionStartTime
    ? Math.floor((Date.now() - state.sessionStartTime) / 1000)
    : 0;

  const handleExport = () => {
    const data = {
      exportedAt: new Date().toISOString(),
      session: {
        mode: state.mode,
        duration,
        speakers: state.speakers.length,
      },
      speakers: state.speakers,
      transcript: state.segments.map((s) => ({
        time: new Date(s.timestamp).toISOString(),
        speaker: state.speakers.find((sp) => sp.id === s.speakerId)?.name ?? s.speakerId,
        text: s.text,
      })),
      topics: state.topics,
      insights: state.insights,
      factChecks: state.factChecks,
      summary: state.summary,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `convo-${new Date().toISOString().slice(0, 16).replace('T', '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const mm = Math.floor(duration / 60);
  const ss = String(duration % 60).padStart(2, '0');

  return (
    <header className="flex items-center justify-between px-4 h-14 border-b border-[#1d2044] bg-[#06060f]/90 backdrop-blur-sm shrink-0 z-10">
      {/* Logo + recording state */}
      <div className="flex items-center gap-3 min-w-[180px]">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-xs font-black tracking-tight select-none shadow-lg shadow-indigo-500/20">
          CM
        </div>
        <span className="font-semibold text-white/90 text-sm hidden md:block">ConvoMapper</span>
        {state.isRecording && (
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-red-500 record-dot" />
              <span className="text-xs text-red-400 tabular-nums font-mono">
                {mm}:{ss}
              </span>
            </div>
            <Waveform />
          </div>
        )}
      </div>

      {/* Mode selector */}
      <div className="flex items-center gap-1 bg-[#0c0d1e] rounded-lg p-1 border border-[#1d2044]">
        {MODES.map(({ value, label, emoji }) => (
          <button
            key={value}
            onClick={() => dispatch({ type: 'SET_MODE', payload: value })}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
              state.mode === value
                ? 'bg-indigo-600 text-white shadow shadow-indigo-500/30'
                : 'text-white/40 hover:text-white/70'
            }`}
          >
            <span className="mr-1">{emoji}</span>
            <span className="hidden sm:inline">{label}</span>
          </button>
        ))}
      </div>

      {/* View selector */}
      <div className="flex items-center gap-1">
        {VIEWS.map(({ value, icon: Icon, label }) => (
          <button
            key={value}
            onClick={() => dispatch({ type: 'SET_ACTIVE_VIEW', payload: value })}
            title={label}
            className={`p-1.5 rounded-md text-xs transition-colors ${
              state.activeView === value
                ? 'text-indigo-400 bg-indigo-500/10'
                : 'text-white/30 hover:text-white/60'
            }`}
          >
            <Icon className="w-4 h-4" />
          </button>
        ))}
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-1 min-w-[200px] justify-end">
        {state.isAnalyzing && (
          <div className="flex items-center gap-1.5 text-xs text-indigo-400 mr-1">
            <div className="w-3 h-3 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
            <span className="hidden sm:inline">Analyzing</span>
          </div>
        )}

        <button
          onClick={runAnalysis}
          disabled={state.isAnalyzing || state.segments.length === state.lastAnalyzedCount}
          title="Analyze now (AI)"
          className="p-2 rounded-md text-white/30 hover:text-indigo-400 hover:bg-indigo-500/10 transition-colors disabled:opacity-20 disabled:cursor-not-allowed"
        >
          <Zap className="w-4 h-4" />
        </button>

        <button
          onClick={handleExport}
          disabled={state.segments.length === 0}
          title="Export session"
          className="p-2 rounded-md text-white/30 hover:text-white/60 hover:bg-white/5 transition-colors disabled:opacity-20 disabled:cursor-not-allowed"
        >
          <Download className="w-4 h-4" />
        </button>

        <button
          onClick={() => dispatch({ type: 'RESET_SESSION' })}
          title="Reset session"
          className="p-2 rounded-md text-white/30 hover:text-white/60 hover:bg-white/5 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        <button
          onClick={() => dispatch({ type: 'TOGGLE_SETTINGS' })}
          title="Settings"
          className="p-2 rounded-md text-white/30 hover:text-white/60 hover:bg-white/5 transition-colors"
        >
          <Settings className="w-4 h-4" />
        </button>

        <button
          onClick={toggleRecording}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-semibold transition-all ${
            state.isRecording
              ? 'bg-red-600 hover:bg-red-700 text-white shadow-lg shadow-red-500/25'
              : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-500/25'
          }`}
        >
          {state.isRecording ? (
            <>
              <MicOff className="w-3.5 h-3.5" />
              Stop
            </>
          ) : (
            <>
              <Mic className="w-3.5 h-3.5" />
              Start
            </>
          )}
        </button>
      </div>
    </header>
  );
}
