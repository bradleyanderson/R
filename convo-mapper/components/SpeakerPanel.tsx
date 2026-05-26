'use client';

import { useState } from 'react';
import { Plus, Trash2, Mic } from 'lucide-react';
import { useConversation } from '@/lib/store';
import { SPEAKER_COLORS, Speaker } from '@/lib/types';

function formatDuration(ms: number) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

function SpeakerCard({ speaker, index }: { speaker: Speaker; index: number }) {
  const { state, dispatch } = useConversation();
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(speaker.name);
  const isActive = state.activeSpeakerId === speaker.id;

  const handleRename = () => {
    if (name.trim()) {
      dispatch({ type: 'UPDATE_SPEAKER', payload: { id: speaker.id, updates: { name: name.trim() } } });
    }
    setEditing(false);
  };

  const totalWords = speaker.wordCount;
  const totalSegments = speaker.segmentCount;

  return (
    <div
      onClick={() => !editing && dispatch({ type: 'SET_ACTIVE_SPEAKER', payload: speaker.id })}
      className={`relative rounded-lg p-3 cursor-pointer transition-all border ${
        isActive
          ? 'bg-[#111228] border-[#1d2044]'
          : 'bg-[#0c0d1e]/60 border-transparent hover:border-[#1d2044]'
      }`}
      style={{ borderLeftColor: speaker.color, borderLeftWidth: 3 }}
    >
      {/* Active mic indicator */}
      {isActive && state.isRecording && (
        <div className="absolute top-2 right-2">
          <Mic className="w-3 h-3 text-red-400 animate-pulse" />
        </div>
      )}

      {/* Keyboard shortcut badge */}
      <span
        className="absolute top-2 right-7 text-[10px] font-mono rounded px-1"
        style={{ background: speaker.color + '22', color: speaker.color }}
      >
        {index + 1}
      </span>

      {/* Name */}
      {editing ? (
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={handleRename}
          onKeyDown={(e) => e.key === 'Enter' && handleRename()}
          onClick={(e) => e.stopPropagation()}
          className="w-full bg-[#06060f] border border-[#1d2044] rounded px-2 py-0.5 text-sm text-white focus:outline-none focus:border-indigo-500"
        />
      ) : (
        <div
          onDoubleClick={(e) => { e.stopPropagation(); setEditing(true); }}
          className="text-sm font-semibold text-white/90 truncate pr-8"
          style={{ color: speaker.color }}
          title="Double-click to rename"
        >
          {speaker.name}
        </div>
      )}

      {/* Stats */}
      {(totalSegments > 0 || totalWords > 0) && (
        <div className="flex gap-3 mt-1.5 text-[11px] text-white/40">
          <span>{totalSegments} segments</span>
          <span>{totalWords} words</span>
        </div>
      )}

      {/* Progress bar */}
      {totalWords > 0 && (
        <div className="mt-2 h-0.5 bg-[#1d2044] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${Math.min(100, (totalWords / Math.max(...(state.speakers.map(s => s.wordCount).filter(w => w > 0).concat([1])))  ) * 100)}%`,
              background: speaker.color,
            }}
          />
        </div>
      )}
    </div>
  );
}

export function SpeakerPanel() {
  const { state, dispatch } = useConversation();

  const addSpeaker = () => {
    const idx = state.speakers.length;
    const color = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
    const newSpeaker: Speaker = {
      id: `speaker-${Date.now()}`,
      name: `Speaker ${idx + 1}`,
      color,
      isActive: false,
      segmentCount: 0,
      wordCount: 0,
      firstSeenAt: 0,
      lastSeenAt: 0,
    };
    dispatch({ type: 'ADD_SPEAKER', payload: newSpeaker });
  };

  const removeSpeaker = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (state.speakers.length <= 1) return;
    dispatch({ type: 'REMOVE_SPEAKER', payload: id });
  };

  const totalWords = state.speakers.reduce((sum, s) => sum + s.wordCount, 0);

  return (
    <aside className="w-52 min-w-52 flex flex-col border-r border-[#1d2044] bg-[#06060f] overflow-hidden">
      {/* Header */}
      <div className="px-3 py-2.5 border-b border-[#1d2044] flex items-center justify-between">
        <span className="text-xs font-semibold text-white/40 uppercase tracking-wider">Speakers</span>
        <button
          onClick={addSpeaker}
          disabled={state.speakers.length >= 8}
          className="p-1 rounded text-white/30 hover:text-white/70 hover:bg-white/5 transition-colors disabled:opacity-20"
          title="Add speaker"
        >
          <Plus className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Speaker list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {state.speakers.map((speaker, i) => (
          <div key={speaker.id} className="relative group">
            <SpeakerCard speaker={speaker} index={i} />
            {state.speakers.length > 1 && (
              <button
                onClick={(e) => removeSpeaker(speaker.id, e)}
                className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-[#1d2044] text-white/40 hover:text-red-400 items-center justify-center hidden group-hover:flex transition-colors"
              >
                <Trash2 className="w-2.5 h-2.5" />
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Keyboard hint */}
      <div className="px-3 py-2 border-t border-[#1d2044] text-[10px] text-white/20 text-center">
        Press 1–{Math.min(9, state.speakers.length)} to switch speaker
      </div>

      {/* Stats footer */}
      {state.sessionStartTime && (
        <div className="px-3 py-2 border-t border-[#1d2044] space-y-1">
          <div className="flex justify-between text-[11px] text-white/30">
            <span>Segments</span>
            <span className="text-white/50">{state.segments.length}</span>
          </div>
          <div className="flex justify-between text-[11px] text-white/30">
            <span>Words</span>
            <span className="text-white/50">{totalWords}</span>
          </div>
          <div className="flex justify-between text-[11px] text-white/30">
            <span>Topics</span>
            <span className="text-white/50">{state.topics.length}</span>
          </div>
        </div>
      )}
    </aside>
  );
}
