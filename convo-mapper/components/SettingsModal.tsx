'use client';

import { useState } from 'react';
import { X, Eye, EyeOff, Key, Globe, Clock, Mic } from 'lucide-react';
import { useConversation } from '@/lib/store';
import { AppMode } from '@/lib/types';

const LANGUAGES = [
  { code: 'en-US', label: 'English (US)' },
  { code: 'en-GB', label: 'English (UK)' },
  { code: 'es-ES', label: 'Spanish' },
  { code: 'fr-FR', label: 'French' },
  { code: 'de-DE', label: 'German' },
  { code: 'it-IT', label: 'Italian' },
  { code: 'pt-BR', label: 'Portuguese (BR)' },
  { code: 'ja-JP', label: 'Japanese' },
  { code: 'zh-CN', label: 'Chinese (Simplified)' },
  { code: 'ko-KR', label: 'Korean' },
  { code: 'ar-SA', label: 'Arabic' },
  { code: 'hi-IN', label: 'Hindi' },
  { code: 'ru-RU', label: 'Russian' },
];

const TRANSLATION_LANGS = [
  { code: 'en', label: 'No translation (English)' },
  { code: 'es', label: 'Spanish' },
  { code: 'fr', label: 'French' },
  { code: 'de', label: 'German' },
  { code: 'it', label: 'Italian' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'ja', label: 'Japanese' },
  { code: 'zh', label: 'Chinese' },
  { code: 'ko', label: 'Korean' },
  { code: 'ar', label: 'Arabic' },
  { code: 'hi', label: 'Hindi' },
  { code: 'ru', label: 'Russian' },
];

export function SettingsModal() {
  const { state, dispatch } = useConversation();
  const [showKey, setShowKey] = useState(false);
  const [localKey, setLocalKey] = useState(state.apiKey);

  if (!state.showSettings) return null;

  const save = () => {
    dispatch({ type: 'SET_API_KEY', payload: localKey.trim() });
    dispatch({ type: 'SET_SHOW_SETTINGS', payload: false });
  };

  const isBrowserSupported =
    typeof window !== 'undefined' &&
    ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
      onClick={() => state.apiKey && dispatch({ type: 'SET_SHOW_SETTINGS', payload: false })}
    >
      <div
        className="bg-[#0c0d1e] border border-[#2d3264] rounded-2xl w-full max-w-lg shadow-2xl shadow-indigo-500/10"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-[#1d2044] flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-white">Settings</h2>
            <p className="text-xs text-white/30 mt-0.5">Configure ConvoMapper AI</p>
          </div>
          {state.apiKey && (
            <button
              onClick={() => dispatch({ type: 'SET_SHOW_SETTINGS', payload: false })}
              className="p-1.5 rounded-lg text-white/40 hover:text-white/80 hover:bg-white/5 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        <div className="px-6 py-5 space-y-5">
          {/* Browser warning */}
          {!isBrowserSupported && (
            <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-xs text-yellow-300">
              ⚠️ Speech recognition requires <strong>Chrome</strong> or <strong>Edge</strong>.
              Firefox and Safari are not supported.
            </div>
          )}

          {/* API Key */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-semibold text-white/50 mb-2">
              <Key className="w-3.5 h-3.5" />
              Anthropic API Key
            </label>
            <div className="relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={localKey}
                onChange={(e) => setLocalKey(e.target.value)}
                placeholder="sk-ant-..."
                className="w-full bg-[#06060f] border border-[#1d2044] focus:border-indigo-500 rounded-lg px-3 py-2.5 text-sm text-white placeholder-white/20 outline-none transition-colors pr-10 font-mono"
              />
              <button
                onClick={() => setShowKey(!showKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors"
              >
                {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <p className="text-[11px] text-white/20 mt-1.5">
              Your key is stored locally and sent only to the app&apos;s server-side proxy.
              Alternatively, set <code className="text-indigo-300 bg-indigo-500/10 px-1 rounded">ANTHROPIC_API_KEY</code> in <code className="text-indigo-300 bg-indigo-500/10 px-1 rounded">.env.local</code>.
            </p>
          </div>

          {/* Input Language */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-semibold text-white/50 mb-2">
              <Mic className="w-3.5 h-3.5" />
              Speech Input Language
            </label>
            <select
              value={state.inputLanguage}
              onChange={(e) => dispatch({ type: 'SET_INPUT_LANGUAGE', payload: e.target.value })}
              className="w-full bg-[#06060f] border border-[#1d2044] focus:border-indigo-500 rounded-lg px-3 py-2.5 text-sm text-white outline-none transition-colors"
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>

          {/* Translation */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-semibold text-white/50 mb-2">
              <Globe className="w-3.5 h-3.5" />
              Translation Output Language
            </label>
            <select
              value={state.outputLanguage}
              onChange={(e) => dispatch({ type: 'SET_OUTPUT_LANGUAGE', payload: e.target.value })}
              className="w-full bg-[#06060f] border border-[#1d2044] focus:border-indigo-500 rounded-lg px-3 py-2.5 text-sm text-white outline-none transition-colors"
            >
              {TRANSLATION_LANGS.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
            <p className="text-[11px] text-white/20 mt-1">
              Topic summaries will be translated if set (requires Translate mode or any mode).
            </p>
          </div>

          {/* Analysis Interval */}
          <div>
            <label className="flex items-center gap-1.5 text-xs font-semibold text-white/50 mb-2">
              <Clock className="w-3.5 h-3.5" />
              Auto-analysis Interval:{' '}
              <span className="text-indigo-400">{state.analysisInterval}s</span>
            </label>
            <input
              type="range"
              min={10}
              max={60}
              step={5}
              value={state.analysisInterval}
              onChange={(e) =>
                dispatch({ type: 'SET_ANALYSIS_INTERVAL', payload: Number(e.target.value) })
              }
              className="w-full accent-indigo-500"
            />
            <div className="flex justify-between text-[10px] text-white/20 mt-0.5">
              <span>10s (responsive)</span>
              <span>60s (efficient)</span>
            </div>
          </div>

          {/* Feature highlights */}
          <div className="grid grid-cols-2 gap-2 pt-1">
            {[
              { icon: '🎙️', text: 'Real-time speech recognition' },
              { icon: '👥', text: 'Multi-speaker tracking' },
              { icon: '🗺️', text: 'Live topic mapping' },
              { icon: '⚡', text: 'AI fact-checking' },
              { icon: '💡', text: 'Argument analysis' },
              { icon: '🌐', text: 'Live translation' },
              { icon: '📋', text: 'Session export (JSON)' },
              { icon: '⌨️', text: 'Keyboard speaker switch' },
            ].map(({ icon, text }) => (
              <div key={text} className="flex items-center gap-2 text-[11px] text-white/30">
                <span>{icon}</span>
                <span>{text}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[#1d2044] flex items-center justify-between">
          <p className="text-[11px] text-white/20">
            Powered by Claude claude-sonnet-4-6
          </p>
          <button
            onClick={save}
            disabled={!localKey.trim()}
            className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg text-sm font-semibold transition-colors"
          >
            Save & Continue
          </button>
        </div>
      </div>
    </div>
  );
}
