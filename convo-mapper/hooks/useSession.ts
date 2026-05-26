'use client';

import { useCallback, useEffect, useRef } from 'react';
import { useConversation } from '@/lib/store';
import { speechService } from '@/lib/speechService';
import { TopicNode } from '@/lib/types';

export function useSession() {
  const { state, dispatch } = useConversation();
  const analysisTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isAnalyzingRef = useRef(false);

  const runAnalysis = useCallback(async () => {
    if (isAnalyzingRef.current || !state.apiKey) return;

    const newSegments = state.segments.slice(state.lastAnalyzedCount);
    if (newSegments.length === 0) return;

    isAnalyzingRef.current = true;
    dispatch({ type: 'SET_ANALYZING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });

    try {
      const transcript = newSegments
        .map((s) => {
          const speaker = state.speakers.find((sp) => sp.id === s.speakerId);
          return `[${speaker?.name ?? s.speakerId}]: ${s.text}`;
        })
        .join('\n');

      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': state.apiKey },
        body: JSON.stringify({
          transcript,
          speakers: state.speakers,
          existingTopics: state.topics,
          mode: state.mode,
          outputLanguage: state.outputLanguage,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error ?? 'Analysis failed');
      }

      const data = await res.json();

      if (data.topics && Array.isArray(data.topics)) {
        const merged: TopicNode[] = [...state.topics];
        for (const incoming of data.topics as TopicNode[]) {
          const idx = merged.findIndex((t) => t.id === incoming.id);
          if (idx >= 0) {
            merged[idx] = { ...merged[idx], ...incoming, createdAt: merged[idx].createdAt, updatedAt: Date.now() };
          } else {
            merged.push({ ...incoming, createdAt: Date.now(), updatedAt: Date.now() });
          }
        }
        dispatch({ type: 'SET_TOPICS', payload: merged });
      }

      if (data.newInsights && Array.isArray(data.newInsights)) {
        dispatch({ type: 'SET_INSIGHTS', payload: [...state.insights, ...data.newInsights] });
      }

      if (data.factChecks && Array.isArray(data.factChecks)) {
        const withIds = data.factChecks.map((fc: object, i: number) => ({
          ...fc,
          id: `fc-${Date.now()}-${i}`,
        }));
        dispatch({ type: 'SET_FACT_CHECKS', payload: [...state.factChecks, ...withIds] });
      }

      if (data.summary) {
        dispatch({ type: 'SET_SUMMARY', payload: data.summary });
      }

      dispatch({ type: 'SET_LAST_ANALYZED_COUNT', payload: state.segments.length });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      dispatch({ type: 'SET_ERROR', payload: msg });
    } finally {
      isAnalyzingRef.current = false;
      dispatch({ type: 'SET_ANALYZING', payload: false });
    }
  }, [state, dispatch]);

  const startRecording = useCallback(() => {
    if (!state.apiKey) {
      dispatch({ type: 'SET_SHOW_SETTINGS', payload: true });
      return;
    }

    speechService.onResult = (text, isFinal) => {
      if (isFinal) {
        dispatch({
          type: 'ADD_SEGMENT',
          payload: {
            id: `seg-${Date.now()}-${Math.random().toString(36).slice(2)}`,
            speakerId: state.activeSpeakerId,
            text,
            timestamp: Date.now(),
            isFinal: true,
          },
        });
      } else {
        dispatch({ type: 'SET_PENDING_TEXT', payload: text });
      }
    };

    speechService.onError = (error) => dispatch({ type: 'SET_ERROR', payload: error });

    speechService.start(state.inputLanguage);
    dispatch({ type: 'START_RECORDING' });

    if (analysisTimerRef.current) clearInterval(analysisTimerRef.current);
    analysisTimerRef.current = setInterval(runAnalysis, state.analysisInterval * 1000);
  }, [state, dispatch, runAnalysis]);

  const stopRecording = useCallback(() => {
    speechService.stop();
    dispatch({ type: 'STOP_RECORDING' });
    if (analysisTimerRef.current) {
      clearInterval(analysisTimerRef.current);
      analysisTimerRef.current = null;
    }
    runAnalysis();
  }, [dispatch, runAnalysis]);

  const toggleRecording = useCallback(() => {
    if (state.isRecording) stopRecording();
    else startRecording();
  }, [state.isRecording, startRecording, stopRecording]);

  // Keyboard shortcuts: 1–9 select speakers
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const n = parseInt(e.key);
      if (n >= 1 && n <= 9) {
        const speaker = state.speakers[n - 1];
        if (speaker) dispatch({ type: 'SET_ACTIVE_SPEAKER', payload: speaker.id });
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [state.speakers, dispatch]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (analysisTimerRef.current) clearInterval(analysisTimerRef.current);
      speechService.stop();
    };
  }, []);

  return { toggleRecording, runAnalysis };
}
