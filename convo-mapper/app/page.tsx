'use client';

import { ConversationProvider } from '@/lib/store';
import { Header } from '@/components/Header';
import { SpeakerPanel } from '@/components/SpeakerPanel';
import { TopicMap } from '@/components/TopicMap';
import { TranscriptFeed } from '@/components/TranscriptFeed';
import { InsightsPanel } from '@/components/InsightsPanel';
import { SettingsModal } from '@/components/SettingsModal';
import { TopicModal } from '@/components/TopicModal';

export default function Home() {
  return (
    <ConversationProvider>
      <div className="h-screen flex flex-col overflow-hidden">
        <Header />
        <div className="flex flex-1 overflow-hidden">
          <SpeakerPanel />
          <TopicMap />
          <div className="flex flex-col w-[310px] min-w-[310px] border-l border-[#1d2044]">
            <TranscriptFeed />
            <InsightsPanel />
          </div>
        </div>
        <SettingsModal />
        <TopicModal />
      </div>
    </ConversationProvider>
  );
}
