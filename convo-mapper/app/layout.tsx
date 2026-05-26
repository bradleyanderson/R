import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ConvoMapper — AI Conversation Intelligence',
  description:
    'Real-time AI-powered conversation mapping. Track speakers, arguments, topics, and insights as you talk.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="h-screen overflow-hidden bg-[#06060f] text-slate-200">{children}</body>
    </html>
  );
}
