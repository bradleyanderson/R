import Anthropic from '@anthropic-ai/sdk';
import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { transcript, speakers, existingTopics, mode, outputLanguage } = body;

  const apiKey = req.headers.get('x-api-key') || process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: 'No API key provided' }, { status: 401 });
  }

  const client = new Anthropic({ apiKey });

  const speakerList = (speakers as { id: string; name: string }[])
    .map((s) => `  - id="${s.id}" name="${s.name}"`)
    .join('\n');

  const existingTopicsSummary = (existingTopics as { id: string; title: string; summary: string }[])
    .map((t) => `  - id="${t.id}" title="${t.title}"`)
    .join('\n');

  const translationNote =
    outputLanguage && outputLanguage !== 'en'
      ? `Also provide a "translationSummary" field (string) for each topic with the summary translated to ${outputLanguage}.`
      : '';

  const systemPrompt = `You are an expert conversation analyst. You extract structured insights from spoken conversations in real time.
Always return valid JSON only — no markdown, no explanation, just the JSON object.
Be concise, insightful, and focus on what's genuinely interesting, surprising, or important.`;

  const userPrompt = `## Conversation Analysis Request

Mode: ${mode}
Speakers:
${speakerList}

## New Transcript
${transcript || '(no new content yet)'}

## Already-Known Topics (update these if relevant new info appears, add new ones as needed)
${existingTopicsSummary || '(none yet)'}

## Return this exact JSON structure:
{
  "topics": [
    {
      "id": "kebab-case-unique-id",
      "title": "Short Topic Title",
      "category": "science|politics|personal|technical|philosophical|economic|other",
      "summary": "2–3 sentence description of what's being discussed",
      "arguments": [
        { "speakerId": "speaker-id-from-list", "position": "for|against|neutral", "text": "Key argument in 1 sentence" }
      ],
      "relatedFacts": ["Relevant real-world fact or context", "Another interesting fact"],
      "aiInsights": ["Non-obvious observation about this part of the conversation"],
      "sentiment": "positive|negative|neutral|debate",
      "keywords": ["keyword1", "keyword2"],
      "notableQuotes": ["exact or near-exact quote if notable"],
      "isNew": true,
      "updated": false
      ${translationNote ? ', "translationSummary": "translated summary here"' : ''}
    }
  ],
  "newInsights": [
    "High-level observation about the overall conversation"
  ],
  "factChecks": [
    {
      "claim": "Specific verifiable claim made",
      "speakerId": "speaker-id",
      "status": "likely-true|uncertain|potentially-false|unverifiable",
      "explanation": "1 sentence explanation"
    }
  ],
  "summary": "2–3 sentence overall summary of the conversation so far"
}

Rules:
- Only include topics actually mentioned in the new transcript
- For existing topic IDs, set "isNew": false and "updated": true
- For brand new topics, set "isNew": true and "updated": false
- Keep arguments tied to specific speaker IDs from the speaker list above
- Facts should be real-world context, not just restatements
- Return only the JSON object`;

  try {
    const message = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 2500,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
    });

    const raw = message.content[0].type === 'text' ? message.content[0].text : '';
    const cleaned = raw.replace(/^```json\s*/m, '').replace(/^```\s*/m, '').replace(/```\s*$/m, '').trim();

    let result;
    try {
      result = JSON.parse(cleaned);
    } catch {
      return NextResponse.json({ error: 'Failed to parse AI response', raw }, { status: 500 });
    }

    return NextResponse.json(result);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
