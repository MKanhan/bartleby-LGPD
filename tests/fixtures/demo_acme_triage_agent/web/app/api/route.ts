import { openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";
import { generateText, tool } from "ai";
import { z } from "zod";

export const runtime = "nodejs";

const reviewerNotes = tool({
  description: "Append notes to an analyst review queue",
  parameters: z.object({
    claim_id: z.string(),
    notes: z.string(),
  }),
  execute: async ({ claim_id, notes }) => {
    const r = await fetch(`https://queue.acme.example/api/notes`, {
      method: "POST",
      body: JSON.stringify({ claim_id, notes }),
    });
    return await r.json();
  },
});

export async function POST(req: Request) {
  const { prompt } = await req.json();
  const { text } = await generateText({
    model: openai("gpt-5.4"),
    prompt,
    tools: { reviewerNotes },
  });
  return Response.json({ text });
}

export async function PUT(req: Request) {
  const { prompt } = await req.json();
  const { text } = await generateText({
    model: anthropic("claude-sonnet-4-6"),
    prompt,
  });
  return Response.json({ text });
}
