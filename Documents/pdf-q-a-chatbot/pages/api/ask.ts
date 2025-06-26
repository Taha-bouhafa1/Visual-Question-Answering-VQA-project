import type { NextApiRequest, NextApiResponse } from "next";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import {HuggingFaceTransformersEmbeddings} from "@langchain/community/dist/embeddings/huggingface_transformers";
import * as dotenv from "dotenv";

dotenv.config();

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_API_KEY = process.env.GROQ_API_KEY;

export const config = {
  api: {
    bodyParser: true,
  },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Only POST method is allowed." });
  }

  try {
    const { question, collectionName } = req.body;

    if (!question || !collectionName) {
      return res.status(400).json({ error: "Missing question or collectionName." });
    }

    if (!GROQ_API_KEY) {
      return res.status(500).json({ error: "GROQ_API_KEY is not set in environment variables." });
    }

    const embeddings = new HuggingFaceTransformersEmbeddings({
      modelName: "sentence-transformers/all-MiniLM-L6-v2",
    });

    const vectorStore = new Chroma({
      collectionName,
      embeddings,
      url: "http://localhost:8000", // adjust if needed
    });

    const results = await vectorStore.similaritySearch(question, 3);
    const contextText = results.map((doc) => doc.pageContent).join("\n---\n");

    const messages = [
      {
        role: "system",
        content: "You are a helpful assistant. Use the provided context to answer the user's question.",
      },
      {
        role: "user",
        content: `Context:\n${contextText}\n\nQuestion: ${question}`,
      },
    ];

    const groqResponse = await fetch(GROQ_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: "meta-llama/llama-4-scout-17b-16e-instruct",
        messages,
        temperature: 0.3,
        max_tokens: 512,
      }),
    });

    if (!groqResponse.ok) {
      const errorText = await groqResponse.text();
      console.error("Groq API error:", errorText);
      return res.status(500).json({ error: "Groq API call failed." });
    }

    const data = await groqResponse.json();
    const answer = data.choices?.[0]?.message?.content || "Sorry, I couldn’t generate a response.";

    return res.status(200).json({ answer });
  } catch (error) {
    console.error("Error in /api/ask:", error);
    return res.status(500).json({ error: "Internal server error." });
  }
}
