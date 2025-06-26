import fs from "fs";
import pdfParse from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { HfInference } from "@huggingface/inference";
import dotenv from "dotenv";

dotenv.config({ path: ".env.local" });
const hfApiKey = process.env.HUGGINGFACE_API_KEY;
if (!hfApiKey) {
  throw new Error("Missing HUGGINGFACE_API_KEY in environment variables");
}

const hfClient = new HfInference(hfApiKey);

export const embedPdfToChroma = async (pdfPath: string, collectionName: string) => {
  console.log("📄 Loading PDF text with pdf-parse...");
  const dataBuffer = fs.readFileSync(pdfPath);
  const data = await pdfParse(dataBuffer);
  const rawText = data.text;

  console.log("Raw PDF text length:", rawText.length);
  console.log("📚 Splitting PDF text into chunks...");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });

  const docs = await splitter.splitDocuments([{ pageContent: rawText, metadata: {} }]);

  console.log("⚙️ Creating embeddings...");
  const embeddings = new HuggingFaceInferenceEmbeddings({
    client: hfClient,
    model: "sentence-transformers/all-MiniLM-L6-v2",
  });

  console.log("💾 Storing embeddings in Chroma vector DB...");
  const vectorStore = await Chroma.fromDocuments(docs, embeddings, {
    collectionName,
    url: "http://localhost:8000",
  });

  console.log("✅ Embedding complete. Ready for querying.");
  return vectorStore;
};

if (require.main === module) {
  (async () => {
    try {
      const pdfPath = "public/pdfs/rag_paper.pdf";
      const collectionName = "my-pdf-collection";

      await embedPdfToChroma(pdfPath, collectionName);
      console.log("🎉 Done embedding PDF.");
    } catch (error) {
      console.error("❌ Error embedding PDF:", error);
    }
  })();
}
