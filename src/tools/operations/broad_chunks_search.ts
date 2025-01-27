import { chunksVectorStore } from "../../lancedb/client.js";
import { BaseTool, ToolParams } from "../base/tool.js";

export interface BroadSearchParams extends ToolParams {
  text: string;
  subject_id?: string;
}

export class BroadSearchTool extends BaseTool<BroadSearchParams> {
  name = "all_chunks_search";
  description = "Search for relevant record chunks in the vector store across all data sources. Use with caution as it can return information from irrelevant sources";
  inputSchema = {
    type: "object" as const,
    properties: {
      text: {
        type: "string",
        description: "Search string",
        default: {},
      },
      subject_id: {
        type: "string",
        description: "Specific subject id to filter the search",
        default: {},
      },
    },
    required: ["text"],
  };

  async execute(params: BroadSearchParams) {
    try {
      const retriever = chunksVectorStore.asRetriever(10, params.subject_id ? "metadata.subject_id="+params.subject_id : undefined);
      const results = await retriever.invoke(params.text);

      return {
        content: [
          { type: "text" as const, text: JSON.stringify(results, null, 2) },
        ],
        isError: false,
      };
    } catch (error) {
      return this.handleError(error);
    }
  }
}
