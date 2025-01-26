import { chunksVectorStore } from "../../lancedb/client.js";
import { BaseTool, ToolParams } from "../base/tool.js";

export interface ChunksSearchParams extends ToolParams {
  text: string;
  source: string;
}

export class ChunksSearchTool extends BaseTool<ChunksSearchParams> {
  name = "chunks_search";
  description = "Search for relevant document chunks in the vector store based on a source document from the catalog";
  inputSchema = {
    type: "object" as const,
    properties: {
      text: {
        type: "string",
        description: "Search string",
        default: {},
      },
      source: {
        type: "string",
        description: "Source document to filter the search",
        default: {},
      },
    },
    required: ["text", "source"],
  };

  async execute(params: ChunksSearchParams) {
    try {
      const retriever = chunksVectorStore.asRetriever(10);
      const results = await retriever.invoke(params.text);

      // Filter results by source if provided
      // TODO: this needs to be pushed down to LanceDB
      if (params.source) {
        let sourceList = params.source.split(/\s*,\s*/);

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                results.filter((result: any) => sourceList.indexOf(result.metadata.source) > -1),
                null,
                2
              ),
            },
          ],
          isError: false,
        };
      }

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
