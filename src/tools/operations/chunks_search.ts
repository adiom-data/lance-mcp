import { chunksTable, chunksVectorStore } from "../../lancedb/client.js";
import { BaseTool, ToolParams } from "../base/tool.js";

export interface ChunksSearchParams extends ToolParams {
  text: string;
  source: string;
  subject_id?: string;
}

export class ChunksSearchTool extends BaseTool<ChunksSearchParams> {
  name = "chunks_search";
  description = "Search for relevant records chunks in the vector store based on a data source from the catalog";
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
        description: "Specific data source (or a comma-separated list) to filter the search",
        default: {},
      },
      subject_id: {
        type: "string",
        description: "Specific subject id to filter the search",
        default: {},
      },
    },
    required: ["text", "source"],
  };

  async execute(params: ChunksSearchParams) {
    try {
      const retriever = chunksVectorStore.asRetriever(10, params.subject_id ? "metadata.subject_id="+params.subject_id : undefined);
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
