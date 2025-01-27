import { catalogTable, catalogVectorStore, chunksVectorStore } from "../../lancedb/client.js";
import { BaseTool, ToolParams } from "../base/tool.js";

export interface CatalogSearchParams extends ToolParams {
  text: string;
}

export class CatalogSearchTool extends BaseTool<CatalogSearchParams> {
  name = "catalog_search";
  description = "Search for relevant data sources in the catalog";
  inputSchema = {
    type: "object" as const,
    properties: {
      text: {
        type: "string",
        description: "Search string",
        default: {},
      },
    },
    required: ["text"],
  };

  async execute(params: CatalogSearchParams) {
    try {
      const retriever = catalogVectorStore.asRetriever({});
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
