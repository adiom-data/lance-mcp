import { BaseTool } from "./base/tool.js";
import { BroadSearchTool } from "./operations/broad_chunks_search.js";
import { CatalogSearchTool } from "./operations/catalog_search.js";
import { ChunksSearchTool } from "./operations/chunks_search.js";
import { McpError, ErrorCode, Tool } from "@modelcontextprotocol/sdk/types.js";
import { ValidateTool } from "./operations/validate.js";

export class ToolRegistry {
  private tools: Map<string, BaseTool<any>> = new Map();

  constructor() {
    this.registerTool(new ChunksSearchTool());
    this.registerTool(new CatalogSearchTool());
    this.registerTool(new BroadSearchTool());
    this.registerTool(new ValidateTool());
  }

  registerTool(tool: BaseTool<any>) {
    this.tools.set(tool.name, tool);
  }

  getTool(name: string): BaseTool<any> | undefined {
    const tool = this.tools.get(name);
    if (!tool) {
      throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
    }
    return tool;
  }

  getAllTools(): BaseTool<any>[] {
    return Array.from(this.tools.values());
  }

  getToolSchemas(): Tool[] {
    return this.getAllTools().map((tool) => {
      const inputSchema = tool.inputSchema as any;
      return {
        name: tool.name,
        description: tool.description,
        inputSchema: {
          type: "object",
          properties: inputSchema.properties || {},
          ...(inputSchema.required && { required: inputSchema.required }),
        },
      };
    });
  }
}
