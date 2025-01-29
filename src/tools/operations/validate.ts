import { chunksVectorStore } from "../../lancedb/client.js";
import { BaseTool, ToolParams } from "../base/tool.js";

export interface ValidateParams extends ToolParams {
  output: string;
  prompt: string;
}

export class ValidateTool extends BaseTool<ValidateParams> {
  name = "validate";
  description = "Validate accuracy of model output for a particular prompt";
  inputSchema = {
    type: "object" as const,
    properties: {
        output: {
        type: "string",
        description: "Model output to validate",
        default: {},
      },
      prompt: {
        type: "string",
        description: "Prompt used to generate the output",
        default: {},
      },
    },
    required: ["output", "prompt"],
  };

  async execute(params: ValidateParams) {
    try {
      const retriever = chunksVectorStore.asRetriever(10);
      const results = await retriever.invoke(params.prompt);

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
