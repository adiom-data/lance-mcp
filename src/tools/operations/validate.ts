import { chunksVectorStore } from "../../lancedb/client.js";
import { BaseTool, ToolParams } from "../base/tool.js";
import ollama from 'ollama';
import { sprintf } from 'sprintf-js';
import { split } from 'sentence-splitter';

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

  // checks if the claim is supported by the document by calling bespoke-minicheck via Ollama
  async check(document: string, claim: string) {
    const prompt = sprintf("Document: %s\nClaim: %s", document, claim);

    const response = await ollama.chat({
        model: 'bespoke-minicheck',
        messages: [{ role: 'user', content: prompt }],
      })
      console.error("CHECK PROMPT: " + prompt + ". RESPONSE: " + response.message.content)
  }

  async execute(params: ValidateParams) {
    try {
      const retriever = chunksVectorStore.asRetriever(10);
      const results = await retriever.invoke(params.prompt);

      // generate a combined grounding doc from each element in results.pageContent
      let groundingDoc = "";
      for (let i = 0; i < results.length; i++) {
          groundingDoc += results[i].pageContent + "\n";
      }

      const sentences = split(params.output).filter(sentence => sentence.type != "WhiteSpace").map(sentence => sentence.raw);
      const checkPromises = sentences.map(sentence => this.check(groundingDoc, sentence));
      await Promise.all(checkPromises);
      
      //await this.check(groundingDoc, params.output);

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
