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
  async checkClaim(document: string, claim: string) {
    const prompt = sprintf("Document: %s\nClaim: %s", document, claim);

    const response = await ollama.chat({
        model: 'bespoke-minicheck',
        messages: [{ role: 'user', content: prompt }],
      })
      console.error("CHECK CLAIM PROMPT: " + prompt + ". RESPONSE: " + response.message.content)

      return response.message.content === "Yes"
  }

  async checkOutputClaims(groundingDoc: string, output: string) {
    const sentences = split(output).filter(sentence => sentence.type != "WhiteSpace").map(sentence => sentence.raw);
    const checkPromises = sentences.map(sentence => this.checkClaim(groundingDoc, sentence));
    //await Promise.all(checkPromises);
    const checkResults = await Promise.all(checkPromises);
    const trueCount = checkResults.filter(result => result).length;
    const truePercentage = (trueCount / checkResults.length) * 100;
    console.error(`CHECK CLAIM TOTAL: Percentage of claims supported: ${truePercentage}%`);
  }

    // checks the output for mistakes 
    async checkOutputCorrect(groundTruth: string, output: string) {
        const prompt = sprintf("Given the ground truth below, check output for any inaccurate statements. Respond with a score representing accuracy percentage, followed by comments.\n Ground Truth: %s\nOutput: %s", groundTruth, output);
    
        const response = await ollama.chat({
            model: 'qwen2.5-coder',
            messages: [{ role: 'user', content: prompt }],
          })
          console.error("CHECK CORRECT PROMPT: " + prompt + ". RESPONSE: " + response.message.content)
      }

    // checks the output for hallucinations 
    async checkOutputHallucinations(groundTruth: string, output: string) {
        const prompt = sprintf("Given the ground truth below, check output for any unsupported statements. Respond with a score representing accuracy percentage, followed by comments.\n Ground Truth: %s\nOutput: %s", groundTruth, output);
    
        const response = await ollama.chat({
            model: 'qwen2.5-coder',
            messages: [{ role: 'user', content: prompt }],
          })
          console.error("CHECK HALLUCINATION PROMPT: " + prompt + ". RESPONSE: " + response.message.content)
      }

    // checks the output for omissions 
    async checkOutputOmissions(question: string, groundTruth: string, output: string) {
        const prompt = sprintf("Given the ground truth below, check output for omissions of any information or facts that might be important or relevant to the original question. Respond with a score representing completeness percentage, followed by comments.\n Question: %s\nGround Truth: %s\nOutput: %s", question, groundTruth, output);
    
        const response = await ollama.chat({
            model: 'qwen2.5-coder',
            messages: [{ role: 'user', content: prompt }],
          })
          console.error("CHECK OMISSIONS: " + prompt + ". RESPONSE: " + response.message.content)
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

      this.checkOutputClaims(groundingDoc, params.output);
      this.checkOutputCorrect(groundingDoc, params.output);
      this.checkOutputHallucinations(groundingDoc, params.output);
      this.checkOutputOmissions(params.prompt, groundingDoc, params.output);
      
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
