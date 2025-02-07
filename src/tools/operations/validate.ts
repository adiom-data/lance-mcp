import { chunksTable, chunksVectorStore } from "../../lancedb/client.js";
import { BaseTool, ToolParams } from "../base/tool.js";
import ollama from 'ollama';
import { sprintf } from 'sprintf-js';
import { split } from 'sentence-splitter';
import fs from 'fs'; // Import the fs module

interface Verdict {
  verdict: string;
  reason?: string;
}

interface Fact {
  fact: string;
  value: string;
  reason: string;
}

interface VerdictResponse {
  verdicts: Verdict[];
}

interface FactResponse {
  missed_facts: Fact[];
}

function extractVerdictsFromMarkdown(markdownString: string): Verdict[] {
  try {
      // Remove the markdown code block formatting
      const jsonString = markdownString
          .replace(/^```json\s*/, '') // Remove opening ```json
          .replace(/\s*```$/, '');    // Remove closing ```
      
      // Parse the JSON
      const parsed: VerdictResponse = JSON.parse(jsonString);
      
      return parsed.verdicts;
  } catch (error) {
      console.error('Error parsing JSON from markdown:', error);
      throw new Error('Failed to parse JSON from markdown string');
  }
}

function extractFactsFromMarkdown(markdownString: string): Fact[] {
  try {
      // Remove the markdown code block formatting
      const jsonString = markdownString
          .replace(/^```json\s*/, '') // Remove opening ```json
          .replace(/\s*```$/, '');    // Remove closing ```
      
      // Parse the JSON
      const parsed: FactResponse = JSON.parse(jsonString);
      
      return parsed.missed_facts;
  } catch (error) {
      console.error('Error parsing JSON from markdown:', error);
      throw new Error('Failed to parse JSON from markdown string');
  }
}

export interface ValidateParams extends ToolParams {
  output: string;
  prompt: string;
}

function logToFile(data: string) {
  const fileName = 'validate.log'
  fs.appendFile(fileName, data + '\n', (err) => {
    if (err) {
      console.error(`Failed to write to file ${fileName}:`, err);
    }
  });
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
      logToFile("CHECK CLAIM PROMPT: " + prompt + ". RESPONSE: " + response.message.content)

      return response.message.content === "Yes"
  }

  
  async checkOutputClaims(groundingDoc: string, output: string) {
    const sentences = JSON.parse(output)
    const checkPromises = sentences.map(sentence => this.checkClaim(groundingDoc, sentence));
    //await Promise.all(checkPromises);
    const checkResults = await Promise.all(checkPromises);
    const trueCount = checkResults.filter(result => result).length;
    const truePercentage = (trueCount / checkResults.length) * 100;
    logToFile(`CHECK CLAIM TOTAL: Percentage of claims supported: ${truePercentage}%`);

    return {yes: trueCount, no: checkResults.length - trueCount}
  }

    // checks the output for mistakes 
    async checkOutputCorrect(groundTruth: string, output: string) {
        const prompt = `Based on the given claims, which is a list of strings, generate a list of JSON objects to indicate whether EACH claim contradicts any facts in the retrieval context. The JSON will have 2 fields: 'verdict' and 'reason'.
The 'verdict' key should STRICTLY be either 'yes', 'no', or 'idk', which states whether the given claim agrees with the context. 
Provide a 'reason' ONLY if the answer is 'no'. 
The provided claim is drawn from the actual output. Try to provide a correction in the reason using the facts in the retrieval context.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects.
Example retrieval contexts: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
Example claims: ["Barack Obama is a caucasian male.", "Zurich is a city in London", "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.", "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect.", "Einstein was a Germen chef."]

Example:
{{
    "verdicts": [
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein won the Nobel Prize in 1969, which is untrue as the retrieval context states it is 1968 instead."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output claims Einstein is a Germen chef, which is not correct as the retrieval context states he was a German scientist instead."
        }},
    ]  
}}
===== END OF EXAMPLE ======

The length of 'verdicts' SHOULD BE STRICTLY EQUAL to that of claims.
You DON'T have to provide a reason if the answer is 'yes' or 'idk'.
ONLY provide a 'no' answer if the retrieval context DIRECTLY CONTRADICTS the claims. YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
Claims made using vague, suggestive, speculative language such as 'may have', 'possibility due to', does NOT count as a contradiction.
Claims that is not backed up due to a lack of information/is not mentioned in the retrieval contexts MUST be answered 'idk', otherwise I WILL DIE.
**

Retrieval Contexts:
${groundTruth}

Claims:
${output}
`
    
        const response = await ollama.chat({
            model: 'qwen2.5-coder',
            messages: [{ role: 'user', content: prompt }],
          })
          logToFile("CHECK CORRECT PROMPT: " + prompt + ". RESPONSE: " + response.message.content)

          const verdicts = extractVerdictsFromMarkdown(response.message.content);
          const verdictCounts = {
            yes: 0,
            no: 0,
            idk: 0,
          };

          verdicts.forEach(verdict => {
            if (verdict.verdict === 'yes') {
              verdictCounts.yes++;
            } else if (verdict.verdict === 'no') {
              verdictCounts.no++;
            } else if (verdict.verdict === 'idk') {
              verdictCounts.idk++;
            }
          });

          logToFile(`VERDICT COUNTS (${verdicts.length} total): Yes: ${verdictCounts.yes}, No: ${verdictCounts.no}, IDK: ${verdictCounts.idk}`);

          return verdictCounts
      }

    // checks the output for hallucinations 
    async checkOutputHallucinations(groundTruth: string, output: string) {
        const prompt = sprintf("Given the ground truth below, check output for any unsupported statements. Respond with a score representing accuracy percentage, followed by comments.\n Ground Truth: %s\nOutput: %s", groundTruth, output);
    
        const response = await ollama.chat({
            model: 'qwen2.5-coder',
            messages: [{ role: 'user', content: prompt }],
          })
          logToFile("CHECK HALLUCINATION PROMPT: " + prompt + ". RESPONSE: " + response.message.content)
      }

    // checks the output for omissions 
    async checkOutputOmissions(question: string, groundTruth: string, output: string) {
      const prompt = `Based on the original question and the retrieval context, generate a list of JSON objects to indicate important facts that were omitted in the provided list of facts, which is a list of strings. The JSON will have 3 fields: 'fact', 'value', and 'reason'.
      The 'fact' key should describe the fact from the retrieval context that was missed in the provided list of facts.  
      The 'value' key should STRICTLY be either 'high', 'medium', or 'low', which states how critical is the fact given the original question. 
      Provide a 'reason' to explain why the fact is important to the original question.
      
      **
      IMPORTANT: Please make sure to only return in JSON format, with the 'missed_facts' key as a list of JSON objects.
      Example retrieval contexts: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1968. Einstein is a German Scientist."
      Example question: "What did Einstein win the Nobel Prize for?"
      Example facts: ["Einstein won the Nobel Prize.", "Einstein won the Nobel Prize in 1968"]
      
      Example:
      {{
          "missed_facts": [
              {{
                  "fact": "Einstein won the Nobel Prize for his discovery of the photoelectric effect",
                  "value": "high",
                  "reason": "The original question is about the Nobel Prize, so it's important to mention why Einstein won it.",
              }},
              {{
                  "fact": "Einstein is a German Scientist.",
                  "value": "low",
                  "reason": "The original question is about the Nobel Prize, so it's not very important to mention where Einstein is from.",
              }}
          ]  
      }}
      ===== END OF EXAMPLE ======
      
      YOU SHOULD NEVER USE YOUR PRIOR KNOWLEDGE IN YOUR JUDGEMENT.
      **

      Original Question:
      ${question}
      
      Retrieval Contexts:
      ${groundTruth}
      
      Provided Facts:
      ${output}
      `    
        const response = await ollama.chat({
            model: 'qwen2.5-coder',
            messages: [{ role: 'user', content: prompt }],
          })
          logToFile("CHECK OMISSIONS: " + prompt + ". RESPONSE: " + response.message.content)

          const facts = extractFactsFromMarkdown(response.message.content);
          const factCounts = {
            missed: 0,
          };

          facts.forEach(verdict => {
            if (verdict.value === 'high' || verdict.value === 'medium') {
              factCounts.missed++;
            }
          });

          logToFile(`MISSED COUNTS : ${facts.length}`);

          return factCounts
      }
    
    async validate(prompt: string, groundingDoc: string, output: string) {
      logToFile("BEGINNING VALIDATION");
      const startCorrectM = Date.now();
      let resCorrectM = await this.checkOutputClaims(groundingDoc, output);
      const endCorrectM = Date.now();

      const startCorrect = Date.now();
      let resCorrect = await this.checkOutputCorrect(groundingDoc, output);
      const endCorrect = Date.now();

      // const startHalluc = Date.now();
      // await this.checkOutputHallucinations(groundingDoc, output);
      // const endHalluc = Date.now();

      const startHalluc = Date.now();
      let resHalluc = await this.checkOutputOmissions(prompt, groundingDoc, output);
      const endHalluc = Date.now();

      logToFile(`Time taken for checkOutputClaims: ${endCorrectM - startCorrectM}ms`);
      logToFile(`Time taken for checkOutputCorrect: ${endCorrect - startCorrect}ms`);
      logToFile(`Time taken for checkOutputOmissions: ${endHalluc - startHalluc}ms`);

      //regular LLM-based
      const tp = resCorrect.yes
      const fp = resCorrect.no + resCorrect.idk
      const fn = resHalluc.missed

      const f1 = tp / (tp + (fp + fn) / 2)

      //with mini-check for fact-checking
      const tpM = resCorrectM.yes
      const fpM = resCorrectM.no
      const f1M = tpM / (tpM + (fpM + fn) / 2)
      logToFile(`END VALIDATION. F1 score: ${f1} (with minicheck: ${f1M})`);
    }

  async execute(params: ValidateParams) {
    try {
      // const retriever = chunksVectorStore.asRetriever(10);
      // const results = await retriever.invoke(params.prompt);
      const results = await chunksTable.query().where("subject_id = 10000032").limit(10).toArray();

      // generate a combined grounding doc from each element
      // ignore duplicates and remove unneeded metadata
      let groundingDoc = "";
      const seenIds = new Set();
      for (let i = 0; i < results.length; i++) {
          if (seenIds.has(results[i].id)) {
            continue;
          }
          seenIds.add(results[i].id);
          // copy the results[i] object
          const resultCopy = { ...results[i] };
          // remove extra metadata
          delete resultCopy.vector;
          delete resultCopy.text;
          delete resultCopy.loc;
          groundingDoc += resultCopy.full_text + "\n";
      }

      this.validate(params.prompt, groundingDoc, params.output);

      return {
        content: [
          { type: "text" as const, text: groundingDoc },
        ],
        isError: false,
      };
    } catch (error) {
      return this.handleError(error);
    }
  }
}
