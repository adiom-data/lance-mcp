import * as lancedb from "@lancedb/lancedb";
import minimist from 'minimist';
import {
  RecursiveCharacterTextSplitter
} from 'langchain/text_splitter';
import {
  DirectoryLoader
} from 'langchain/document_loaders/fs/directory';
import {
  LanceDB, LanceDBArgs
} from "@langchain/community/vectorstores/lancedb";
import { Document } from "@langchain/core/documents";
import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import * as fs from 'fs';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { loadSummarizationChain } from "langchain/chains";
import { BaseLanguageModelInterface, BaseLanguageModelCallOptions } from "@langchain/core/language_models/base";
import { PromptTemplate } from "@langchain/core/prompts";
import * as crypto from 'crypto';
import * as defaults from './config'

const argv: minimist.ParsedArgs = minimist(process.argv.slice(2),{boolean: "overwrite"});

const databaseDir = argv["dbpath"];
const filesDir = argv["filesdir"];
const overwrite = argv["overwrite"];

function validateArgs() {
    if (!databaseDir || !filesDir) {
        console.error("Please provide a database path (--dbpath) and a directory with files (--filesdir) to process");
        process.exit(1);
    }
    
    console.log("DATABASE PATH: ", databaseDir);
    console.log("FILES DIRECTORY: ", filesDir);
    console.log("OVERWRITE FLAG: ", overwrite);
}

const contentOverviewPromptTemplate = `Write a high-level one sentence content overview based on the text below:


"{text}"


WRITE THE CONTENT OVERVIEW ONLY, DO NOT WRITE ANYTHING ELSE:`;


const contentOverviewPrompt = new PromptTemplate({
  template: contentOverviewPromptTemplate,
  inputVariables: ["text"],
});

async function generateContentOverview(rawDocs: any, model: BaseLanguageModelInterface<any, BaseLanguageModelCallOptions>) {
  // This convenience function creates a document chain prompted to summarize a set of documents.
  const chain = loadSummarizationChain(model, { type: "map_reduce", combinePrompt: contentOverviewPrompt});
  const res = await chain.invoke({
    input_documents: rawDocs,
  });

  return res;
}

async function catalogRecordExists(catalogTable: lancedb.Table, hash: string): Promise<boolean> {
  if (overwrite) { // nothing exists if we are in the overwrite mode //TODO: this should probably be replaced by checking if the table exists
    return false;
  }

  const query = catalogTable.query().where(`hash="${hash}"`).limit(1);
  const results = await query.toArray();
  return results.length > 0;
}

const directoryLoader = new DirectoryLoader(
  filesDir,
  {
   ".pdf": (path: string) => new PDFLoader(path),
  },
);

const model = new Ollama({ model: defaults.SUMMARIZATION_MODEL });

// prepares documents for summarization
// returns already existing sources and new catalog records
async function processDocuments(rawDocs: any, catalogTable: lancedb.Table) {
    // group rawDocs by source for further processing
    const docsBySource = rawDocs.reduce((acc: Record<string, any[]>, doc: any) => {
        const source = doc.metadata.source;
        if (!acc[source]) {
            acc[source] = [];
        }
        acc[source].push(doc);
        return acc;
    }, {});

    let skipSources = [];
    let catalogRecords = [];

    // iterate over individual sources and get their summaries
    for (const [source, docs] of Object.entries(docsBySource)) {
        // Calculate hash of the source document
        const fileContent = await fs.promises.readFile(source);
        const hash = crypto.createHash('sha256').update(fileContent).digest('hex');

        // Check if a source document with the same hash already exists in the catalog
        const exists = await catalogRecordExists(catalogTable, hash);
        if (exists) {
            console.log(`Document with hash ${hash} already exists in the catalog. Skipping...`);
            skipSources.push(source);
        } else {
            const contentOverview = await generateContentOverview(docs, model);
            console.log(`Content overview for source ${source}:`, contentOverview);
            catalogRecords.push(new Document({ pageContent: contentOverview["text"], metadata: { source, hash } }));
        }
    }

    return { skipSources, catalogRecords };
}   

async function seed() {
    validateArgs();

    const db = await lancedb.connect(databaseDir);

    let catalogTable : lancedb.Table;
    let chunksTable : lancedb.Table;

    catalogTable = await db.openTable(defaults.CATALOG_TABLE_NAME);
    chunksTable = await db.openTable(defaults.CHUNKS_TABLE_NAME);

    // try dropping the tables if we need to overwrite
    if (overwrite) {
        try {
            await db.dropTable("catalog");
            await db.dropTable("langchain");
        } catch (e) {
            console.log("Error dropping tables. Maybe they don't exist!");
        }
    }

    // load files from the files path
    console.log("Loading files...")
    const rawDocs = await directoryLoader.load();

    // overwrite the metadata as large metadata can give lancedb problems
    for (const doc of rawDocs) {
        doc.metadata = { loc: doc.metadata.loc, source: doc.metadata.source };
    }

    console.log("Loading LanceDB catalog store...")

    const { skipSources, catalogRecords } = await processDocuments(rawDocs, catalogTable);
    const catalogStore = catalogRecords.length > 0 ? 
        await LanceDB.fromDocuments(catalogRecords, 
            new OllamaEmbeddings({model: defaults.EMBEDDING_MODEL}), 
            { mode: overwrite ? "overwrite" : undefined, uri: databaseDir, tableName: defaults.CATALOG_TABLE_NAME } as LanceDBArgs) :
        new LanceDB(new OllamaEmbeddings({model: defaults.EMBEDDING_MODEL}), { uri: databaseDir, table: catalogTable});
    console.log(catalogStore);

    console.log("Number of new catalog records: ", catalogRecords.length);
    console.log("Number of skipped sources: ", skipSources.length);
    //remove skipped sources from rawDocs
    const filteredRawDocs = rawDocs.filter((doc: any) => !skipSources.includes(doc.metadata.source));

    console.log("Loading LanceDB vector store...")
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 10,
      });
    const docs = await splitter.splitDocuments(filteredRawDocs);
    
    const vectorStore = docs.length > 0 ? 
        await LanceDB.fromDocuments(docs, 
        new OllamaEmbeddings({model: defaults.EMBEDDING_MODEL}), 
        { mode: overwrite ? "overwrite" : undefined, uri: databaseDir, tableName: defaults.CHUNKS_TABLE_NAME } as LanceDBArgs) :
        new LanceDB(new OllamaEmbeddings({model: defaults.EMBEDDING_MODEL}), { uri: databaseDir, table: chunksTable });

    console.log("Number of new chunks: ", docs.length);
    console.log(vectorStore);
}

seed();
