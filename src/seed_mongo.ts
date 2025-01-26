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
import { MongoClient } from 'mongodb';
import { UnionMode } from "@lancedb/lancedb/dist/arrow";

const argv: minimist.ParsedArgs = minimist(process.argv.slice(2),{boolean: "overwrite"});

const databaseDir = argv["dbpath"];
const connectionString = argv["connstr"];
const dbName = argv["db"];
const colName = argv["col"];
const overwrite = argv["overwrite"];

function validateArgs() {
    if (!databaseDir || !connectionString || !dbName || !colName) {
        console.error("Please provide a database path (--dbpath), a connection string (--connstr), db name (--db) and collection name (--col) to process");
        process.exit(1);
    }
    
    console.log("DATABASE PATH: ", databaseDir);
    console.log("CONNECTION STRING: ", connectionString);
    console.log("DB NAME: ", dbName);
    console.log("COLLECTION NAME: ", colName);
    console.log("OVERWRITE FLAG: ", overwrite);
}

const contentOverviewPromptTemplate = `Write a high-level one sentence database table content overview based on the sample entry:


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
  const query = catalogTable.query().where(`hash="${hash}"`).limit(1);
  const results = await query.toArray();
  return results.length > 0;
}

async function getRecordsFromMongoDB(connectionString: string, dbName: string, collectionName: string, limit: number = 1000):Promise<Document[]> {
    const client = new MongoClient(connectionString);
    try {
        await client.connect();
        const database = client.db(dbName);
        const collection = database.collection(collectionName);
        const pipeline = limit > 0 ? [{ $limit: limit }] : [{ $match: { } }];
        //const pipeline = [{ $sample: { size: count } }];
        const records = await collection.aggregate(pipeline).toArray();

        return records.map(record => {
            let text = record.text
            if (typeof text === 'string' && text.trim().length > 0) //there's a text field so we can use that
                delete record.text
            else 
                text = JSON.stringify(record) //no text field so we'll use the JSON object
            return new Document({ id: record._id.toString(), pageContent: text, metadata: { source: dbName + "." + collectionName} })
    });
    } finally {
        await client.close();
    }
}

async function getRandomRecordsFromMongoDB(connectionString: string, dbName: string, collectionName: string, count: number = 10) {
    const client = new MongoClient(connectionString);
    try {
        await client.connect();
        const database = client.db(dbName);
        const collection = database.collection(collectionName);
        const pipeline = [{ $sample: { size: count } }];
        const records = await collection.aggregate(pipeline).toArray();

        return records.map(record => {
            if (typeof record.text === 'string' && record.text.length > 100)
                record.text = record.text.substring(0,100)+"..."
            return new Document({ id: record._id.toString(), pageContent: JSON.stringify(record), metadata: { source: dbName + "." + collectionName}})});
    } finally {
        await client.close();
    }
}

const model = new Ollama({ model: defaults.SUMMARIZATION_MODEL });

// prepares documents for summarization
// returns already existing sources and new catalog records
async function processDocuments(rawDocs: any, catalogTable: lancedb.Table, skipExistsCheck: boolean) {
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
        const hash = crypto.createHash('sha256').update(source).digest('hex');

        // Check if a source document with the same hash already exists in the catalog
        const exists = skipExistsCheck ? false : await catalogRecordExists(catalogTable, hash);
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
    let catalogTableExists = true;
    let chunksTable : lancedb.Table;

    try {
        catalogTable = await db.openTable(defaults.CATALOG_TABLE_NAME);
    } catch (e) {
        console.error(`Looks like the catalog table "${defaults.CATALOG_TABLE_NAME}" doesn't exist. We'll create it later.`);
        catalogTableExists = false;
    }

    try {
        chunksTable = await db.openTable(defaults.CHUNKS_TABLE_NAME);
    } catch (e) {
        console.error(`Looks like the chunks table "${defaults.CHUNKS_TABLE_NAME}" doesn't exist. We'll create it later.`);
    }

    // try dropping the tables if we need to overwrite
    if (overwrite) {
        try {
            await db.dropTable(defaults.CATALOG_TABLE_NAME);
            await db.dropTable(defaults.CHUNKS_TABLE_NAME);
        } catch (e) {
            console.log("Error dropping tables. Maybe they don't exist!");
        }
    }

    // load files from the files path
    console.log("Loading files...")
    const rawRandomDocs = await getRandomRecordsFromMongoDB(connectionString, dbName, colName);
    const rawDocs = await getRecordsFromMongoDB(connectionString, dbName, colName);

    console.log("Loading LanceDB catalog store...")

    const { skipSources, catalogRecords } = await processDocuments(rawRandomDocs, catalogTable, overwrite || !catalogTableExists);

    let catalogStore
    if (overwrite) {
        catalogStore = await LanceDB.fromDocuments(catalogRecords, 
            new OllamaEmbeddings({model: defaults.EMBEDDING_MODEL}), 
            { uri: databaseDir, tableName: defaults.CATALOG_TABLE_NAME } as LanceDBArgs)
    } else {
        catalogStore = new LanceDB(new OllamaEmbeddings({model: defaults.EMBEDDING_MODEL}), { uri: databaseDir, table: catalogTable});
        if (catalogRecords.length > 0)
            await catalogStore.addDocuments(catalogRecords)
    }
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
    
    let vectorStore
    if (overwrite) {
        vectorStore = await LanceDB.fromDocuments(docs, 
                        new OllamaEmbeddings({model: defaults.EMBEDDING_MODEL}), 
                        { uri: databaseDir, tableName: defaults.CHUNKS_TABLE_NAME } as LanceDBArgs)
    } else {
        vectorStore = new LanceDB(new OllamaEmbeddings({model: defaults.EMBEDDING_MODEL}), { uri: databaseDir, table: chunksTable });
        if (docs.length > 0) {
            await vectorStore.addDocuments(docs)
        }
    }   

    console.log("Number of new chunks: ", docs.length);
    console.log(vectorStore);
}

seed();
