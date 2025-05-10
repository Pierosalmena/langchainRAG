#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from pathlib import Path
import xml.etree.ElementTree as ET

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.base import Embeddings

from llama_cpp import Llama

class LlamaEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        # Initialize local LLaMA model for embeddings
        self.client = Llama(model_path=model_path)

    def embed_documents(self, texts):
        # Returns list of embedding vectors for document texts
        return [self.client.embed(text).embeddings for text in texts]

    def embed_query(self, text):
        # Returns a single embedding vector for query text
        return self.client.embed(text).embeddings

def load_items_from_xml(xml_path: str):
    """
    Parse the XML file to extract items and their combined query text.
    Returns the XML tree and a list of tuples (element, query_text).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    items = []
    for item in root.findall('.//item'):
        name = item.find('name').text or ''
        text = item.find('text').text or ''
        query = f"{name} {text}".strip()
        items.append((item, query))
    return tree, items

def enrich_items_with_rag(tree: ET.ElementTree, items, qa_chain, output_xml_path: str):
    """
    Enrich each XML item element with RAG match results and save the new XML.
    """
    for elem, query in items:
        result = qa_chain({'query': query})
        answer = result['result']
        sources = [doc.metadata.get('source', '') for doc in result['source_documents']]

        rag = ET.SubElement(elem, 'rag_match')
        match_text = ET.SubElement(rag, 'match_text')
        match_text.text = answer
        rag_sources = ET.SubElement(rag, 'sources')
        for src in sources:
            s = ET.SubElement(rag_sources, 'source')
            s.text = src

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

def main():
    # Load environment variables from .env
    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('Missing OPENAI_API_KEY in your environment or .env file.')

    # Paths (relative to where you run this script)
    pdf_path = 'service-specification.pdf'
    xml_input = 'output.xml'
    xml_output = 'enriched_output.xml'
    llama_model_path = os.environ.get('LLAMA_MODEL_PATH', '/path/to/llama-2-7b.gguf')

    # 1. Load and chunk the PDF
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 2. Create embeddings and vectorstore using local LLaMA embeddings
    embedding_model = LlamaEmbeddings(model_path=llama_model_path)
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # 3. Set up retriever and RAG chain
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # 4. Load XML items and enrich
    tree, items = load_items_from_xml(xml_input)
    enrich_items_with_rag(tree, items, qa_chain, xml_output)

    print(f"Enriched XML saved to {xml_output}")

if __name__ == '__main__':
    main()
