import os
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

import pinecone
import textwrap


config = {'max_new_tokens': 8000, 'temperature': 0}
llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML', config=config)

template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
<</SYS>>

Answer the question below from context below :
{context}
{question} [/INST]
"""

loader = PyPDFLoader("attention.pdf")
pdf = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=0)
docs = text_splitter.split_documents(pdf)


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

pinecone.init(
    api_key="12d01438-407e-48d5-bef5-7493ea86daca", environment="gcp-starter"
)

docsearch = Pinecone.from_existing_index("langchain-self-retriever-demo", hf)




retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)



def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))


# this might take longer 
query ="explain Multi head attention to me "
llm_response = qa_chain(query)
process_llm_response(llm_response)