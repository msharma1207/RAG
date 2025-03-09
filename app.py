import os
import re
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint


# Load environment variables
load_dotenv()
huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_KEY")

if not huggingfacehub_api_token:
    raise ValueError("Hugging Face API key is missing. Set it in a .env file.")

app = Flask(__name__)

# Initialize the LLM
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    max_new_tokens=512,
    top_k=30,
    temperature=0.1,
    repetition_penalty=1.03
)

# Load documents
with open("data/documents.txt", "r", encoding="utf-8") as file:
    full_text = file.read()

# Text splitting and embedding
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(full_text)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(texts, embeddings)
retriever = db.as_retriever()

# RAG chatbot function
def chat_with_rag(message):
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(message)

@app.route('/')
def home():
    return render_template('bot_1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_input']
    bot_message = chat_with_rag(user_message)

    # Extract the answer using regex
    pattern = r"Answer:\s*(.*)"
    match = re.search(pattern, bot_message, re.DOTALL)

    if match:
        answer = match.group(1).strip()
        return jsonify({'response': answer})
    else:
        return jsonify({'response': "I couldn't find a relevant answer in the context."})

if __name__ == '__main__':
    app.run(debug=True)