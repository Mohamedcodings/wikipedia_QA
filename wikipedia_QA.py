import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from datasets import load_dataset

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not set. Please set it as an environment variable.")

# Load and preprocess the dataset
dataset = load_dataset("wikipedia", "20220301.simple", trust_remote_code=True)

def preprocess_article(article, max_length=512):
    return article['text'][:max_length]

articles = [preprocess_article(article) for article in dataset['train'].select(range(100))]

# Create embeddings and index using FAISS
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vectorstore = FAISS.from_texts(articles, embeddings)

# Set up the retriever
retriever = vectorstore.as_retriever()

# Load the ChatOpenAI model
llm = ChatOpenAI(
    api_key=openai_api_key,
    model_name="gpt-3.5-turbo",  # Use the chat model
    temperature=0.5,  # Control response randomness
    max_tokens=400,  # Set maximum tokens for the response
)

# Create the Conversational Retrieval QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, return_source_documents=False
)

# Define a question and stylistic hints in the prompt
question = "What is the capital of Morocco?"
tone = "Respond as if you are talking with Mohamed, be friendly: "
full_prompt = tone + question

# Get the answer from the chain
answer = qa_chain.invoke({"question": full_prompt, "chat_history": []})

# Print the response
print(f"\n\n{question}")
print(answer['answer'])
