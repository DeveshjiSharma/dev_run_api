# 7th
from flask import Flask, request, jsonify
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(_name_)

def get_conversational_chain():
    prompt_template = """
    'INPUT TEXT':
        {context}
        Question:
        {question}
    PROMPT: Your role is a ayurvedic doctor bot SwastVeda now what you have to do is analyze the given text given that is delimited by text 
    'context' 4and analyze the text delimited by 'Question'.

Now context contains the information from which you need to generate the answer to this prompt

questions contains the answers to following question:
    "What symptoms are you experiencing?",
    "How long have you been experiencing these symptoms?",
    "Have you experienced any recent changes in your health or lifestyle?",
    "Do you have any medical conditions or are you taking any medications?",
    "Have you had any recent injuries or been exposed to any potential hazards?"

Now depending on the answers provided by the patient to its respective doctor's question you need to analyze the context and diagnosis the 
disease and also provide the medicines with dosage that are mentioned in the diagnosis and preventions that patient should take to recover.

IMP: In RESPONSES question one question is 'How long have you been experiencing these symptoms?" In answer if patient is mentioned that he 
has facing the issue from more than one month then suggest the medicines and preventions but also say user that 'He should consider to go for doctor from our app as patient is facing symptoms for more than one month'.

    
    Now according to the disease that you identified from question provide the user correct medicines with proper dosage and timining to 
    consume it. Also provide the do's, dont's and preventions that user must take to recover from the disease.

    'Don't say you dont have any idea like that'
    """

    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    model = ChatGoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


@app.route('/api/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "Question parameter is missing"}), 400
    response_text = user_input(user_question)
    return jsonify({"response": response_text})

if _name_ == '_main_':
    app.run(debug=True,host="0.0.0.0",port=5000)
