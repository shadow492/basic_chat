import streamlit as st

import os
from dotenv import load_dotenv
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from langchain import PromptTemplate,LLMChain
st.set_page_config(page_title="Travel Guru")
st.header("Hello there, welcome to Book Guru, your personal book guide")

from langchain.prompts import(
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chains import LLMChain,ConversationChain

model_id = "FuseAI/FuseChat-Llama-3.2-1B-Instruct"
a=0
if a==0:
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    a+=1



pipeline = pipeline(
    model=model, 
    tokenizer=tokenizer, 
    task="text-generation", 
    device_map="auto",
    max_new_tokens = 512,
    device=0
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain  
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipeline)

prompt = st.chat_input("Say something")
if prompt:
    output = llm(prompt)
    st.write(output)












