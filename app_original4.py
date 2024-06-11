import os
import queue
import re
import tempfile
import threading

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import streamlit as st

import embedchain
from embedchain import App
from embedchain.config import BaseLlmConfig
from embedchain.helpers.callbacks import (StreamingStdOutCallbackHandlerYield, generate)

SUPPORTED_FILE_TYPES = ["pdf", "docx", "csv", "jpeg", "jpg", "webp"]

embedchain.config.llm.base.DOCS_SITE_DEFAULT_PROMPT = """
You are an expert AI assistant for developer support product. Your responses must always be rooted in the context provided for each query. Wherever possible, give complete code snippet. Dont make up any code snippet on your own.

Here are some guidelines to follow:

1. Refrain from explicitly mentioning the context provided in your response.
2. The context should silently guide your answers without being directly acknowledged.
3. Do not use phrases such as 'According to the context provided', 'Based on the context, ...' etc.
4. You can "quote" any part of the context provided in your response. but it should not violate point number 3.

Context information:
----------------------
$context
----------------------

Query: $query
Answer:
""" 

def embedchain_bot(db_path, api_key, api_provider):
    if api_provider == "OpenAI":  # OpenAI API
        llm_config = {
            "provider": "openai",
            "config": {
                "model": "gpt-3.5-turbo-1106",
                "temperature": 0.2,
                "max_tokens": 1000,
                "top_p": 1,
                "stream": True,
            },
        }
        embedder_config = {
            "provider": "openai",
            "config": {
                "model": 'text-embedding-3-small',
                "api_key": api_key
            }
        }
    else:  # Google API
        llm_config = {
            "provider": "google",
            "config": {
                "model": "gemini-pro",
                "max_tokens": 1000,
                "temperature": 0.2,
                "top_p": 1,
                "stream": True,
            },
        }
        embedder_config = {
            "provider": "google",
            "config": {
                "model": 'models/embedding-001',
                "task_type": "retrieval_document",
                "title": "chat-files"
            }
        }

    return App.from_config(
        config={
            "llm": llm_config,
            "vectordb": {
                "provider": "weaviate",
                "config": {"collection_name": "chat-files"},
            },
            "embedder": embedder_config,
            "chunker": {"chunk_size": 2000, "chunk_overlap": 0, "length_function": "len"},
        }
    )

def get_db_path():
    tmpdirname = tempfile.mkdtemp()
    return tmpdirname

def get_ec_app(api_key, api_provider):
    # if "app" in st.session_state:
    #     print("Found app in session state")
    #     app = st.session_state.app
    # else:
    print("Creating app")
    db_path = get_db_path()
    app = embedchain_bot(db_path, api_key, api_provider)
    st.session_state.app = app
    return app

def custom_datatypes(filetype: str):
    if isinstance(filetype, str):
        if filetype in ['jpg', 'jpeg', 'webp']:
            return 'image'
        elif filetype in ['pdf']:
            return 'pdf_file'
        elif filetype in ['csv']:
            return 'csv'
        elif filetype in ['docx']:
            return 'docx'
        else:
            raise ValueError(f"Filetype not supported: \'{filetype}\'")
    else:
        raise ValueError(f"Filetype not a string: \'{filetype}\'")

def process_file(file, app):
    file_name = file.name.split(".")[0]
    file_type = file.name.split(".")[-1]
    temp_file_name = None
    temp_dir = tempfile.gettempdir()
    legit_file_type = custom_datatypes(file_type)
    new_filename = f"{file_name}.{legit_file_type}"
    temp_file_path = os.path.join(temp_dir, new_filename)
    with open(temp_file_path, 'wb') as f:
        f.write(file.getvalue())
        temp_file_name = f.name
    if temp_file_name:
        st.markdown(f"Adding {file.name} to knowledge base...")
        print(f'legit_file_type: {legit_file_type}\ntemp_file_name: {temp_file_name}')
        app.add(temp_file_name, data_type=f"{legit_file_type}")
        print(f'Added')
        st.markdown("")
        os.remove(temp_file_name)
        st.session_state.messages.append({"role": "assistant", "content": f"Added {file.name} to knowledge base!"})

with st.sidebar:
    api_provider = st.radio("Choose API Provider", ('OpenAI', 'Google'))
    if api_provider == 'OpenAI':
        api_key = st.text_input("OpenAI API Key", key="api_key", type="password")
        "WE DO NOT STORE YOUR OPENAI KEY."
        "Just paste your OpenAI API key here and we'll use it to power the chatbot."
    else:
        api_key = st.text_input("Google API Key", key="api_key", type="password")
        "WE DO NOT STORE YOUR GOOGLE API KEY."
        "Just paste your Google API key here and we'll use it to power the chatbot."

    if st.session_state.api_key:
        os.environ["WEAVIATE_ENDPOINT"] = "https://dimitri-fsp2m585.weaviate.network"
        os.environ["WEAVIATE_API_KEY"] = "Fnf6dw7CLFZBssWxTHK8mQiRjCBPqP2O9hoo"
        if api_provider == 'OpenAI':
            os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        else:
            os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
        print(f'api_provider: {api_provider}\napi_key: {st.session_state.api_key}')
        app = get_ec_app(st.session_state.api_key, api_provider)

    uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, type=SUPPORTED_FILE_TYPES)
    add_files = st.session_state.get("add_files", [])
    for file in uploaded_files:
        print(f'Uploaded File: {file.name}')
        file_name = file.name
        if file_name in add_files:
            continue
        try:
            if not st.session_state.api_key:
                st.error("Please enter your API Key")
                st.stop()
            else:
                process_file(file, app)
            add_files.append(file_name)
        except Exception as e:
            st.error(f"Error adding {file_name} to knowledge base: {e}")
            st.stop()
    st.session_state["add_files"] = add_files

cols = st.columns([1, 2, 1])
with cols[1]:
    st.image('./logo.png', use_column_width=True)
st.title("DIMITRI")
st.title("📄 Chat with Your Files")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
                Hi! I'm a multi-modal chatbot , which can answer questions about your documents and images.\n
                Upload your documents here and I'll answer your questions about them! 
            """,
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    if not st.session_state.api_key:
        st.error("Please enter your API Key", icon="🤖")
        st.stop()

    app = get_ec_app(st.session_state.api_key, api_provider)

    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(prompt)

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        msg_placeholder.markdown("Thinking...")
        full_response = ""

        q = queue.Queue()

        def app_response(result):
            llm_config = app.llm.config.as_dict()
            llm_config["callbacks"] = [StreamingStdOutCallbackHandlerYield(q=q)]
            config = BaseLlmConfig(**llm_config)
            answer, citations = app.chat(prompt, config=config, citations=True)
            result["answer"] = answer
            result["citations"] = citations

        results = {}
        thread = threading.Thread(target=app_response, args=(results,))
        thread.start()

        for answer_chunk in generate(q):
            full_response += answer_chunk
            msg_placeholder.markdown(full_response)

        thread.join()
        answer, citations = results["answer"], results["citations"]
        if citations:
            full_response += "\n\n**Sources**:\n"
            sources = []
            for i, citation in enumerate(citations):
                source = citation[1]["url"]
                sources.append(os.path.basename(source))
            sources = list(set(sources))
            for source in sources:
                full_response += f"- {source}\n"

        msg_placeholder.markdown(full_response)
        print("Answer: ", full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})