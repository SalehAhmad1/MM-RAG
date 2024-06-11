import os
import queue
import re
import tempfile
import threading

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import streamlit as st

from embedchain import App
from embedchain.config import BaseLlmConfig
from embedchain.helpers.callbacks import (StreamingStdOutCallbackHandlerYield,generate)

SUPPORTED_FILE_TYPES = ["pdf", "docx", "csv", "jpeg", "jpg", "webp"]

def embedchain_bot(db_path, api_key):
    return App.from_config(
        config={
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-3.5-turbo-1106",
                    "temperature": 0.5,
                    "max_tokens": 1000,
                    "top_p": 1,
                    "stream": True,
                    "api_key": api_key,
                },
            },
            "vectordb": {
                "provider": "weaviate",
                "config": {"collection_name": "chat-files"},
            },
            "embedder": { 
                "provider": "openai",
                "config": {"model": 'text-embedding-3-small', "api_key": api_key}
            },
            "chunker": {"chunk_size": 2000, "chunk_overlap": 0, "length_function": "len"},
        }
    )

def get_db_path():
    tmpdirname = tempfile.mkdtemp()
    return tmpdirname

def get_ec_app(api_key):
    if "app" in st.session_state:
        print("Found app in session state")
        app = st.session_state.app
    else:
        print("Creating app")
        db_path = get_db_path()
        app = embedchain_bot(db_path, api_key)
        st.session_state.app = app
    return app

def custom_datatypes(filetype: str):
    if isinstance(filetype, str):
        if filetype in ['jpg','jpeg','webp']:
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
    openai_access_token = st.text_input("OpenAI API Key", key="api_key", type="password")
    "WE DO NOT STORE YOUR OPENAI KEY."
    "Just paste your OpenAI API key here and we'll use it to power the chatbot. [Get your OpenAI API key](https://platform.openai.com/api-keys)"  # noqa: E501

    if st.session_state.api_key:
        os.environ["WEAVIATE_ENDPOINT"] = "https://dimitri-fsp2m585.weaviate.network"
        os.environ["WEAVIATE_API_KEY"] = "Fnf6dw7CLFZBssWxTHK8mQiRjCBPqP2O9hoo"
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        app = get_ec_app(st.session_state.api_key)

    uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, type=SUPPORTED_FILE_TYPES)
    add_files = st.session_state.get("add_files", [])
    for file in uploaded_files:
        print(f'Uploaded File: {file.name}')
        file_name = file.name
        if file_name in add_files:
            continue
        try:
            if not st.session_state.api_key:
                st.error("Please enter your OpenAI API Key")
                st.stop()
            else:
                process_file(file, app)
            add_files.append(file_name)
        except Exception as e:
            st.error(f"Error adding {file_name} to knowledge base: {e}")
            st.stop()
    st.session_state["add_files"] = add_files

st.title("📄 Embedchain - Chat with Your Files")
styled_caption = '<p style="font-size: 17px; color: #aaa;">🚀 An <a href="https://github.com/embedchain/embedchain">Embedchain</a> app powered by OpenAI!</p>'  # noqa: E501
st.markdown(styled_caption, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """
                Hi! I'm chatbot powered by Embedchain, which can answer questions about your documents and images.\n
                Upload your documents here and I'll answer your questions about them! 
            """,
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    if not st.session_state.api_key:
        st.error("Please enter your OpenAI API Key", icon="🤖")
        st.stop()

    app = get_ec_app(st.session_state.api_key)

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