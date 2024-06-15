# Multi-Modal RAG Chatbot

Welcome to the Multi-Modal RAG Chatbot! This project is designed to help you interact with various file formats (PDF, images, CSV, DOCX) using a sophisticated chatbot deployed on [Hugging Face Spaces](https://huggingface.co/spaces/DMITRI00/DMITRI). Our chatbot leverages powerful language models and a vector database to provide accurate and contextually relevant responses.

## Features

- **File Support**: Chat with your PDF, images, CSV, and DOCX files.
- **Language Models**: Utilize Google Gemini and OpenAI for natural language processing.
- **Vector Database**: Powered by Weaviate for efficient data retrieval.
- **Input Methods**: Supports both text and voice inputs for flexibility.

## Getting Started

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/multimodal-rag-chatbot.git
    cd multimodal-rag-chatbot
    ```

2. **Set up the virtual environment**:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3. **Install the requirements**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Run the application**:
    ```bash
    streamlit run app.py
    ```

2. **Access the chatbot**: Open your browser and navigate to the local URL provided by Streamlit.

## Deployment

The chatbot is also deployed on [Hugging Face Spaces](https://huggingface.co/spaces/DMITRI00/DMITRI). You can interact with it directly through this link.

## Acknowledgements

- **Google Gemini** and **OpenAI** for the language models.
- **Weaviate** for the vector database.
- **Streamlit** for the web application framework.
- **Hugging Face Spaces** for hosting the deployment.
- **Embedchain** as the RAG framework.
