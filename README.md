# AI-Powered PDF Q&A

This is an AI-powered PDF Q&A system built using Streamlit, LangChain, Groq, FAISS, and PyMuPDF. It allows users to upload PDF files and ask questions related to the content of the PDFs. The system will extract text from the PDFs, split it into chunks, and use a retrieval-augmented generation (RAG) approach to answer questions based on the document contents.

## Features

- Upload multiple PDF files.
- Ask context-aware questions related to the content of the PDFs.
- Text extraction from PDFs using PyMuPDF (fitz).
- Chunking and embedding of document content using LangChain and FAISS.
- Contextualized Q&A using a Groq-powered language model (Gemma2-9b-It).
- Track interactions with LangChain's integrated tracing.

## Requirements

To run this project locally, you'll need to have the following dependencies installed:

- Python 3.x
- Streamlit
- LangChain
- LangChain-Groq
- FAISS
- Sentence-Transformers
- PyMuPDF
- python-dotenv

You can install the required dependencies with:

```bash
pip install -r requirements.txt
Setup
Clone this repository to your local machine:

bash
Copy
Edit
git clone https://github.com/yourusername/ai-pdf-qa.git
cd ai-pdf-qa
Create a .env file in the project root directory and add your API keys:

plaintext
Copy
Edit
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_langchain_project_id
GROQ_API_KEY=your_groq_api_key
Run the application:

bash
Copy
Edit
streamlit run app.py
This will start the Streamlit application on your local server.

How It Works
PDF Upload: Users can upload one or more PDF files through the Streamlit UI.

Text Extraction: The system extracts the text from the uploaded PDFs using PyMuPDF.

Text Chunking: The extracted text is split into smaller chunks for easier processing and embedding.

Embedding and Retrieval: Each text chunk is embedded using the SentenceTransformerEmbeddings model and stored in a FAISS vector store for efficient retrieval.

Context-Aware Q&A: When the user asks a question, the system uses a history-aware retrieval mechanism with LangChain to find the most relevant chunks from the PDFs. It then generates a response using the Groq-powered language model (Gemma2-9b-It).

Response Display: The response is displayed on the Streamlit UI.

Example Usage
Upload one or more PDF files.

Enter a question in the input box.

Click "Get Answer" to retrieve an AI-generated response based on the document content.

Troubleshooting
If you receive a 401 Unauthorized error, ensure that your API keys are correctly set in the .env file.

If the system cannot extract text from the PDF, ensure the PDF is not encrypted or corrupted.

Contributing
Feel free to fork this repository and submit pull requests. Contributions are welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

markdown
Copy
Edit

### Explanation:
- **Features**: Describes the core functionality of the app.
- **Requirements**: Lists the necessary libraries to run the app.
- **Setup**: Provides instructions for setting up and running the project locally.
- **How It Works**: Details the process flow of the application.
- **Example Usage**: Gives an example of how users can interact with the app.
- **Troubleshooting**: Provides basic troubleshooting steps for common issues.
- **Contributing**: Encourages others to contribute to the project.
- **License**: Mentions that the project is licensed under the MIT License.
