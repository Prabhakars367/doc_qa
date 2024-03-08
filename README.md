**Doc Q/A Streamlit App**

This Streamlit application allows users to ask questions and receive answers from uploaded PDF documents. It leverages generative AI models and vector stores to enable a conversational question-answering experience.

**Key Features:**

- Upload PDF documents
- Ask questions in a natural language style
- Get comprehensive answers derived from the document content
- Streamlit user interface for ease of use

**Requirements:**

- Python 3.x
- Streamlit (`pip install streamlit`)
- langchain-community (`pip install langchain-community`)
- PyPDF2 (`pip install PyPDF2`)
- genai package from Google Cloud AI Platform (`pip install genai`) - Requires setting up a Google Cloud project and enabling necessary APIs
- langchain libraries (`pip install langchain[google]`)

**How to Use:**

1. Create a virtual environment (recommended).
2. Install the required dependencies mentioned above.
3. Make sure you have a Google Cloud project set up and the genai API is enabled.
4. Set your Google Cloud project API key as an environment variable named `GOOGLE_API_KEY`.
5. Place your PDF documents in the same directory as the application script (`app.py`).
6. Run the application using `streamlit run app.py`.

**Code Structure:**

The application code consists of several functions:

- `get_pdf_text(pdf_docs)`: Extracts text from uploaded PDFs.
- `get_text_chunks(text)`: Splits the extracted text into smaller chunks for processing.
- `get_vector_store(text_chunks)`: Creates a vector store using embeddings to represent the text chunks.
- `get_conversational_chain()`: Defines a prompt template and loads a generative AI model for question answering.
- `user_input(user_questions)`: Handles user input, retrieves relevant information from the vector store, and uses the Q&A chain to answer questions.
- `main()`: Sets up the Streamlit UI and handles user interactions.

**Further Considerations:**

- This application is a basic example and can be extended to support more features like document summarization or named entity recognition.
- The accuracy of the Q&A system depends on the quality of the uploaded documents and the chosen generative AI model.

**Feel free to customize and enhance this project according to your specific needs!**
