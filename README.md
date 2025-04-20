# An Intelligent Data Science Tutor Chatbot

This project is a web-based chatbot designed to answer questions related to data science. It uses a knowledge base built from uploaded PDF textbooks and leverages advanced natural language processing (NLP) techniques to provide accurate and concise answers.

---

## Warning

- **Running this on a local machine would likely take a lot of resources with the web interface since this is poorly optimized.**
- **You can choose to run this on Colab without any web interface.**
- **The chatbot's entire code is provided in the Colab file in this repository.**
- **`Chatbot.ipynb` contains options to upload a knowledge base as well as add your own knowledge base.**
- **A Colab file to create your own knowledge base is also available in this repository.**

---

## Features

- **PDF Upload**: Upload data science-related PDFs to expand the chatbot's knowledge base.
- **Question Answering**: Ask questions, and the chatbot will provide relevant answers based on the uploaded content.
- **Summarization**: Summarizes the most relevant sections of the knowledge base to provide concise answers.
- **GPU Support**: Utilizes GPU (if available) for faster processing of embeddings and queries.
- **Web Interface**: A user-friendly web interface for interacting with the chatbot.

---

## Technologies Used

### Backend

- **Python**: Core programming language.
- **Flask**: Web framework for handling API requests and serving the frontend.
- **FAISS**: For efficient similarity search and indexing.
- **PyMuPDF (fitz)**: For extracting text from PDF files.
- **Sentence Transformers**: For generating embeddings from text.
- **Transformers**: For summarization using pre-trained models like `facebook/bart-large-cnn`.
- **Torch**: For GPU acceleration and deep learning operations.

### Frontend

- **HTML/CSS**: For the structure and styling of the web interface.
- **JavaScript**: For handling user interactions and communicating with the backend.

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/An-intelligent-Data-Science-Tutor-Chatbot.git
   cd An-intelligent-Data-Science-Tutor-Chatbot
   ```

2. **Set Up a Virtual Environment**:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Start the Flask Server**:

   ```bash
   python app.py
   ```

2. **Access the Web Interface**:
   Open your browser and go to:

   ```
   http://127.0.0.1:5000/
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## File and Their Roles

1. **`app.py`**:
   Handles the Flask server and routes.

2. **`chatbot.py`**:
   Contains the chatbot's initialization and query processing logic.

3. **`create_index.py`**:
   Builds and updates the FAISS index from uploaded PDFs.

4. **`index.html`**:
   The main frontend page for interacting with the chatbot.

5. **`script.js`**:
   Handles frontend interactions and communicates with the backend.

---

## Issues

- **Poorly Optimized Solution**: To run `chatbot.ipynb` using Colab as an alternative.
