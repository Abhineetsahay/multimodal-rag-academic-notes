# Multimodal RAG Academic Notes

A Retrieval-Augmented Generation (RAG) system designed to intelligently search and extract information from academic notes, presentations, and documents. This project combines multimodal embeddings (text and images) with ChromaDB vector storage and CLIP embeddings to provide accurate context-based answers from your study materials.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abhineetsahay/multimodal-rag-academic-notes/blob/main/RAG_PDF.ipynb)

## 🌟 Features

- **Multimodal Processing**: Extracts and processes both text and images from PDF and PPTX files
- **Semantic Search**: Uses CLIP (ViT-B-32) embeddings for intelligent text and image retrieval
- **Vector Database**: Leverages ChromaDB for efficient similarity search
- **Text Chunking**: Smart text splitting with configurable chunk size and overlap
- **Context-Aware Responses**: Returns relevant information with source page references
- **Duplicate Filtering**: Removes redundant results for cleaner outputs

## 📋 Prerequisites

- Python 3.7+
- Google Colab (recommended) or local Python environment
- PDF/PPTX files containing your academic notes

## 🚀 Installation

Install all required dependencies using pip:

```bash
pip install numpy pillow pymupdf pytesseract opencv-python open-clip-torch sentence-transformers chromadb google-generativeai langchain langchain-community langchain-google-genai
```

### Dependencies

- **numpy**: Numerical computing
- **pillow**: Image processing
- **pymupdf (fitz)**: PDF/PPTX parsing
- **pytesseract**: OCR capabilities
- **opencv-python**: Image manipulation
- **open-clip-torch**: CLIP model for multimodal embeddings
- **sentence-transformers**: Text embedding models
- **chromadb**: Vector database
- **google-generativeai**: Google AI integration
- **langchain**: LLM framework
- **langchain-community**: Community LangChain integrations
- **langchain-google-genai**: Google GenAI for LangChain

## 💻 Usage

### 1. Quick Start in Google Colab

Click the "Open in Colab" badge above to run the notebook directly in Google Colab.

### 2. Local Usage

```python
import torch
import open_clip
from PIL import Image

# Initialize CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

# Load your document
loader = LoadFile("/path/to/your/document.pdf")
all_text, all_images = loader.openFile()
text_chunks = loader.get_chunks(all_text)

# Create embeddings
embeddings = CreateEmbedding(model, preprocess, tokenizer)

# Initialize ChromaDB
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="lecture_notes")

# Add documents to vector store
# ... (see notebook for complete code)

# Query your documents
query_text = "What is Intermediate COCOMO Model?"
query_embedding = embeddings.embed_text(query_text).cpu().detach().numpy().flatten().tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
```

### 3. Example Queries

```python
# Ask questions about your notes
query_text = "What is Intermediate COCOMO Model?"
query_text = "Explain software project management principles"
query_text = "What are the key concepts in Chapter 4?"
```

## 🏗️ Architecture

### Components

1. **LoadFile Class**
   - Loads PDF/PPTX files using PyMuPDF
   - Extracts text and images from each page
   - Splits text into manageable chunks (600 chars with 100 char overlap)
   - Returns structured data with page metadata

2. **CreateEmbedding Class**
   - Generates embeddings for text using CLIP
   - Generates embeddings for images using CLIP
   - Supports both text and image similarity search

3. **Vector Database (ChromaDB)**
   - Stores document embeddings
   - Enables fast similarity search
   - Maintains metadata (page numbers, content types)

4. **Query Processing**
   - Converts user queries to embeddings
   - Retrieves top-k similar contexts
   - Filters duplicates
   - Returns formatted answers with source references

## 📊 How It Works

1. **Document Ingestion**: Upload your PDF or PPTX file
2. **Text Extraction**: Extract text content page by page
3. **Image Extraction**: Extract embedded images
4. **Chunking**: Split text into semantic chunks
5. **Embedding Generation**: Create vector embeddings using CLIP
6. **Storage**: Store embeddings in ChromaDB with metadata
7. **Query Processing**: Convert questions to embeddings
8. **Retrieval**: Find most similar chunks using vector similarity
9. **Response**: Return relevant contexts with page references

## 🎯 Use Cases

- **Exam Preparation**: Quickly find specific topics in your study materials
- **Research**: Extract relevant information from academic papers
- **Note Taking**: Search across multiple lecture presentations
- **Knowledge Management**: Build a searchable knowledge base from your notes
- **Study Groups**: Share and query collective study materials

## 🔧 Configuration

### Text Chunking Parameters

```python
chunk_size = 600        # Characters per chunk
chunk_overlap = 100     # Overlap between chunks
```

### Model Configuration

```python
model_name = "ViT-B-32"      # CLIP model variant
pretrained = "openai"        # Pretrained weights
```

### Search Parameters

```python
n_results = 3               # Number of results to return
```

## 📝 Example Output

```
✨ **Answer for:** What is Intermediate COCOMO Model?
========================================
Based on your lecture notes, here is what I found:

📍 **Point 1:** The Intermediate COCOMO model is an extension of the Basic COCOMO model that includes additional cost drivers and provides more accurate estimates for software development effort.

📍 **Point 2:** It considers 15 cost driver attributes including product, computer, personnel, and project characteristics to refine effort estimates.

📚 **Sources:** Page(s) 12, 13
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Abhineet Sahay**
- GitHub: [@Abhineetsahay](https://github.com/Abhineetsahay)

## 🙏 Acknowledgments

- OpenAI CLIP for multimodal embeddings
- ChromaDB for vector storage
- LangChain for RAG framework
- PyMuPDF for document processing

## 📞 Support

For questions or issues, please open an issue on the [GitHub repository](https://github.com/Abhineetsahay/multimodal-rag-academic-notes/issues).

---

**Note**: This project is designed for educational purposes and works best with well-structured academic documents like lecture slides, textbooks, and notes.