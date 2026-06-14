# smart-text-summarizer
An AI-powered Text Summarization web application built using FastAPI, PyTorch, and Hugging Face Transformers. The application uses a trained T5 Transformer model to generate concise summaries from long text passages.

Features
Automatic text summarization
FastAPI backend
T5 Transformer model
Simple and responsive web interface
Real-time summary generation
Tech Stack
Python
FastAPI
PyTorch
Hugging Face Transformers
T5 Model
Jinja2 Templates
HTML/CSS
Project Structure
text-summarizer/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── saved_model/
│
└── Dataset/
Installation
Clone the Repository
git clone https://github.com/SuMiTpAtEl-sp5106/text-summarizer.git
cd text-summarizer
Install Dependencies
pip install -r requirements.txt
Run the Application
uvicorn app:app --reload


How It Works
User enters a long text passage.
The text is processed by the trained T5 model.
The model generates a concise summary.
The summarized text is displayed on the web page.
Future Improvements
PDF summarization
URL/article summarization
Multi-language support
Adjustable summary length
API documentation
Author

Sumit Patel

License

This project is for educational and learning purposes.
