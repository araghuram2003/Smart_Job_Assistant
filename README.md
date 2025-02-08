# Smart Job Assistant

A powerful AI-powered tool for resume analysis and cold mail generation.

## Features

- Resume Analysis with multiple AI models
- Cold Mail Generation
- Multi-language support (English, Hindi, Telugu)
- ATS Optimization
- Skills Gap Analysis
- Professional Cold Mail Templates

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a .env file with your API keys:
   ```
   Google_Gemini_ai_key=your_gemini_api_key_here
   Groq_api_key=your_groq_api_key_here
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Choose between Resume Analyzer or Cold Mail Generator
2. Upload your resume (PDF/DOC/DOCX)
3. Enter job description
4. Select analysis type or cold mail style
5. Get AI-powered insights and suggestions

## Technologies Used

- Streamlit
- Google Gemini AI
- Groq AI
- Python 3.8+ 