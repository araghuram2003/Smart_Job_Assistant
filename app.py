import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Smart Job Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# Smart Job Assistant\nPowered by Gen AI 🚀"})

import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from groq import Groq
import docx2txt
from datetime import datetime
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (local development)
load_dotenv()

# Get API keys
if 'Google_Gemini_ai_key' in st.secrets:
    gemini_api_key = st.secrets['Google_Gemini_ai_key']
else:
    gemini_api_key = os.getenv('Google_Gemini_ai_key')

if 'Groq_api_key' in st.secrets:
    groq_api_key = st.secrets['Groq_api_key']
else:
    groq_api_key = os.getenv('Groq_api_key')

# Configure Gemini AI
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    st.success('✅ Google Gemini AI configured successfully!')
else:
    st.error("⚠️ Google Gemini API key not found. Please check your configuration.")

# Configure Groq AI
try:
    if not groq_api_key:
        st.warning("⚠️ Groq API key not found. Some features may be limited.")
        groq_client = None
    else:
        groq_client = Groq(api_key=groq_api_key)
        st.success('✅ Groq AI configured successfully!')
except Exception as e:
    logger.error(f"Error initializing Groq client: {str(e)}")
    st.warning("⚠️ Error initializing Groq client. Some features may be limited.")
    groq_client = None

class ATSAnalyzer:
    # Language prompts dictionary
    LANGUAGE_PROMPTS = {
        "English": {
            "resume_analysis": """
Analyze the resume and provide:
1. Match Score (%)
2. Key Strengths
3. Missing Skills
4. Improvement Suggestions
            """,
            "labels": {
                "upload": "Upload your resume (PDF or DOC/DOCX format)",
                "job_desc": "Job Description",
                "analyze": "Analyze Resume",
                "results": "Analysis Results"
            }
        },
        "हिंदी": {
            "resume_analysis": """
रिज्यूमे का विश्लेषण करें और प्रदान करें:
1. मैच स्कोर (%)
2. प्रमुख शक्तियां
3. कमी वाले कौशल
4. सुधार के सुझाव
            """,
            "labels": {
                "upload": "अपना रिज्यूमे अपलोड करें (PDF या DOC/DOCX प्रारूप)",
                "job_desc": "नौकरी का विवरण",
                "analyze": "रिज्यूमे का विश्लेषण करें",
                "results": "विश्लेषण परिणाम"
            }},
        "తెలుగు": {
            "resume_analysis": """
రెస్యూమ్ విశ్లేషణ చేసి ఈ క్రింది వాటిని అందించండి:
1. మ్యాచ్ స్కోర్ (%)
2. ముఖ్య బలాలు
3. కొరవడిన నైపుణ్యాలు
4. మెరుగుదల సూచనలు
            """,
            "labels": {
                "upload": "మీ రెస్యూమ్‌ని అప్‌లోడ్ చేయండి (PDF లేదా DOC/DOCX ఫార్మాట్)",
                "job_desc": "ఉద్యోగ వివరణ",
                "analyze": "రెస్యూమ్ విశ్లేషించండి",
                "results": "విశ్లేషణ ఫలితాలు"
            }}}

    # Analysis types with their prompts
    ANALYSIS_TYPES = {
        "Complete Analysis": """Analyze my resume against the provided job description(s) and provide a comprehensive evaluation, including:
1.Overall Match Score (%): Calculate the candidate's overall suitability (%). Explain the weighting of Key Skills, Experience, and Education.

2.Key Skills Match:
Matching: List proficient skills.
Potential: List skills needing assessment.
Missing: List crucial missing skills.

3.Experience Alignment:
Relevant: Detail correlating experience, quantifying achievements.
Transferable: Identify applicable skills from other roles.
Gaps: Note experience gaps.

4.Education Fit:
Required: State minimum qualifications.
Candidate's: List degrees, certifications, coursework.
Gaps: Identify education discrepancies.
Improvement Suggestions: Offer constructive feedback for strengthening their profile.
        """,
         "ATS Optimization": """
I need you to act as an expert resume writer and optimization specialist. Your ultimate goal is to create a powerful and highly effective resume for me that excels in all aspects: ATS compatibility, recruiter appeal, and alignment with industry best practices.

1.ATS Compatibility Analysis:Thoroughly review my resume for any elements that might hinder its performance in ATS scans.  Identify specific areas for improvement, including:Formatting issues (e.g., use of tables, images, special characters, unusual fonts)
File format (recommend the most ATS-friendly format)
Keyword optimization (lack of relevant keywords, keyword stuffing)
Section headings and organization (ensure logical structure and standard headings)
Date formats and other data inconsistencies

2.Content Enhancement for Recruiter Appeal:  Suggest specific changes to better highlight my technical skills, projects, and achievements.  Focus on making these elements stand out to recruiters:
Quantifiable achievements:Help me rephrase accomplishments to showcase quantifiable results (e.g., "Increased sales by 15%" instead of "Increased sales").
Project descriptions: Advise on how to write concise and compelling project descriptions that emphasize my contributions and the project's impact.
Technical skills: Ensure my technical skills are prominently displayed and categorized effectively. Suggest ways to showcase proficiency levels (e.g., beginner, intermediate, expert).
Impactful language: Help me use action verbs and strong language to make my resume more dynamic and engaging.

3.Industry Alignment and Tailoring: Provide recommendations on how to tailor my resume language and structure to align with common industry standards and specific job descriptions.
This includes Keyword matching:Explain how to identify and incorporate relevant keywords from job descriptions.
Industry-specific terminology: Suggest appropriate terminology and jargon to use.

4.Resume length and format: Advise on the ideal length and format for my industry and experience level.
        """,
        "Skills Gap Analysis": """
        Provide a concise skills analysis for the candidate, focusing on the following areas:
1.Matching Skills: List the candidate's skills that directly align with the job requirements, quantifying their proficiency where possible.
2.Missing Critical Skills: List the essential skills required for the role that the candidate lacks, prioritizing them based on their importance to job performance.
3.Recommended Skills to Add: List skills that would significantly enhance the candidate's suitability for the role or their future growth within the company, explaining the rationale behind each recommendation.
4.Skill Level Assessment: Provide a qualitative assessment of the candidate's skill level for each matching skill using terms like Beginner, Intermediate, Proficient, and Expert.
        """,
        "Quick Summary": """
Provide a brief overview:
1.Match: Overall suitability (%). Weighting of criteria (e.g., skills, experience, education).
2.Strengths: Top 3, with examples.
3.Gaps: Top 3, prioritized.
4.Next Steps: 2-3 recommendations.
        """}

    # AI models
    AI_MODELS = {"Google Gemini": "🤖 Google Gemini (High accurate and reliable)","Groq": "🤖 Groq (Fast but moderately accurate)"}

    # Cold mail types
    COLD_MAIL_TYPES = {
        "📑 Professional and Straightforward": {
            "description": "A formal and direct approach, ideal for traditional industries and corporate settings",
            "template": """
Subject: Seeking Internship Opportunity to Learn and Contribute

Dear [Recipient's Name],

I hope you're doing well. My name is [Your Name], and I am currently a [Your Year] student pursuing [Your Degree] at [Your College/University Name].

I am writing to express my interest in an internship opportunity at [Company Name]. I have been following your company's work in [specific field/area], and I am truly inspired by your innovative contributions to the industry.

My academic background and hands-on experience in [specific skills/tools] have prepared me to contribute meaningfully to your team. I am eager to learn from industry experts like you and enhance my skills further.

Could we connect to discuss any available internship opportunities? I have attached my resume for your review and would be happy to provide additional information if needed. Thank you for considering my application. I look forward to the possibility of contributing to your team.

Warm regards,
[Your Full Name]
[Your Phone Number]
[Your Email Address]
[LinkedIn Profile link or Portfolio]
            """
        },
        "🤝 Friendly Yet Professional": {
            "description": "A balanced approach combining warmth with professionalism, suitable for modern companies and startups",
            "template": """
Subject: Excited to Learn and Contribute - Internship Inquiry

Hi [Recipient's Name],

I hope you're having a great day! I'm [Your Name], currently pursuing [Your Degree] at [Your College/University Name], and I'm reaching out to explore internship opportunities with [Company Name].

I've always admired your company's commitment to [specific value or field]. As someone passionate about [specific area], I believe this could be an incredible place for me to learn and grow.

I've gained practical knowledge in [specific skills or projects] and I'm eager to contribute to your team while gaining real-world experience in the industry/role.

Would it be possible to discuss how I can support your team? I've attached my resume for your reference and would be delighted to provide any further details. Looking forward to hearing from you!

Best regards,
[Your Full Name]
[Your Phone Number]
[Your Email Address]
[LinkedIn Profile Link or Portfolio]
            """
        },
        "🌟 Enthusiastic and Curious": {
            "description": "An energetic approach emphasizing eagerness to learn and contribute, great for innovation-focused companies",
            "template": """
Subject: Internship Inquiry: Eager to Learn and Make an Impact

Dear [Recipient's Name],

I hope this email finds you well. My name is [Your Name], and I am a [Year of Study] student specializing in [Your Field of Study] at [Your College/University Name].

I am writing to express my interest in an internship opportunity at [Company Name]. Your organization's work in [specific domain] has always inspired me, particularly [mention a specific project, value, or achievement of the company].

With foundational experience in [your skills/experience], I'm keen to contribute to your team while learning from the expertise of your professionals. I'm confident that this internship will give me an opportunity to develop my skills and create value for your organization.

I would be thrilled to connect and discuss how I can contribute to your team. I've attached my resume for your consideration. Thank you for your time, and I look forward to hearing from you.

Best regards,
[Your Full Name]
[Your Phone Number]
[Your Email Address]
[LinkedIn Profile Link or Portfolio]
            """}}

    @staticmethod
    def get_prompts(language="English"):
        """Get language-specific prompts"""
        prompts = {
            "English": {
                "system_msg": """You are a professional resume analyzer. Your task is to analyze resumes in English.
                Always structure your response as follows:
                1. Match Score (%)
                2. Key Strengths
                3. Missing Skills
                4. Improvement Suggestions""",
                "user_msg": """Please analyze this resume against the job description in English.
                Ensure you follow the exact format mentioned above.""",
                "result_prefix": "Analysis Results:\n\n"},
            "हिंदी": {
                "system_msg": """आप एक पेशेवर रिज्यूमे विश्लेषक हैं। आपका काम रिज्यूमे का विश्लेषण हिंदी में करना है।
                कृपया अपना जवाब इस प्रारूप में दें:
                1. मैच स्कोर (%)
                2. मुख्य ताकत
                3. कमी वाले कौशल
                4. सुधार के सुझाव""",
                "user_msg": """कृपया इस रिज्यूमे का विश्लेषण नौकरी के विवरण के अनुसार हिंदी में करें।
                कृपया ऊपर दिए गए प्रारूप का पालन करें।""",
                "result_prefix": "विश्लेषण परिणाम:\n\n"},
            "తెలుగు": {
                "system_msg": """మీరు ఒక వృత్తిపరమైన రెస్యూమ్ విశ్లేషకులు. మీ పని రెస్యూమ్‌ని తెలుగులో విశ్లేషించడం.
                దయచేసి మీ సమాధానాన్ని ఈ ఫార్మాట్‌లో ఇవ్వండి:
                1. మ్యాచ్ స్కోర్ (%)
                2. ముఖ్య బలాలు
                3. కొరవడిన నైపుణ్యాలు
                4. మెరుగుదల సూచనలు""",
                "user_msg": """దయచేసి ఈ రెస్యూమ్‌ని ఉద్యోగ వివరణతో పోల్చి తెలుగులో విశ్లేషించండి.
                పైన పేర్కొన్న ఫార్మాట్‌ని ఖచ్చితంగా పాటించండి.""",
                "result_prefix": "విశ్లేషణ ఫలితాలు:\n\n"}}
        return prompts.get(language, prompts["English"])

    @staticmethod
    def get_error_message(language):
        """Get language-specific error messages"""
        error_messages = {
            "English": "Error in analysis. Please try again or contact support.",
            "हिंदी": "विश्लेषण में त्रुटि हुई। कृपया पुनः प्रयास करें या सहायता से संपर्क करें।",
            "తెలుగు": "విశ్లేషణలో లోపం. దయచేసి మళ్లీ ప్రయత్నించండి లేదా సహాయం కోసం సంప్రదించండి."}
        return error_messages.get(language, error_messages["English"])

    @staticmethod
    def format_groq_messages(selected_lang, input_prompt, job_description, pdf_text, language):
        """Format messages for Groq API"""
        return [{"role": "system","content": selected_lang["system_msg"]},{"role": "user","content": f"""{selected_lang["user_msg"]}

Analysis Requirements:
{input_prompt}

Job Description:
{job_description}

Resume Content:
{pdf_text}

Remember to:
1. Keep the analysis in {language}
2. Follow the exact format specified
3. Provide clear, actionable feedback
4. Include a numerical match score"""}]

    @staticmethod
    def get_ai_response(model_choice, input_prompt, pdf_text, job_description, language="English"):
        """Get AI response from selected model"""
        try:
            selected_lang = ATSAnalyzer.get_prompts(language)
            
            if model_choice == "Google Gemini":
                return ATSAnalyzer.get_gemini_response(input_prompt, pdf_text, job_description, language)
            
            # For Groq model
            if groq_client is None:
                st.error("⚠️ Groq AI is not available. Please use Google Gemini instead.")
                return ATSAnalyzer.get_gemini_response(input_prompt, pdf_text, job_description, language)
                
            messages = ATSAnalyzer.format_groq_messages(selected_lang, input_prompt, job_description, pdf_text, language)

            # Using Mixtral model with optimized parameters
            chat_completion = groq_client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",
                temperature=0.5,
                max_tokens=4000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0)
            
            response = chat_completion.choices[0].message.content
            if not response or len(response.strip()) < 10:
                raise Exception("Invalid or empty response received")
                
            # Add language-specific formatting
            return selected_lang["result_prefix"] + response
            
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return ATSAnalyzer.get_error_message(language)

    @staticmethod
    def get_gemini_response(input_prompt, pdf_text, job_description, language="English"):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"Analyze in {language}. {input_prompt}"
            response = model.generate_content([prompt, pdf_text, job_description])
            return response.text
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    @staticmethod
    def extract_text(uploaded_file):
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            if file_type == 'pdf':
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            elif file_type in ['doc', 'docx']:
                return docx2txt.process(uploaded_file)
            else:
                st.error("Unsupported file format")
                return None
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return None

    @staticmethod
    def extract_data_from_response(response):
        """Extract structured data from AI response"""
        try:
            # Extract match score
            match_pattern = r'Match Score:?\s*(\d+)%'
            match_result = re.search(match_pattern, response)
            match_score = float(match_result.group(1)) if match_result else 0

            return {'match_score': match_score,'raw_response': response}
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return None

    @staticmethod
    def generate_cold_mail(model_choice, prompt, resume_text, job_description, personal_info):
        """Generate a cold mail using the selected AI model"""
        try:
            if model_choice == "Google Gemini":
                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content([prompt, resume_text, job_description])
                generated_content = response.text
            else:
                if groq_client is None:
                    st.error("⚠️ Groq AI is not available. Please use Google Gemini instead.")
                    model = genai.GenerativeModel("gemini-pro")
                    response = model.generate_content([prompt, resume_text, job_description])
                    generated_content = response.text
                else:
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {"role": "user","content": f"{prompt}\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description}"}],
                        model="mixtral-8x7b-32768",
                        temperature=0.5,)
                    generated_content = chat_completion.choices[0].message.content

            # Replace basic placeholders with personal information
            generated_content = generated_content.replace("[Your Name]", personal_info.get("name", "[Your Name]"))
            generated_content = generated_content.replace("[Your Email Address]", personal_info.get("email", "[Your Email]"))
            generated_content = generated_content.replace("[Your Phone Number]", personal_info.get("phone", "[Your Phone]"))
            generated_content = generated_content.replace("[Your College/University Name]", personal_info.get("university", "[Your University]"))
            generated_content = generated_content.replace("[LinkedIn Profile or Portfolio link]", personal_info.get("linkedin", "[Your LinkedIn]"))
            generated_content = generated_content.replace("[Your Degree]", personal_info.get("degree", "[Your Degree]"))

            return generated_content

        except Exception as e:
            logger.error(f"Error generating cold mail: {str(e)}")
            return None

def main():
    # Theme configuration
    st.markdown("""
        <style>
        /* Updated color variables */
        :root {
            --primary: #4F46E5;  /* Indigo */
            --primary-light: #818CF8;
            --accent: #06B6D4;   /* Cyan */
            --accent-light: #67E8F9;
            --success: #10B981;  /* Emerald */
            --warning: #F59E0B;  /* Amber */
            --dark-bg: #0F172A;  /* Slate 900 */
            --card-bg: rgba(15, 23, 42, 0.7);
            --neon-blue: #0066cc;
            --neon-glow: 0 0 10px rgba(0, 102, 204, 0.3);
        }

        /* Global styles */
        .stApp {
            background: transparent !important;
        }

        /* Animated gradient background */
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .main {
            background: transparent !important;
        }

        /* Futuristic cards with glass effect */
        .glass-card {
            background: var(--card-bg) !important;
            border: 1px solid rgba(0, 243, 255, 0.1) !important;
            border-radius: 15px !important;
            backdrop-filter: blur(10px) !important;
            padding: 20px !important;
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.1) !important;
            transition: all 0.3s ease !important;
        }

        .glass-card:hover {
            border-color: #0066cc !important;
            box-shadow: 0 0 30px rgba(0, 102, 204, 0.2) !important;
            transform: translateY(-2px) !important;
        }

        /* Neon text effects */
        h1 {
            color: #fff !important;
            text-shadow: 0 0 10px #0066cc,
                         0 0 20px #0066cc,
                         0 0 30px #0066cc !important;
            font-weight: 700 !important;
            letter-spacing: 2px !important;
        }

        /* Futuristic inputs */
        .stTextArea textarea, .stTextInput input {
            background: rgba(10, 10, 31, 0.7) !important;
            border: 1px solid rgba(0, 243, 255, 0.2) !important;
            border-radius: 10px !important;
            color: #fff !important;
            transition: all 0.3s ease !important;
        }

        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #0066cc !important;
            box-shadow: 0 0 15px rgba(0, 102, 204, 0.3) !important;
        }

        /* Glowing buttons */
        .stButton > button {
            position: relative !important;
            background: linear-gradient(45deg, rgba(0, 10, 30, 0.9), rgba(0, 30, 60, 0.9)) !important;
            border: 1px solid var(--neon-blue) !important;
            color: #fff !important;
            font-weight: 600 !important;
            letter-spacing: 1px !important;
            padding: 0.6em 2em !important;
            overflow: hidden !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            box-shadow: 0 0 10px rgba(0, 102, 204, 0.3) !important;
        }

        .stButton > button::before {
            content: '' !important;
            position: absolute !important;
            top: -2px !important;
            left: -2px !important;
            right: -2px !important;
            bottom: -2px !important;
            background: linear-gradient(45deg, 
                #0066cc, 
                #60a5fa, 
                #0066cc
            ) !important;
            background-size: 300% 300% !important;
            animation: moveGradient 4s ease infinite !important;
            z-index: -2 !important;
        }

        .stButton > button::after {
            content: '' !important;
            position: absolute !important;
            inset: 2px !important;
            background: inherit !important;
            z-index: -1 !important;
            border-radius: 4px !important;
        }

        @keyframes moveGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 0 20px rgba(0, 102, 204, 0.2) !important;
            color: #fff !important;
        }

        .stButton > button:hover::before {
            animation: moveGradient 2s ease infinite !important;
        }

        .stButton > button:active {
            transform: translateY(1px) !important;
        }

        /* Loading state animation */
        .stButton > button.loading {
            position: relative !important;
            cursor: wait !important;
        }

        .stButton > button.loading::after {
            content: '' !important;
            position: absolute !important;
            inset: 0 !important;
            background: linear-gradient(90deg,
                transparent,
                rgba(255, 255, 255, 0.2),
                transparent
            ) !important;
            transform: translateX(-100%) !important;
            animation: loading 1.5s infinite !important;
        }

        @keyframes loading {
            100% {
                transform: translateX(100%) !important;
            }
        }

        /* Button text glow effect */
        .stButton > button span {
            position: relative !important;
            z-index: 1 !important;
            background: linear-gradient(90deg, #fff, #a5f3fc, #fff) !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            color: transparent !important;
            background-size: 200% !important;
            animation: shine 3s linear infinite !important;
        }

        @keyframes shine {
            0% {
                background-position: -200% center;
            }
            100% {
                background-position: 200% center;
            }
        }

        /* Futuristic sidebar */
        .css-1d391kg {
            background: var(--card-bg) !important;
            border-right: 1px solid rgba(0, 243, 255, 0.1) !important;
        }

        /* Progress animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .stProgress > div > div {
            background: linear-gradient(90deg, var(--neon-blue), #0066cc) !important;
            animation: pulse 2s infinite !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background: var(--card-bg) !important;
            border: 2px dashed var(--neon-blue) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #0066cc !important;
            box-shadow: 0 0 20px rgba(0, 102, 204, 0.2) !important;
        }

        /* Success message animation */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .success-message {
            animation: fadeInUp 0.5s ease-out !important;
            color: var(--neon-blue) !important;
            text-shadow: 0 0 10px rgba(0, 243, 255, 0.5) !important;
        }

        /*background with animated circuit lines */
        .stApp::after {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                linear-gradient(90deg, transparent 95%, rgba(0, 102, 204, 0.1) 95%),
                linear-gradient(transparent 95%, rgba(0, 102, 204, 0.1) 95%);
            background-size: 20px 20px;
            animation: circuit-flow 20s linear infinite;
            pointer-events: none;
            z-index: 0;
        }

        @keyframes circuit-flow {
            0% { transform: translate(0, 0); }
            100% { transform: translate(20px, 20px); }
        }

        /* Glowing title effect */
        .title-glow {
            position: relative;
            color: #fff;
            text-shadow: 0 0 10px #0066cc,
                        0 0 20px #0066cc,
                        0 0 30px #0066cc;
            animation: title-pulse 2s ease-in-out infinite;
        }

        @keyframes title-pulse {
            0%, 100% { text-shadow: 0 0 10px #0066cc, 0 0 20px #0066cc, 0 0 30px #0066cc; }
            50% { text-shadow: 0 0 15px #0066cc, 0 0 25px #0066cc, 0 0 35px #0066cc; }
        }

        /* Cyber borders for cards */
        .cyber-card {
            position: relative;
            background: rgba(0, 10, 30, 0.7);
            border: 1px solid #0066cc;
            border-radius: 8px;
            overflow: hidden;
        }

        .cyber-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(0, 102, 204, 0.2),
                transparent
            );
            animation: cyber-scan 3s linear infinite;
        }

        @keyframes cyber-scan {
            0% { left: -100%; }
            100% { left: 200%; }
        }

        /* buttons with cyber effect */
        .stButton > button {
            position: relative;
            overflow: hidden;
            background: linear-gradient(45deg, rgba(0, 10, 30, 0.9), rgba(0, 30, 60, 0.9)) !important;
            border: 1px solid #0066cc !important;
            color: #fff !important;
            transition: all 0.3s ease !important;
        }

        .stButton > button::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: conic-gradient(
                transparent,
                transparent,
                transparent,
                #0066cc
            );
            animation: btn-rotate 4s linear infinite;
        }

        .stButton > button::after {
            content: '';
            position: absolute;
            inset: 3px;
            background: inherit;
            border-radius: 8px;
            transition: 0.5s;
        }

        @keyframes btn-rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Animated sidebar */
        .css-1d391kg {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(0, 10, 30, 0.95), rgba(0, 20, 40, 0.95)) !important;
        }

        .css-1d391kg::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 30%, rgba(79, 70, 229, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(6, 182, 212, 0.08) 0%, transparent 50%);
            animation: sidebar-glow 4s ease-in-out infinite;
        }

        @keyframes sidebar-glow {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }

        /* Cool file upload animation */
        [data-testid="stFileUploader"] {
            position: relative;
            overflow: hidden;
            background: rgba(0, 10, 30, 0.7) !important;
        }

        [data-testid="stFileUploader"]::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #0066cc, transparent, #0066cc);
            background-size: 400% 400%;
            animation: upload-border 3s ease infinite;
            z-index: -1;
        }

        @keyframes upload-border {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Glowing success messages */
        .success-message {
            position: relative;
            overflow: hidden;
            padding: 10px;
            background: rgba(0, 102, 204, 0.1);
            border-radius: 8px;
            animation: success-pulse 2s ease-in-out infinite;
        }

        @keyframes success-pulse {
            0%, 100% { box-shadow: 0 0 10px rgba(0, 102, 204, 0.3); }
            50% { box-shadow: 0 0 20px rgba(0, 102, 204, 0.5); }
        }

        /* text inputs */
        .stTextArea textarea, .stTextInput input {
            background: rgba(0, 10, 30, 0.7) !important;
            border: 1px solid rgba(0, 102, 204, 0.3) !important;
            transition: all 0.3s ease !important;
        }

        .stTextArea textarea:focus, .stTextInput input:focus {
            background: rgba(0, 20, 40, 0.8) !important;
            border-color: #0066cc !important;
            box-shadow: 0 0 15px rgba(0, 102, 204, 0.3),
                        inset 0 0 10px rgba(0, 102, 204, 0.1) !important;
        }

        /* Cool progress animation */
        .stProgress > div > div {
            background: linear-gradient(90deg, 
                rgba(0, 102, 204, 0.7),
                rgba(0, 102, 204, 1),
                rgba(0, 102, 204, 0.7)
            ) !important;
            background-size: 200% 100% !important;
            animation: progress-slide 2s linear infinite !important;
        }

        @keyframes progress-slide {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* Add subtle particle effect */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 30%, rgba(79, 70, 229, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, var(--dark-bg) 0%, #020617 100%);
            animation: bg-pulse 8s ease-in-out infinite;
            z-index: -2;
        }

        .stApp::after {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='50' cy='50' r='1' fill='rgba(129, 140, 248, 0.2)'/%3E%3C/svg%3E"),
                      url("data:image/svg+xml,%3Csvg width='150' height='150' viewBox='0 0 150 150' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='75' cy='75' r='1' fill='rgba(103, 232, 249, 0.1)'/%3E%3C/svg%3E");
            opacity: 0.6;
            animation: particle-drift 20s linear infinite;
            z-index: -1;
        }

        @keyframes bg-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }

        @keyframes particle-drift {
            0% {
                background-position: 0% 0%, 0% 0%;
            }
            100% {
                background-position: 100% 100%, -100% -100%;
            }
        }

        /* the overall depth with subtle shadow */
        .stApp {
            background: transparent !important;
        }

        .main {
            background: transparent !important;
        }

        /* text styling */
        .subheader {
            color: var(--neon-blue) !important;
            text-shadow: var(--neon-glow) !important;
            font-weight: 600 !important;
            letter-spacing: 1px !important;
            margin-bottom: 10px !important;
        }

        /* Labels and text inputs with neon effect */
        .stSelectbox label, .stTextArea label, .stFileUploader label {
            color: var(--neon-blue) !important;
            text-shadow: var(--neon-glow) !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
        }

        /* markdown text */
        .element-container div.stMarkdown p {
            color: rgba(255, 255, 255, 0.9) !important;
            text-shadow: 0 0 5px rgba(0, 102, 204, 0.2) !important;
        }

        /* Section headers with neon effect */
        h2, h3, h4 {
            color: var(--neon-blue) !important;
            text-shadow: var(--neon-glow) !important;
            font-weight: 600 !important;
            letter-spacing: 1px !important;
            margin: 15px 0 !important;
        }

        /* radio buttons */
        .stRadio > label {
            color: var(--neon-blue) !important;
            text-shadow: var(--neon-glow) !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
        }

        /* selectbox */
        .stSelectbox > div > div {
            background: rgba(0, 10, 30, 0.7) !important;
            border: 1px solid var(--neon-blue) !important;
            box-shadow: var(--neon-glow) !important;
        }

        /* multiselect */
        .stMultiSelect > label {
            color: var(--neon-blue) !important;
            text-shadow: var(--neon-glow) !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
        }

        /* Success messages with neon effect */
        .element-container div.stMarkdown p.success-message {
            color: var(--neon-blue) !important;
            text-shadow: var(--neon-glow) !important;
            animation: success-pulse 2s ease-in-out infinite !important;
        }

        /* Warning messages with neon effect */
        .stWarning {
            background: rgba(245, 158, 11, 0.1) !important;
            border: 1px solid var(--warning) !important;
            box-shadow: 0 0 10px rgba(245, 158, 11, 0.2) !important;
        }

        /* Error messages with neon effect */
        .stError {
            background: rgba(239, 68, 68, 0.1) !important;
            border: 1px solid #ef4444 !important;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.2) !important;
        }

        /* Analysis results with neon effect */
        .element-container div.stMarkdown pre {
            background: rgba(0, 10, 30, 0.7) !important;
            border: 1px solid var(--neon-blue) !important;
            box-shadow: var(--neon-glow) !important;
            padding: 15px !important;
            border-radius: 8px !important;
            color: rgba(255, 255, 255, 0.9) !important;
            text-shadow: 0 0 5px rgba(0, 102, 204, 0.2) !important;
        }

        /* expander */
        .streamlit-expanderHeader {
            color: var(--neon-blue) !important;
            text-shadow: var(--neon-glow) !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
        }

        /* download button */
        .stDownloadButton > button {
            background: linear-gradient(45deg, rgba(0, 10, 30, 0.9), rgba(0, 30, 60, 0.9)) !important;
            border: 1px solid var(--neon-blue) !important;
            box-shadow: var(--neon-glow) !important;
            color: var(--neon-blue) !important;
            text-shadow: var(--neon-glow) !important;
        }

        .stDownloadButton > button:hover {
            background: var(--neon-blue) !important;
            color: white !important;
            box-shadow: 0 0 20px rgba(0, 102, 204, 0.4) !important;
        }
        
        /* Main action buttons with animation */
        .stButton > button[kind="primary"] {
            background: linear-gradient(45deg, var(--neon-blue), #1a56db) !important;
            border: 2px solid var(--neon-blue) !important;
            color: white !important;
            font-weight: 600 !important;
            letter-spacing: 1px !important;
            padding: 0.75rem 2rem !important;
            position: relative !important;
            overflow: hidden !important;
            transition: all 0.3s ease !important;
            transform-style: preserve-3d !important;
            box-shadow: 0 0 15px rgba(0, 102, 204, 0.3) !important;
        }

        .stButton > button[kind="primary"]::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(
                120deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            ) !important;
            animation: shine 3s infinite !important;
        }

        @keyframes shine {
            0% {
                left: -100%;
                opacity: 0.8;
            }
            20% {
                left: 100%;
                opacity: 0.8;
            }
            100% {
                left: 100%;
                opacity: 0;
            }
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 0 30px rgba(0, 102, 204, 0.5) !important;
            background: linear-gradient(45deg, #1a56db, var(--neon-blue)) !important;
        }

        .stButton > button[kind="primary"]:active {
            transform: translateY(1px) !important;
            box-shadow: 0 0 20px rgba(0, 102, 204, 0.4) !important;
        }

        /* Pulsing animation for button text */
        .stButton > button[kind="primary"] span {
            position: relative !important;
            z-index: 1 !important;
            animation: pulse 2s infinite !important;
        }

        @keyframes pulse {
            0%, 100% {
                text-shadow: 0 0 10px rgba(255, 255, 255, 0.5),
                           0 0 20px rgba(255, 255, 255, 0.3);
            }
            50% {
                text-shadow: 0 0 15px rgba(255, 255, 255, 0.7),
                           0 0 25px rgba(255, 255, 255, 0.5);
            }
        }

        /* Loading animation when button is clicked */
        .stButton > button[kind="primary"].loading::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, 
                transparent 25%, 
                rgba(255, 255, 255, 0.1) 50%, 
                transparent 75%
            );
            background-size: 200% 200%;
            animation: loading 1s infinite linear;
        }

        @keyframes loading {
            0% {background-position: 200% 200%;}
            100% {background-position: 0 0;}}

        /* Cyber border effect */
        .stButton > button[kind="primary"]::after {
            content: '';
            position: absolute;
            inset: -2px;
            background: linear-gradient(45deg,
                var(--neon-blue),
                #60a5fa,
                var(--neon-blue));
            filter: blur(5px);
            z-index: -1;
            animation: borderGlow 3s infinite;}

        @keyframes borderGlow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }}
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 20px;'>
                <h1 class='title-glow' style='margin: 0; font-size: 26px;'>SMART JOB ASISTANT</h1>
                <p style='margin: 0; background: linear-gradient(120deg, #818CF8, #67E8F9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 11px; text-transform: uppercase; letter-spacing: 2px;'>
                    Smart fusion of Gemini and Groq API's
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Page Selection with cool icons
        page = st.radio("NAVIGATE",["Smart Resume Analyzer", "Smart Cold Mail Generator"],
            format_func=lambda x: f"📄 {x}" if x == "Smart Resume Analyzer" else f"✉️ {x}")
        
        # Language selector only for Resume Analyzer
        selected_language = "English"  # Default language for Cold Mail
        if page == "Smart Resume Analyzer":
            # Language selector with modern flags
            language = st.selectbox("🌐 Select Language",
                ["🇺🇸 English","🇮🇳 हिंदी","🇮🇳 తెలుగు"],index=0,
                help="Choose your preferred language for analysis")

            # Update language mapping
            language_mapping = {"🇺🇸 English": "English","🇮🇳 हिंदी": "हिंदी","🇮🇳 తెలుగు": "తెలుగు"}
            
            # Get the actual language key for prompts
            selected_language = language_mapping[language]

        if page == "Smart Resume Analyzer": 
            st.markdown("<p style='color: #0066cc; margin-top: 20px;'>Choose your preferred Analysis Type</p>", unsafe_allow_html=True)
            # Analysis type selector with cool badges
            analysis_types = st.multiselect("SELECT ANALYSIS MODULES",
                list(ATSAnalyzer.ANALYSIS_TYPES.keys()),
                default=["Complete Analysis"],
                format_func=lambda x: f"{'🔍' if 'Complete' in x else '🎯' if 'Skills' in x else '🤖' if 'ATS' in x else '📊'} {x}")
            
            # Model selection with tech badges
            model_choice = st.selectbox("SELECT AI MODEL",
                list(ATSAnalyzer.AI_MODELS.keys()),
                format_func=lambda x: ATSAnalyzer.AI_MODELS[x])
        else:
            st.markdown("<p style='color: #0066cc; margin-top: 20px;'>Choose your preferred AI MODEL</p>", unsafe_allow_html=True)
            
            # Model selection for cold mail
            model_choice = st.selectbox("SELECT AI MODEL",
                list(ATSAnalyzer.AI_MODELS.keys()),
                format_func=lambda x: ATSAnalyzer.AI_MODELS[x])

    # Get language-specific labels
    labels = ATSAnalyzer.LANGUAGE_PROMPTS[selected_language]["labels"] if page == "Smart Resume Analyzer" else ATSAnalyzer.LANGUAGE_PROMPTS["English"]["labels"]
    
    if page == "Smart Resume Analyzer":
        # Futuristic Header for Resume Analyzer
        st.markdown('''
            <div class="cyber-card" style="text-align: center; padding: 2rem; margin-bottom: 2rem;">
                <h1 class="title-glow">📄 SMART RESUME ANALYZER</h1>
                <p style="color: #0066cc; text-transform: uppercase; letter-spacing: 2px;">
                    🎯 Smart Analysis • 🔍 Deep Insights • ⚡ Quick Results
                </p>
            </div>
        ''', unsafe_allow_html=True)
        
        # Resume Analyzer page content
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📝 Job Description")
            job_title = st.text_input("Job Title", placeholder="e.g., Software Engineer (optional)")
            job_description = st.text_area("Job Description",
                height=200,
                placeholder="Paste the job description here...")

        with col2:
            st.subheader("📄 Your Resume")
            uploaded_file = st.file_uploader("Upload your resume (PDF, DOC, DOCX)",
                type=["pdf", "doc", "docx"])

            if uploaded_file:
                st.markdown(f'<p class="success-message">✅ {uploaded_file.name} uploaded successfully!</p>', unsafe_allow_html=True)

        # Analysis section
        if uploaded_file and job_description and analysis_types:
            if st.button(labels["analyze"], use_container_width=True):
                doc_text = ATSAnalyzer.extract_text(uploaded_file)
                
                if doc_text:
                    for analysis_type in analysis_types:
                        with st.spinner(f"Performing {analysis_type}..."):
                            # Get analysis prompt
                            analysis_prompt = ATSAnalyzer.ANALYSIS_TYPES[analysis_type]
                            
                            # Get response
                            response = ATSAnalyzer.get_ai_response(model_choice,analysis_prompt,doc_text,job_description,selected_language)
                            
                            if response:
                                # Use the class method instead of global function
                                analysis_data = ATSAnalyzer.extract_data_from_response(response)
                                if analysis_data:
                                    st.markdown("## 📊 Analysis Results")
                                    
                                    # Display analysis results in text format
                                    st.markdown("### 📝 Detailed Analysis")
                                    st.markdown(response)
                    
                    # Download button for complete analysis
                    st.download_button("📥 Download Complete Analysis",
                        "\n\n".join([f"=== {at} ===\n{ATSAnalyzer.get_ai_response(model_choice, ATSAnalyzer.ANALYSIS_TYPES[at], doc_text, job_description, selected_language)}" for at in analysis_types]),
                        file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain")

    else:
        # Futuristic Header for Cold Mail Generator
        st.markdown('''
            <div class="cyber-card" style="text-align: center; padding: 2rem; margin-bottom: 2rem;">
                <h1 class="title-glow">✉️ SMART COLD MAIL GENERATOR</h1>
                <p style="color: #0066cc; text-transform: uppercase; letter-spacing: 2px;">
                    💼 Professional • 🎯 Targeted • ✨ Impactful
                </p>
            </div>''', unsafe_allow_html=True)
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Job Description Input
            st.markdown("### 📝 Job Description")
            job_description = st.text_area(
                "Paste the job description here",
                height=300,
                help="Paste the complete job description to generate a tailored cold mail")
        
        with col2:
            # Resume Upload
            st.markdown("### 📄 Your Resume")
            uploaded_resume = st.file_uploader("Upload your resume",
                type=["pdf", "doc", "docx"],
                help="Upload your resume to personalize the cold mail")
            
            if uploaded_resume:
                st.success(f"✅ Resume uploaded: {uploaded_resume.name}")
        
        # Cold Mail Type Selection with descriptions
        st.markdown("### 📋 Select Cold Mail Style")
        cold_mail_type = st.selectbox("Choose your preferred style",
            list(ATSAnalyzer.COLD_MAIL_TYPES.keys()),
            format_func=lambda x: f"{x} - {ATSAnalyzer.COLD_MAIL_TYPES[x]['description']}")

        # Personal Information
        with st.expander("✍🏽Enter Your Personal Information (Optional)"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Your Full Name", placeholder="Your Name")
                email = st.text_input("Email Address", placeholder="YourName@email.com")
                university = st.text_input("University/College Name", placeholder="Your University/College Name")
                
            with col2:
                phone = st.text_input("Phone Number", placeholder="98XXXXXXXX")
                linkedin = st.text_input("LinkedIn Profile URL", placeholder="https://linkedin.com/in/yourusername")
                degree = st.text_input("Degree & Year", placeholder="3rd Year, B.Sc.Stream")

        # Generate Button
        if uploaded_resume and job_description:
            if st.button("Generate Cold Mail", use_container_width=True):
                doc_text = ATSAnalyzer.extract_text(uploaded_resume)
                
                if doc_text:
                    with st.spinner("📨 Crafting your personalized cold mail..."):
                        # Get the selected template
                        template = ATSAnalyzer.COLD_MAIL_TYPES[cold_mail_type]["template"]
                        
                        # Generate cold mail
                        cold_mail = ATSAnalyzer.generate_cold_mail(
                            model_choice=model_choice,
                            prompt=template,
                            resume_text=doc_text,
                            job_description=job_description,
                            personal_info={"name": name,
                                "email": email,
                                "phone": phone,
                                "university": university,
                                "linkedin": linkedin,
                                "degree": degree})
                        
                        if cold_mail:
                            st.markdown("### 📧 Your Generated Cold Mail")
                            st.markdown('''
                                <div class="glass-card" style="padding: 2rem; margin-top: 1rem;">
                                    <pre style="white-space: pre-wrap; word-wrap: break-word;">{}</pre>
                                </div>
                            '''.format(cold_mail), unsafe_allow_html=True)
                            
                            # Download button
                            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"cold_mail_{current_time}.txt"
                            
                            st.download_button(label="💾 Download Cold Mail",
                                data=cold_mail,
                                file_name=filename,
                                mime="text/plain",
                                use_container_width=True)
        
    # Footer with futuristic style
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <div style='color: #0066cc; font-size: 12px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 5px;'>
                ⚠️ AI can make mistakes.Please Double Check the Response
            </div>
            <div style='font-size: 10px; color: rgba(255,255,255,0.7);'>
                💼 Smart Job Assistant • © 2025 A.Raghu Ram. All rights reserved.
            </div>
        </div>
        """,
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
