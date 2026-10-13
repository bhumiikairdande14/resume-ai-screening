from flask import Flask, render_template, request
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

app = Flask(__name__)

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------------------------------
# SKILLS DATABASE
# -------------------------------------------------

SKILLS = [
    "python",
    "java",
    "c++",
    "c",
    "javascript",
    "html",
    "css",
    "react",
    "angular",
    "node.js",
    "node",
    "flask",
    "django",

    "sql",
    "mysql",
    "mongodb",
    "postgresql",

    "machine learning",
    "deep learning",
    "artificial intelligence",
    "nlp",
    "natural language processing",
    "data science",

    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",

    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "google cloud",

    "power bi",
    "tableau",
    "excel",

    "project management",
    "agile",
    "scrum",
    "jira",
    "trello",
    "leadership",
    "communication",
    "team management",
    "risk management",
    "time management",
    "problem solving",

    "rest api",
    "api",
    "figma",
    "ui",
    "ux"
]


# -------------------------------------------------
# NLP STOPWORDS
# -------------------------------------------------

STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an",
    "and", "or", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "this",
    "that", "these", "those", "be", "has", "have",
    "had", "will", "can", "should", "our", "your",
    "their", "they", "we", "you", "he", "she",
    "it", "its", "into", "about", "who", "which",
    "using", "looking", "candidate"
}


# -------------------------------------------------
# PDF TEXT EXTRACTION
# -------------------------------------------------

def extract_text_from_pdf(file_path):

    try:

        reader = PdfReader(file_path)

        text = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text.append(page_text)

        final_text = "\n".join(text)

        print("PDF pages:", len(reader.pages))
        print("Extracted characters:", len(final_text))

        return final_text.strip()

    except Exception as error:

        print("PDF extraction error:", error)

        return ""


# -------------------------------------------------
# NLP PREPROCESSING
# -------------------------------------------------

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove unwanted characters
    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    # Tokenization
    tokens = text.split()

    # Stopword removal
    tokens = [
        word
        for word in tokens
        if word not in STOPWORDS
    ]

    # Join tokens
    cleaned_text = " ".join(tokens)

    # Remove extra spaces
    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text
    )

    return cleaned_text.strip()


# -------------------------------------------------
# SKILL EXTRACTION
# -------------------------------------------------

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, text):

            found_skills.append(skill)

    return sorted(set(found_skills))


# -------------------------------------------------
# TF-IDF + COSINE SIMILARITY
# -------------------------------------------------

def calculate_similarity(resume_text, job_description):

    resume_clean = preprocess_text(resume_text)

    job_clean = preprocess_text(job_description)

    documents = [
        resume_clean,
        job_clean
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )

    score = similarity[0][0] * 100

    return round(score, 2)


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        # Get job description
        job_description = request.form.get(
            "job_description",
            ""
        ).strip()

        # Get uploaded file
        resume = request.files.get("resume")

        # -----------------------------------------
        # VALIDATION
        # -----------------------------------------

        if not job_description:

            return render_template(
                "index.html",
                error="Please enter a job description."
            )

        if resume is None:

            return render_template(
                "index.html",
                error="Please upload a resume."
            )

        if resume.filename == "":

            return render_template(
                "index.html",
                error="Please select a resume file."
            )

        # Only PDF
        if not resume.filename.lower().endswith(".pdf"):

            return render_template(
                "index.html",
                error="Please upload a PDF resume."
            )

        # -----------------------------------------
        # SAVE FILE
        # -----------------------------------------

        filename = resume.filename

        # Remove unsafe characters
        safe_filename = re.sub(
            r"[^a-zA-Z0-9_.-]",
            "_",
            filename
        )

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            safe_filename
        )

        resume.save(file_path)

        # -----------------------------------------
        # EXTRACT RESUME TEXT
        # -----------------------------------------

        resume_text = extract_text_from_pdf(
            file_path
        )

        # -----------------------------------------
        # PDF EXTRACTION ERROR
        # -----------------------------------------

        if not resume_text:

            return render_template(
                "index.html",
                error=(
                    "Could not extract text from this PDF. "
                    "Please upload a text-based PDF resume."
                )
            )

        # -----------------------------------------
        # NLP PROCESSING
        # -----------------------------------------

        cleaned_resume = preprocess_text(
            resume_text
        )

        cleaned_job = preprocess_text(
            job_description
        )

        # -----------------------------------------
        # SKILL EXTRACTION
        # -----------------------------------------

        resume_skills = extract_skills(
            resume_text
        )

        job_skills = extract_skills(
            job_description
        )

        # -----------------------------------------
        # MATCHED / MISSING SKILLS
        # -----------------------------------------

        matched_skills = sorted(
            set(resume_skills)
            & set(job_skills)
        )

        missing_skills = sorted(
            set(job_skills)
            - set(resume_skills)
        )

        # -----------------------------------------
        # SKILL SCORE
        # -----------------------------------------

        if len(job_skills) > 0:

            skill_score = (
                len(matched_skills)
                / len(job_skills)
            ) * 100

        else:

            skill_score = 0

        skill_score = round(
            skill_score,
            2
        )

        # -----------------------------------------
        # TF-IDF SCORE
        # -----------------------------------------

        try:

            similarity_score = calculate_similarity(
                resume_text,
                job_description
            )

        except Exception as error:

            print("Similarity error:", error)

            similarity_score = 0

        # -----------------------------------------
        # FINAL SCORE
        # -----------------------------------------

        final_score = round(
            (similarity_score * 0.70)
            +
            (skill_score * 0.30),
            2
        )

        # -----------------------------------------
        # SCREENING DECISION
        # -----------------------------------------

        if final_score >= 70:

            status = "Shortlisted"
            status_class = "success"

        elif final_score >= 50:

            status = "Under Consideration"
            status_class = "warning"

        else:

            status = "Not Shortlisted"
            status_class = "danger"

        # -----------------------------------------
        # RESULT PAGE
        # -----------------------------------------

        return render_template(
            "result.html",

            filename=safe_filename,

            final_score=final_score,

            similarity_score=similarity_score,

            skill_score=skill_score,

            matched_skills=matched_skills,

            missing_skills=missing_skills,

            status=status,

            status_class=status_class
        )

    # GET REQUEST
    return render_template(
        "index.html"
    )


# -------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )