import subprocess
import typer
import json
import shutil
import logging
import sys
import re
import os
from pathlib import Path
from huggingface_hub import snapshot_download, login, HfApi
from llama_cpp import Llama 
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = typer.Typer()

# ✅ Fix logging so messages always appear
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]  # ✅ Ensure logs appear in the terminal
)
logger = logging.getLogger(__name__)

# ✅ Ensure logs appear instantly without buffering
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODEL_CACHE_DIR = Path.home() / ".starfishdata/models"
DEFAULT_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct-GGUF"  # ✅ Hugging Face repo name
DEFAULT_MODEL_FILE = "qwen2-0_5b-instruct-q2_k.gguf"
SMALLEST_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct-GGUF"
SMALLEST_MODEL_FILE = "qwen2-0_5b-instruct-q2_k.gguf"  # ✅ Smallest GGUF model
MAX_RETRIES = 3

def jaccard_similarity(q1: str, q2: str):
    """Calculate Jaccard similarity between two questions."""
    set1, set2 = set(q1.lower().split()), set(q2.lower().split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0  # Avoid division by zero

def is_duplicate_tfidf(new_qa, existing_questions, threshold=0.8):
    """Check if the new Q&A question is too similar using TF-IDF & cosine similarity."""
    if not existing_questions:
        return False  # No existing questions

    vectorizer = TfidfVectorizer()
    all_questions = existing_questions + [new_qa["question"]]
    tfidf_matrix = vectorizer.fit_transform(all_questions)

    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return max(similarities[0]) >= threshold  # If max similarity is above threshold, it's a duplicate


def is_duplicate(new_qa, existing_questions, jaccard_threshold=0.7, tfidf_threshold=0.8):
    """Combines Jaccard & TF-IDF similarity for better duplicate detection."""
    if not existing_questions:
        return False

    # Jaccard check
    for existing_q in existing_questions:
        if jaccard_similarity(new_qa["question"], existing_q) >= jaccard_threshold:
            return True
    
    # TF-IDF check
    return is_duplicate_tfidf(new_qa, existing_questions, threshold=tfidf_threshold)


def get_model_path(model_name: str, model_file: str) -> Path:
    """Returns the full path where the model is stored."""
    return MODEL_CACHE_DIR / model_file


def model_exists(model_name: str, model_file: str) -> bool:
    """Checks if a model exists locally."""
    return get_model_path(model_name, model_file).exists()


def cleanup_failed_download(model_name: str):
    """Delete partially downloaded model files if a failure occurs."""
    model_dir = MODEL_CACHE_DIR / model_name
    if model_dir.exists():
        logger.warning(f"🗑 Removing incomplete downloads from {model_dir}...")
        shutil.rmtree(model_dir)
        logger.info("✅ Cleanup completed. Try downloading again.")
    else:
        logger.info("ℹ️ No partial downloads found.")


def validate_hf_token(hf_token: str):
    """Validate the Hugging Face token before proceeding."""
    try:
        api = HfApi()
        user_info = api.whoami(token=hf_token)
        logger.info(f"🔑 Authenticated as: {user_info['name']}")
    except Exception as e:
        logger.error(f"❌ Invalid Hugging Face token: {e}")
        sys.exit(1)


def download_model(model_name: str, model_file: str, hf_token: str):
    """Download a model from Hugging Face using huggingface_hub."""
    model_path = get_model_path(model_name, model_file)

    if model_path.exists():
        logger.info(f"✅ Model {model_name} already exists at {model_path}.")
        return

    if not hf_token:
        logger.error("❌ This model requires authentication. Please provide a Hugging Face token.")
        logger.info("🔑 Get a token here: https://huggingface.co/settings/tokens")
        sys.exit(1)

    try:
        # Validate the token first
        validate_hf_token(hf_token)

        # Authenticate with Hugging Face
        login(token=hf_token)

        logger.info(f"📥 Downloading {model_name} GGUF model... This may take time.")
        snapshot_download(repo_id=model_name, local_dir=str(MODEL_CACHE_DIR))  # ✅ Removed timeout argument
        logger.info(f"✅ Model downloaded successfully to {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to download the model: {e}")
        cleanup_failed_download(model_name)  # 🔹 Cleanup on failure
        sys.exit(1)


@app.command()
def download(
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, "--name", "-n", help="Hugging Face model name."),
    model_file: str = typer.Option(DEFAULT_MODEL_FILE, "--file", "-f", help="The specific GGUF model file to use."),
    hf_token: str = typer.Option(..., "--hf-token", "-t", help="Hugging Face API token (required)."),
):
    """Download a model from Hugging Face and store it locally."""
    download_model(model_name, model_file, hf_token)


def extract_question_answer(text: str):
    """Extracts the question and answer from model-generated text."""
    match = re.search(r"Question:\s*(.+?)\s*Answer:\s*(.+)", text, re.DOTALL)
    if match:
        return {"question": match.group(1).strip(), "answer": match.group(2).strip()}
    return None

    
def generate_single_qa(model, prompt: str, record_num: int, retry_count: int = 0):
    """Generate a single valid Q&A pair with retries."""
    
    input_text = (
        f"You are an AI assistant generating question-answer pairs about: '{prompt}'.\n"
        "DO NOT generate general knowledge questions. Stick **ONLY** to this topic.\n\n"
        "Your response must follow **EXACTLY** this format:\n"
        "Question: <your generated question about the topic>\n"
        "Answer: <your generated answer about the topic>\n\n"
        "Strict Rules:\n"
        "1️⃣ The question **must** include the topic or be directly about it.\n"
        "2️⃣ The answer **must** be relevant and provide specific details.\n"
        "3️⃣ Do **not** include placeholders like '<your generated question>' or '<your generated answer>'.\n"
        "4️⃣ If the question is unrelated to the topic, regenerate it.\n"
        "5️⃣ Do **not** repeat previous questions.\n\n"
        f"Now, generate a new Q&A about '{prompt}' following the rules above."
    )
    
    logger.info(f"🛠 Generating record {record_num} (Attempt {retry_count + 1}/{MAX_RETRIES})...")

    output = model(input_text, max_tokens=300)

    generated_text = output["choices"][0]["text"].strip()

    qa_pair = extract_question_answer(generated_text)

    if qa_pair:
        logger.info(f"✅ Record {record_num} generated: {qa_pair['question']}")  # ✅ Logs question immediately
        return qa_pair
    elif retry_count < MAX_RETRIES - 1:
        logger.warning(f"⚠️ Invalid output. Retrying record {record_num} ({retry_count + 1}/{MAX_RETRIES})...")
        return generate_single_qa(model, prompt, record_num, retry_count + 1)  # ✅ Retry generating the Q&A
    else:
        logger.error(f"❌ Failed to generate valid Q&A after {MAX_RETRIES} retries. Skipping record {record_num}.")
        return None


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Topic to generate synthetic Q&A data."),
    num_records: int = typer.Option(1, "--num-records", "-n", help="Number of Q&A pairs to generate (max: 100)."),
    model_file: str = typer.Option(DEFAULT_MODEL_FILE, "--file", "-f", help="The specific GGUF model file to use."),
    output_file: str = typer.Option("output.jsonl", "--output-file", "-o", help="Path to save the output JSONL file."),
    cleanup: bool = typer.Option(False, "--cleanup", help="Automatically delete the model after inference."),
):
    """Generate synthetic Q&A data related to the given topic and save to a JSONL file."""

    if num_records > 100:
        logger.error("❌ The maximum number of records you can generate is 100.")
        sys.exit(1)

    model_path = get_model_path("", model_file)

    if not model_exists("", model_file):
        logger.error(f"❌ No model found at {model_path}. Please run 'starfishdata download' first.")
        sys.exit(1)

    logger.info(f"🚀 Starting generation of {num_records} Q&A pairs for topic: '{prompt}'...")

    # ✅ Load Llama model
    model = Llama(model_path=str(model_path), n_ctx=512, n_batch=4, verbose=False)  # No embeddings

    results = []
    existing_questions = []  # Store only questions for comparison

    with open(output_file, "w") as jsonl_file:
        generated_count = 0
        while generated_count < num_records:
            logger.info(f"⏳ Processing record {generated_count + 1}/{num_records}...")

            attempt = 0
            while attempt < MAX_RETRIES:
                qa_pair = generate_single_qa(model, prompt, generated_count + 1)

                if qa_pair and not is_duplicate(qa_pair, existing_questions):
                    results.append(qa_pair)
                    jsonl_file.write(json.dumps(qa_pair) + "\n")

                    existing_questions.append(qa_pair["question"])  # Store only questions
                    generated_count += 1
                    break
                else:
                    logger.warning(f"⚠️ Duplicate detected. Retrying record {generated_count + 1}... ({attempt + 1}/{MAX_RETRIES})")
                    attempt += 1

            if attempt == MAX_RETRIES:
                logger.error(f"❌ Failed to generate unique Q&A after {MAX_RETRIES} retries. Skipping record {generated_count + 1}.")


    logger.info(f"📄 Output saved to: {output_file}")

    if cleanup:
        cleanup_models()


@app.command()
def cleanup_models():
    """Delete all locally downloaded models."""
    if MODEL_CACHE_DIR.exists():
        logger.warning(f"🗑 Deleting all downloaded models in {MODEL_CACHE_DIR}...")
        shutil.rmtree(MODEL_CACHE_DIR)
        logger.info("✅ All models have been deleted.")
    else:
        logger.info("ℹ️ No cached models found.")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
