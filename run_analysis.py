
import argparse
import json
import os
from datetime import datetime
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# --- 1. CONFIGURATION & MODEL LOADING ---

# Load the semantic search model for ranking chunks
print("Loading semantic search model (all-MiniLM-L6-v2)...")
ranking_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the generative model for creating titles
# This runs locally and is great for high-quality summarization/titling
print("Loading title generation model (google/flan-t5-base)...")
title_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
title_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# --- 2. CORE FUNCTIONS ---

def load_input(path):
    """Loads the input JSON configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_sections_from_pdf(pdf_path):
    """Extracts text from a PDF, splitting it into paragraphs (chunks)."""
    sections = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=2, y_tolerance=5) # Tune tolerances for better layout detection
                if text and text.strip():
                    # Splitting by double newline is a good heuristic for paragraphs
                    for para in text.strip().split("\n\n"):
                        if len(para.strip()) > 50: # Ignore very short paragraphs/headings
                            sections.append({
                                "page_number": i + 1,
                                "content": para.strip().replace('\n', ' ') # Normalize newlines within a para
                            })
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return sections

def filter_chunks_by_keywords(sections, forbidden_keywords):
    """
    (STAGE 1 of 2-Stage Retrieval)
    Removes any section that contains a forbidden keyword. Critical for negative constraints.
    """
    filtered_sections = []
    for section in sections:
        content_lower = section['content'].lower()
        if not any(keyword in content_lower for keyword in forbidden_keywords):
            filtered_sections.append(section)
    return filtered_sections

def rank_sections(sections, task_description, model):
    """
    (STAGE 2 of 2-Stage Retrieval)
    Ranks sections based on semantic similarity to the task description.
    """
    if not sections:
        return []
    
    query_embedding = model.encode(task_description, convert_to_tensor=True)
    section_texts = [s["content"] for s in sections]
    section_embeddings = model.encode(section_texts, convert_to_tensor=True, show_progress_bar=True)
    
    scores = util.cos_sim(query_embedding, section_embeddings)[0]
    
    for i, section in enumerate(sections):
        section["score"] = float(scores[i])
        
    return sorted(sections, key=lambda x: x["score"], reverse=True)

def generate_title_with_llm(content):
    """
    Generates a concise, descriptive title for a text chunk using a local T5 model.
    """
    prompt = f"Generate a short, descriptive title for the following text: \"{content[:512]}\"" # Use first 512 chars
    
    try:
        inputs = title_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = title_model.generate(**inputs, max_new_tokens=20, num_beams=4, early_stopping=True)
        title = title_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return title.strip()
    except Exception as e:
        print(f"Error generating title: {e}")
        # Fallback to a simpler heuristic if LLM fails
        return content.strip().split('\n')[0][:80] # First line, max 80 chars


def build_output(input_data, ranked_sections, top_k):
    """Constructs the final JSON output from the ranked sections."""
    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_data["documents"]],
            "persona": input_data["persona"]["role"],
            "job_to_be_done": input_data["job_to_be_done"]["task"],
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    
    seen_content = set()
    rank = 1
    
    print(f"\nGenerating titles for top {top_k} results...")
    for section in tqdm(ranked_sections, total=len(ranked_sections)):
        if rank > top_k:
            break

        content = section["content"]
        if content in seen_content:
            continue
        seen_content.add(content)
        
        # Generate a high-quality title using the LLM
        title = generate_title_with_llm(content)
        
        output["extracted_sections"].append({
            "document": section["filename"],
            "section_title": title,
            "importance_rank": rank,
            "page_number": section["page_number"]
        })
        
        output["subsection_analysis"].append({
            "document": section["filename"],
            "refined_text": content, # Keeping full content for context
            "page_number": section["page_number"]
        })
        
        rank += 1
        
    return output


# --- 3. MAIN ORCHESTRATOR ---

def main():
    """Main function to run the analysis pipeline."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process a document collection based on a persona and task.")
    parser.add_argument('collection_dir', type=str, help='The path to the collection directory (e.g., ./Collection_1/)')
    parser.add_argument('--top_k', type=int, default=15, help='Number of top results to return.')
    args = parser.parse_args()

    collection_path = args.collection_dir
    input_json_path = os.path.join(collection_path, 'challenge1b_input.json')
    output_json_path = os.path.join(collection_path, 'challenge1b_output.json')
    pdf_dir = os.path.join(collection_path, 'PDFs')

    if not os.path.isdir(collection_path):
        print(f"❌ Error: Directory not found at '{collection_path}'")
        return

    # --- Load input and define task ---
    input_data = load_input(input_json_path)
    task = input_data["job_to_be_done"]["task"]
    challenge_id = input_data.get("challenge_info", {}).get("challenge_id")
    print(f"\n🚀 Starting analysis for Challenge: {challenge_id}...")
    print(f"Task: {task}")
    
    # --- Global Extraction: Get all chunks from all PDFs ---
    all_sections = []
    print("\n[Step 1/4] Extracting text from all PDF documents...")
    for doc in tqdm(input_data["documents"], desc="Parsing PDFs"):
        pdf_path = os.path.join(pdf_dir, doc["filename"])
        if os.path.exists(pdf_path):
            sections = extract_sections_from_pdf(pdf_path)
            for section in sections:
                section['filename'] = doc["filename"] # Tag each chunk with its source
            all_sections.extend(sections)
        else:
            print(f"⚠️ Warning: PDF file not found at {pdf_path}")

    if not all_sections:
        print("❌ No content could be extracted from any PDF. Exiting.")
        return
    print(f"✅ Extracted {len(all_sections)} total text chunks.")

    # --- Conditional Filtering (for Recipe Challenge) ---
    print("\n[Step 2/4] Applying filters...")
    if challenge_id == "round_1b_001":
        print("   -> Applying 'vegetarian' keyword filter for recipe challenge.")
        non_veg_keywords = ["sausage", "bacon", "beef", "pork", "chicken", "meat", "pancetta", "prosciutto", "ground beef"]
        all_sections = filter_chunks_by_keywords(all_sections, non_veg_keywords)
        print(f"   -> Filtered down to {len(all_sections)} vegetarian-safe chunks.")
    else:
        print("   -> No special filters applied for this challenge.")

    # --- Global Ranking ---
    print("\n[Step 3/4] Ranking all chunks against the task...")
    globally_ranked_sections = rank_sections(all_sections, task, ranking_model)
    print("✅ Ranking complete.")

    # --- Build and Save Final Output ---
    print("\n[Step 4/4] Building final output file...")
    output = build_output(input_data, globally_ranked_sections, top_k=args.top_k)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 Success! Final analysis saved to {output_json_path}")


if __name__ == "__main__":
    main()

