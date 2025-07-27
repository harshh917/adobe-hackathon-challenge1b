# Approach for Persona-Driven Document Intelligence (Challenge 1B)

Our solution is designed as a robust, multi-stage pipeline that goes beyond simple semantic search to deliver highly relevant, persona-driven insights from document collections.

### Core Problem with Standard AI Search

A standard semantic search approach is effective at finding text with similar meaning but often fails at understanding critical negative constraints. For instance, in the Recipe Collection task, a simple search for a "vegetarian buffet" would incorrectly rank recipes containing "bacon" or "sausage" highly, as these terms are semantically related to food. This is a critical failure.

### Our Advanced Two-Stage Retrieval Pipeline

To solve this, we developed a sophisticated two-stage pipeline:

**Stage 1: Pre-retrieval Keyword Filtering**
Before any AI processing, the system first applies a rapid keyword filter. For the recipe challenge, it scans and removes any text chunk containing non-vegetarian words ("bacon", "beef", "chicken", etc.). This initial step ensures that only 100% relevant, "safe" content is passed to the next stage, perfectly handling the persona's constraints.

**Stage 2: Global Semantic Ranking**
The filtered, relevant chunks from *all* documents in the collection are pooled together. We then use a powerful Sentence Transformer model (`all-MiniLM-L6-v2`) to perform a global ranking. This ensures that the final output contains the absolute best sections from the entire document set, not just the top few from each individual document.

### AI-Powered Title Generation

To make the output clean and professional, we abandoned unreliable heuristics like using the first line as a title. Instead, we use a local generative model (`google/flan-t5-base`) to read the content of each top-ranked section and generate a concise, human-readable title. This transforms a messy list of text into a polished, actionable summary.

### Conclusion

By combining the precision of keyword filtering with the contextual power of semantic search and generative AI, our solution is robust, accurate, and truly user-centric. It successfully understands and adapts to the specific needs of each persona, delivering intelligent insights that matter.
