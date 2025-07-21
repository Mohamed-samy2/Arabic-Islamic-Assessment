# Arabic-Islamic-Assessment- 
# ────────────────────────────────────────────────────────────────
# Preprocsessing for Datata Pipline 
Function                                   | Responsibility
───────────────────────────────────────────|──────────────────────────────────────────────────────────────────────────────
__init__(self, chunk_size, chunk_overlap)  | Initialise an ArabicTextPreprocessor object: sets default chunk size /
                                           | overlap and compiles all Regex patterns (diacritics, kashida, phones,
                                           | emails, URLs, etc.).

extract_text_from_docx(self, path, upl)    | Extract raw text from a DOCX file (uploaded via `window.fs` or local),
                                           | using **mammoth** or **python‑docx**; returns text + metadata.

_extract_from_local_file(self, path)       | Lightweight variant used when the file is local; performs the same
                                           | extraction steps without `window.fs`.

remove_phone_numbers(self, text)           | Remove phone numbers from the text via specialised Regex.

remove_emails(self, text)                  | Remove e‑mail addresses from the text.

remove_flixat(self, text)                  | Remove URLs / “flixat” links from the text.

remove_diacritics(self, text)              | Strip Arabic diacritics (fatha, damma, kasra, etc.).

remove_tatweel(self, text)                 | Remove the kashida (ــ) elongation character.

normalize_punctuation(self, text)          | Convert Arabic punctuation to its English equivalents (“،” → “,”,
                                           | “؟” → “?” …).

remove_non_arabic(self, text)              | Delete any non‑Arabic characters (keeps digits & basic punctuation).

normalize_whitespace(self, text)           | Collapse multiple whitespace into a single space and trim edges.

clean_text(self, text, **flags)            | A full cleaning pipeline; calls any combination of the above removal /
                                           | normalisation helpers according to the flags provided
                                           | (diacritics, tatweel, phone, e‑mail, flixat, ...).

split_into_sentences(self, text)           | Split text into sentences using Arabic sentence terminators (., !, ؟).

chunk_text(self, text, method)             | Dispatch to a specific chunking method (word, sentence, paragraph).

_word_based_chunking(self, text)           | Break the text into fixed‑length word chunks with overlap.

_sentence_based_chunking(self, text)       | Build chunks composed of whole sentences, each ≤ `chunk_size` words.

_paragraph_based_chunking(self, text)      | Build chunks at the paragraph level while respecting the word cap.

generate_chunk_id(self, text, index)       | Create a unique ID for every chunk (short MD5 hash + sequential index).

extract_keywords(self, text, top_k)        | Pull the most frequent keywords after cleaning & stop‑word removal.

process_document(self, path, method, opts) | Full processing pipeline: 1) extract text 2) clean 3) chunk
                                           | 4) wrap each chunk in a `TextChunk` with rich metadata.

save_processed_chunks(self, chunks, path)  | Persist a list of chunks as a JSON file.

load_processed_chunks(self, path)          | Rehydrate the same JSON back into a list of `TextChunk` objects.

if __name__ == "__main__":                 | Demo block: instantiates the preprocessor, passes a DOCX file, enables
                                           | all cleaning switches (including phone / e‑mail / URL removal),
                                           | prints statistics, and saves output to `processed_chunks.json`.
