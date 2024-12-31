#!/usr/bin/env python3

import re
import sys
import argparse
from typing import List, Dict, Any
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

######################################################################
# 1. CONFIGURATION
######################################################################

# Hardcoded configuration option to enable or disable explanations
ENABLE_EXPLANATION = False  # Set to False to disable explanation generation

# Example local models from Hugging Face (download once; no API calls):
CLASSIFICATION_MODEL_NAME = "facebook/bart-large-mnli"   # Zero-shot classification
EXPLANATION_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"       # Local generative model for explanations

# Base list of music-specific topics for direct/keyword detection
MUSIC_TOPICS_KEYWORDS = [
    "mixing", "mastering", "instrumentation", "vocals", "melody",
    "harmony", "drums", "strings", "synths", "sound design",
    "tempo", "key", "chords",
    "transitions", "bridge", "chorus", "verse", "hook", "bass", "guitar",
    "piano", "keyboard", "sampling", "looping", "beat", "timing", "groove",
    "dynamics", "sub-bass", "soundscape", "layering", "instrumental", "texture", "muddy",
    "sync", "fx", "post-rock", "rolls", "rhythms", "pause"
]

# Synonyms/variations for certain topics (helps direct matching & expansions)
TOPIC_SYNONYMS = {
    "sub-bass": ["sub bass", "subbass"],
    "bass": ["bassline", "bass line"],
    "vocals": ["vocal", "voice", "singer"],
    "drums": ["drummer", "drumkit", "snare", "hat", "kick", "percussion"],
    "evolution": ["evolution", "pacing", "arrangement", "structure", "develop"],
    "fx": ["compressor", "limiter", "eq", "effect", "reverb"]
    # Add more synonyms as needed
}

# Classification labels
POSSIBLE_LABELS = ["constructive", "praise", "non-constructive", "neutral"]

# Scoring parameters
TIMESTAMP_WEIGHT = 2           # How much each timestamp is worth
MAX_TIMESTAMP_BONUS = 5        # Maximum timestamps that contribute to score
MIN_CONSTRUCTIVE_TOPICS = 2    # Min number of music topics for fully constructive feedback
BASE_CONSTRUCTIVE_SCORE = 0    # Base score if classified as constructive
BASE_PRAISE_SCORE = 0          # Base score if classified as praise
BASE_NEUTRAL_SCORE = 0         # Base score if classified as neutral
BASE_NONCONSTRUCTIVE_SCORE = 0 # Base score if classified as non-constructive

######################################################################
# 2. LOAD MODELS LOCALLY
######################################################################

# Zero-Shot Classification Pipeline
class_tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL_NAME)
class_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFICATION_MODEL_NAME)
classifier_pipeline = pipeline(
    task="zero-shot-classification",
    model=class_model,
    tokenizer=class_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# KeyBERT for direct keyword extraction (optional usage)
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# Sentence-BERT for advanced topic similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize explanation model conditionally
explanation_tokenizer = None
explanation_model = None

if ENABLE_EXPLANATION:
    print("Loading explanation model...")
    explanation_tokenizer = AutoTokenizer.from_pretrained(EXPLANATION_MODEL_NAME)
    explanation_model = AutoModelForCausalLM.from_pretrained(EXPLANATION_MODEL_NAME)
    explanation_model.to("cuda" if torch.cuda.is_available() else "cpu")
    explanation_model.eval()

######################################################################
# 3. HELPER FUNCTIONS
######################################################################

def extract_timestamps(text: str) -> List[str]:
    """
    Extract timestamps from the text, including formats like:
      - 1:05
      - 12:34
      - 45s
      - 1m
      - 1min 43s
      - 1m43s
    Then normalize them to a consistent "MM:SS" style if possible.
    """
    # Regex patterns for different timestamp styles
    pattern_colon = r"\b(\d{1,2}:\d{1,2}(?::\d{1,2})?)\b"  # e.g. 1:05 or 01:02:03
    pattern_seconds = r"\b(\d{1,3})\s?(?:sec|secs|s)\b"    # e.g. 43s or 43 secs
    pattern_minutes = r"\b(\d{1,3})\s?(?:min|mins|m)\b"   # e.g. 1m or 1 min
    pattern_minsec = r"\b(\d{1,3})\s?(?:min|m)\s?(\d{1,3})\s?(?:sec|secs|s)\b"  # e.g. 1 min 43 s, 1m43s

    found_timestamps = set()

    # Direct colon format
    for match in re.findall(pattern_colon, text, flags=re.IGNORECASE):
        found_timestamps.add(match)

    # x min + y sec format
    for mm, ss in re.findall(pattern_minsec, text, flags=re.IGNORECASE):
        # Convert to ints to normalize
        m_int = int(mm)
        s_int = int(ss)
        # Normalize
        norm = f"{m_int}:{s_int:02d}"
        found_timestamps.add(norm)

    # single format x min
    for mm in re.findall(pattern_minutes, text, flags=re.IGNORECASE):
        m_int = int(mm)
        norm = f"{m_int}:00"
        found_timestamps.add(norm)

    # single format x sec
    for ss in re.findall(pattern_seconds, text, flags=re.IGNORECASE):
        s_int = int(ss)
        # If we have only seconds, assume 0:SS
        norm = f"0:{s_int:02d}"
        found_timestamps.add(norm)

    # Try to standardize colon format e.g. 1:2 -> 1:02
    normalized = set()
    for t in found_timestamps:
        segments = t.split(":")
        if len(segments) == 2:
            mm, ss = segments
            mm_int = int(mm)
            ss_int = int(ss)
            norm = f"{mm_int}:{ss_int:02d}"
            normalized.add(norm)
        elif len(segments) == 3:
            # HH:MM:SS or M:S:SS ...
            hh, mm, ss = segments
            hh_int = int(hh)
            mm_int = int(mm)
            ss_int = int(ss)
            # Just store as H:MM:SS or something. Optionally do total minutes:seconds
            # For simplicity, we keep it as is or transform to mm:ss if HH is small
            total_m = hh_int * 60 + mm_int
            norm = f"{total_m}:{ss_int:02d}"
            normalized.add(norm)
        else:
            # already in some format
            normalized.add(t)

    return list(normalized)


def mask_timestamps(text: str) -> str:
    """Replace recognized timestamps with '[TIMESTAMP]' to avoid spurious topic extraction."""
    # We'll combine all patterns used above in a single re.sub
    pattern = re.compile(
        r"\b(\d{1,2}:\d{1,2}(?::\d{1,2})?|"
        r"\d{1,3}\s?(?:sec|secs|s)|"
        r"\d{1,3}\s?(?:min|mins|m)|"
        r"\d{1,3}\s?(?:min|m)\s?\d{1,3}\s?(?:sec|secs|s))\b",
        flags=re.IGNORECASE
    )
    return pattern.sub("[TIMESTAMP]", text)


def zero_shot_classify(feedback: str) -> List[str]:
    """
    Multi-label classification for 'constructive', 'praise', 'non-constructive', 'neutral'.
    Returns a list of labels that exceed a confidence threshold.
    """
    result = classifier_pipeline(
        sequences=feedback,
        candidate_labels=POSSIBLE_LABELS,
        multi_label=True
    )

    labels = []
    threshold = 0.3  # Tweak based on empirical results
    for lbl, score in zip(result['labels'], result['scores']):
        if score >= threshold:
            labels.append(lbl)
    return labels


def clamp(value, low, high):
    return max(low, min(value, high))


def direct_topic_match(lower_feedback: str) -> List[str]:
    """
    Check direct/keyword presence (including synonyms).
    This captures obvious references like 'drums', 'kick drum' etc.
    """
    matched = set()
    # Expand synonyms
    expanded_dict = {}
    for main_topic, syns in TOPIC_SYNONYMS.items():
        expanded_dict[main_topic] = [main_topic] + syns
    # For everything not in synonyms, just add itself
    for topic in MUSIC_TOPICS_KEYWORDS:
        if topic not in expanded_dict:
            expanded_dict[topic] = [topic]

    for main_topic, variations in expanded_dict.items():
        for variant in variations:
            if variant.lower() in lower_feedback:
                matched.add(main_topic)
                break
    return list(matched)


def semantic_topic_match(feedback: str) -> List[str]:
    """
    Identify music-specific topics using Sentence-BERT for semantic similarity.
    """
    # Embed the feedback and music topics
    feedback_embedding = sentence_model.encode(feedback, convert_to_tensor=True)
    topic_embeddings = sentence_model.encode(MUSIC_TOPICS_KEYWORDS, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(feedback_embedding, topic_embeddings)[0]

    # threshold
    SIMILARITY_THRESHOLD = 0.45  # a bit stricter than 0.3

    selected = []
    for i, score in enumerate(cosine_scores):
        if score.item() >= SIMILARITY_THRESHOLD:
            selected.append(MUSIC_TOPICS_KEYWORDS[i])
    return list(set(selected))


def identify_music_topics(feedback: str) -> List[str]:
    """
    Combines direct keyword matching + semantic similarity matching
    for more robust topic detection.
    """
    lower_feedback = feedback.lower()
    direct_matches = direct_topic_match(lower_feedback)
    semantic_matches = semantic_topic_match(feedback)
    combined = set(direct_matches).union(set(semantic_matches))
    return list(combined)


def generate_explanation(
    feedback: str,
    classifications: List[str],
    score: int,
    topics: List[str],
    improvement: str
) -> str:
    """
    Use local GPT-Neo to generate a short explanation in plain text.
    Focus on why the score was given and how to improve the feedback.
    """
    if not ENABLE_EXPLANATION:
        return ""

    classifications_formatted = ", ".join([label.capitalize() for label in classifications])
    prompt = (
        f"Feedback: {feedback}\n\n"
        f"Classifications: {classifications_formatted}\n"
        f"Score: {score}\n"
        f"Music Topics: {topics}\n"
        f"Improvement Suggestions: {improvement}\n\n"
        "Provide a concise explanation of why this feedback received its classifications and score, "
        "and how the user can improve their feedback to be more constructive and specific.\n\n"
        "Explanation:"
    )

    input_ids = explanation_tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(explanation_model.device)

    with torch.no_grad():
        output_ids = explanation_model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 150,
            temperature=0.7,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    generated_text = explanation_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "Explanation:" in generated_text:
        explanation_part = generated_text.split("Explanation:")[-1].strip()
    else:
        explanation_part = generated_text.strip()
    return explanation_part


def parse_feedback(feedback: str) -> Dict[str, Any]:
    """
    1. Extract timestamps + mask.
    2. Classify (multi-label).
    3. Identify music topics (keyword + semantic).
    4. Score: base + #topics (if constructive) + timestamps bonus
    5. Improvement advice
    6. Explanation (if enabled)
    """
    # 1) Timestamps
    timestamps = extract_timestamps(feedback)
    masked_feedback = mask_timestamps(feedback)

    # 2) Classification
    labels = zero_shot_classify(masked_feedback)

    # 3) Music topics
    topics = identify_music_topics(masked_feedback)

    # 4) Scoring
    raw_score = 0
    # Add scores per label
    for lbl in labels:
        if lbl == "constructive":
            raw_score += BASE_CONSTRUCTIVE_SCORE
        elif lbl == "praise":
            raw_score += BASE_PRAISE_SCORE
        elif lbl == "neutral":
            raw_score += BASE_NEUTRAL_SCORE
        elif lbl == "non-constructive":
            raw_score += BASE_NONCONSTRUCTIVE_SCORE

    # Add #topics if constructive
    if "constructive" in labels:
        raw_score += len(topics)

    # Timestamps weighting
    t_count = len(timestamps)
    t_bonus = TIMESTAMP_WEIGHT * clamp(t_count, 0, MAX_TIMESTAMP_BONUS)
    raw_score += t_bonus

    # 5) Improvement advice
    improvement_suggestions = []
    if "constructive" not in labels:
        improvement_suggestions.append("Provide more specific, actionable suggestions related to music production.")
    if t_count == 0:
        improvement_suggestions.append("Include timestamps to reference specific sections of the track.")
    if "constructive" in labels and len(topics) < MIN_CONSTRUCTIVE_TOPICS:
        improvement_suggestions.append("Mention more music-production elements (e.g., mixing, arrangement) to enhance feedback.")
    improvement_msg = "; ".join(improvement_suggestions) if improvement_suggestions else "Your feedback seems thorough."

    # 6) Explanation
    explanation_text = ""
    if ENABLE_EXPLANATION:
        explanation_text = generate_explanation(
            feedback=feedback,
            classifications=labels,
            score=int(raw_score),
            topics=topics,
            improvement=improvement_msg
        )

    return {
        "classifications": labels,
        "score": int(raw_score),
        "timestamps": timestamps,
        "topics": topics,
        "explanation": explanation_text,
        "improvement": improvement_msg
    }

######################################################################
# 4. MAIN
######################################################################

def main():
    parser = argparse.ArgumentParser(description="Local LLM-based Music Feedback Parser with improved topic & timestamp detection.")
    parser.add_argument("feedback", type=str, help="Feedback text in quotes.")
    args = parser.parse_args()

    feedback_text = args.feedback.strip()
    if not feedback_text:
        print("Empty feedback provided.")
        sys.exit(1)

    result = parse_feedback(feedback_text)

    # Output
    cls_fmt = ", ".join([label.upper() for label in result['classifications']])
    # print(f"Classification: {cls_fmt}")
    print(f"\nScore: {result['score']}")
    # if result["topics"]:
    #     print(f"Music Topics: {result['topics']}")
    print(f"Score given because it is: {cls_fmt}, containing {len(result['topics'])} areas of feedback: {result['topics']}")
    if not result["timestamps"]:
        # print(f"Timestamps Detected: {result['timestamps']}")
        print("Improve your feedback by including things like timestamps for the places you think can be improved.")
    if ENABLE_EXPLANATION and result["explanation"]:
        print("Explanation & Improvement:")
        print(result["explanation"])

    # Clean up if explanation was used + GPU
    if ENABLE_EXPLANATION and torch.cuda.is_available():
        del explanation_model
        del explanation_tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
