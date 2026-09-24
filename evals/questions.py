"""
Evaluation question set for the cancer pipeline.

Each entry has:
    question:          what the user asks
    expected_keywords: substrings that MUST appear in a retrieved chunk
                       (case-insensitive). If none match → retrieval miss.
    expected_sources:  optional source substrings (e.g. a PMID)
    category:          used to group results in the report

Location: evals/questions.py
"""

EVAL_QUESTIONS = [
    {
        "question": "What is the standard treatment for triple-negative breast cancer?",
        "expected_keywords": ["triple-negative", "triple negative"],
        "category": "tnbc",
    },
    {
        "question": "How does pembrolizumab work in lung cancer?",
        "expected_keywords": ["pembrolizumab", "pd-1", "pd-l1"],
        "category": "immunotherapy",
    },
    {
        "question": "What is the mechanism of trastuzumab in HER2-positive breast cancer?",
        "expected_keywords": ["trastuzumab", "her2"],
        "category": "her2",
    },
    {
        "question": "What are common BRCA1 mutations associated with?",
        "expected_keywords": ["brca1", "brca"],
        "category": "genomics",
    },
    {
        "question": "What is the role of immune checkpoint inhibitors in solid tumors?",
        "expected_keywords": ["checkpoint", "pd-1", "pd-l1"],
        "category": "immunotherapy",
    },
    {
        "question": "What are targeted therapies for non-small cell lung cancer?",
        "expected_keywords": ["non-small cell", "nsclc", "egfr"],
        "category": "nsclc",
    },
    {
        "question": "How is colorectal cancer immunotherapy used in treatment?",
        "expected_keywords": ["colorectal", "immunotherapy"],
        "category": "colorectal",
    },
    {
        "question": "What is the standard adjuvant therapy for early breast cancer?",
        "expected_keywords": ["adjuvant", "breast"],
        "category": "breast",
    },
    {
        "question": "What biomarkers predict response to immunotherapy?",
        "expected_keywords": ["biomarker", "pd-l1", "msi"],
        "category": "biomarkers",
    },
    {
        "question": "What is the weather in Nairobi?",  # negative test
        "expected_keywords": [],
        "expected_route": "off_topic",
        "category": "off_topic_control",
    },
]

EVAL_QUESTIONS = EVAL_QUESTIONS[:3]