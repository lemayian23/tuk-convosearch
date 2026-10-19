"""
Evaluation question sets.

Each entry has:
    question:          what the user asks
    expected_keywords: substrings that MUST appear in a retrieved chunk
                       (case-insensitive). If none match → retrieval miss.
    expected_sources:  optional source substrings (a filename, a PMID)
    expected_route:    defaults to internal_docs; use off_topic/small_talk
                       for the controls that must never reach the model
    category:          used to group results in the report

The TU-K set is grounded in the documents actually in corpus/tuk/. Several
entries are deliberately unanswerable, because a student assistant that
invents exam dates or fee structures is worse than one that says it does not
know — those are the `refusal` category and they are scored on NOT retrieving.

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

TUK_EVAL_QUESTIONS = [
    {
        "question": "When does Semester II begin at TUK in 2026?",
        "expected_keywords": ["january 5", "semester ii begins"],
        "expected_sources": ["Academic Year Calendar"],
        "category": "calendar",
    },
    {
        "question": "What are the dates of the second Continuous Assessment Tests?",
        "expected_keywords": ["march 23", "second continuous assessment"],
        "category": "calendar",
    },
    {
        "question": "Is there a revision week before the Semester II exams?",
        "expected_keywords": ["revision week", "april"],
        "category": "calendar",
    },
    {
        "question": "When do the end of semester two examinations for 2025/2026 run?",
        # The reschedule notice overrides the calendar; the answer must come
        # from the notice, not from the April dates.
        "expected_keywords": ["25th may", "9th june", "rescheduled"],
        "expected_sources": ["NOTICE"],
        "category": "conflicting_dates",
    },
    {
        "question": "Why were the end-of-semester exams moved?",
        "expected_keywords": ["matatu strike", "senate"],
        "category": "conflicting_dates",
    },
    {
        "question": "What time is the Health Education paper on Monday 18 May 2026?",
        "expected_keywords": ["8:00", "health education"],
        "category": "timetable",
    },
    {
        "question": "Which room is the Cryptography and Network Security exam in?",
        "expected_keywords": ["cryptography", "uccc"],
        "category": "timetable",
    },
    {
        "question": "Can a diploma holder join a TU-K degree programme directly?",
        "expected_keywords": ["upgrade", "year 2", "diploma"],
        "category": "admissions",
    },
    {
        "question": "When do classes start for the February 2026 upgrade intake?",
        "expected_keywords": ["23rd february", "february 2026"],
        "category": "admissions",
    },
    {
        "question": "Where is the main TU-K campus located?",
        "expected_keywords": ["haile selassie", "nairobi"],
        "category": "location",
    },
    {
        "question": "What is the postal address of the university?",
        "expected_keywords": ["52428"],
        "category": "location",
    },
    {
        "question": "How much are tuition fees for 2026?",
        # Nothing in the corpus states fees, so a correct run retrieves weakly
        # and the answer must refuse. Scoring this as a miss is the point.
        "expected_keywords": [],
        "expect_no_answer": True,
        "category": "refusal",
    },
    {
        "question": "How do I register for units on the student portal?",
        "expected_keywords": [],
        "expect_no_answer": True,
        "category": "refusal",
    },
    {
        "question": "What are the attachment logbook requirements?",
        # 05-Student-Attachment-Guide.pdf and 11-Attachment-Log-Book.pdf are
        # scanned images: they extract to zero characters until someone OCRs
        # them. This question stays red until then, on purpose.
        "expected_keywords": [],
        "expect_no_answer": True,
        "category": "needs_ocr",
    },
    {
        "question": "What is the student GPS balance top-up number?",
        "expected_keywords": [],
        "expect_no_answer": True,
        "category": "refusal",
    },
    {
        "question": "Hi, are you there?",
        "expected_route": "small_talk",
        "category": "router_control",
    },
    {
        "question": "Who won the Premier League title?",
        "expected_route": "off_topic",
        "category": "router_control",
    },
]
