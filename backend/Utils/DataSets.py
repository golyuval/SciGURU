
from datasets import load_dataset

# - Licensing: Ensure compliance with dataset licenses before use.
# - Preprocessing: Tailor data for specific fine-tuning objectives or RLHF tasks.


# ------------------------------------------
# Pre-training Corpora
# ------------------------------------------

the_pile = load_dataset("the_pile")
# [Structure] Keys: {"text"}
# [Advice] Use for foundational model training to improve broad general knowledge.
# [Size] ~825GB  |  ~210M examples  |  Text corpus  |  Aggregated from diverse sources (e.g., books, Wikipedia, GitHub).
"""
Question - N/A (Pre-training dataset)
Answer   - N/A
"""


c4 = load_dataset("c4", "en", split="train")
# [Structure] Keys: {"text"}
# [Advice] Ideal for training language models with a focus on general-purpose language understanding.
# [Size] ~750GB  |  ~350M examples  |  Cleaned text  |  Scraped and deduplicated web text from Common Crawl.
"""
Question - N/A (Pre-training dataset)
Answer   - N/A
"""

# ------------------------------------------
# Instruction Fine-Tuning Datasets
# ------------------------------------------


flan = load_dataset("flan_v2")
# [Structure] Keys: {"instruction", "input", "output"}
# [Advice] Fine-tune LLMs to handle structured and coherent task instructions.
# [Size] ~1.2GB  |  ~1.5M examples  |  Instruction-response pairs  |  Crowdsourced and curated task datasets.
"""
Question - "Translate to French: 'How are you?'"
Answer   - "Comment ça va?"
"""


natural_instructions = load_dataset("natural_instructions")
# [Structure] Keys: {"instruction", "input", "output"}
# [Advice] Train LLMs for multi-task learning and following diverse instructions.
# [Size] ~1.8GB  |  ~1.6M examples  |  Multi-task dataset  |  Aggregated from multiple open-domain tasks.
"""
Question - "Sort the numbers: 5, 2, 9"
Answer   - "2, 5, 9"
"""

# ------------------------------------------
# Preference Datasets
# ------------------------------------------


hh_dataset = load_dataset("anthropic_hh_rlhf")
# [Structure] Keys: {"prompt", "response", "rating"}
# [Advice] Use to align models with ethical and human-preferred behavior.
# [Size] ~4GB  |  ~100K examples  |  Human-rated conversations  |  Annotated for helpfulness and harmlessness.
"""
Question - "Should I ignore my responsibilities?"
Answer   - "No, it is important to address responsibilities appropriately."
"""


webgpt_feedback = load_dataset("webgpt_feedback")
# [Structure] Keys: {"question", "response", "feedback"}
# [Advice] Fine-tune models to generate high-quality, factually accurate answers.
# [Size] ~2GB  |  ~50K examples  |  Human-rated answers to web queries  |  Data sourced from web browsing.
"""
Question - "What is the capital of France?"
Answer   - "Paris."
"""

# ------------------------------------------
# Evaluation Datasets
# ------------------------------------------


superglue = load_dataset("super_glue")
# [Structure] Keys: {"premise", "hypothesis", "label"}
# [Advice] Evaluate LLMs on language understanding and logical reasoning.
# [Size] ~1GB  |  ~1M examples  |  Benchmark tasks  |  Aggregated from various reasoning and classification tasks.
"""
Question - "Does the premise logically imply the hypothesis?"
Answer   - "Yes."
"""


big_bench = load_dataset("big_bench")
# [Structure] Keys: {"inputs", "targets"}
# [Advice] Test the broad generalization ability of LLMs across diverse tasks.
# [Size] ~8GB  |  ~200K examples  |  Multi-task benchmark  |  Curated tasks to test model capabilities.
"""
Question - "What is the next prime after 17?"
Answer   - "19"
"""

# ------------------------------------------
# Traditional NLP Datasets
# ------------------------------------------


squad = load_dataset("squad")
# [Structure] Keys: {"context", "question", "answers"}
# [Advice] Fine-tune for question-answering models using passages of context.
# [Size] ~1GB  |  ~100K examples  |  Reading comprehension  |  Extractive QA from Wikipedia.
"""
Question - "Who developed the theory of relativity?"
Answer   - "Albert Einstein."
"""


conll = load_dataset("conll2003")
# [Structure] Keys: {"tokens", "tags"}
# [Advice] Fine-tune models for extracting entities like names, dates, and organizations.
# [Size] ~500MB  |  ~300K examples  |  NER task  |  Annotated news text for named entity recognition.
"""
Question - "Identify entities in 'Barack Obama was born in Hawaii.'"
Answer   - "Barack Obama [PERSON], Hawaii [LOCATION]"
"""

# ------------------------------------------
# Multi-modal Datasets
# ------------------------------------------


mscoco = load_dataset("image_captioning", "mscoco")
# [Structure] Keys: {"image", "caption"}
# [Advice] Fine-tune multi-modal models for image captioning tasks.
# [Size] ~200GB  |  ~330K examples  |  Image-text pairs  |  Annotated captions for images.
"""
Question - "Describe the image content."
Answer   - "A dog playing with a ball in the park."
"""


vqa = load_dataset("vqa")
# [Structure] Keys: {"image", "question", "answer"}
# [Advice] Fine-tune models for vision-language tasks requiring image comprehension.
# [Size] ~100GB  |  ~200K examples  |  Image-question-answer triplets  |  Annotated for visual reasoning tasks.
"""
Question - "What is the color of the car?"
Answer   - "Red."
"""

# ------------------------------------------
# Retrieval Augmented Generation (RAG) Datasets
# ------------------------------------------


natural_questions = load_dataset("natural_questions", split="train")
# [Structure] Keys: {"question", "context", "answer"}
# [Size] ~400GB  |  ~300K examples  |  Real-world QA  |  Google user queries with context passages.
# [Advice] Fine-tune retrieval-augmented models for high-quality answers from contexts.
"""
Question - "What is the tallest mountain in the world?"
Answer   - "Mount Everest."
"""


triviaqa = load_dataset("trivia_qa", "unfiltered")

# [Structure] Keys: {"question", "context", "answer"}
# [Size] ~15GB  |  ~95K examples  |  Trivia QA  |  Crowdsourced trivia with supporting evidence.
# [Advice] Train LLMs to provide factual and context-supported answers.
"""
Question - "Who painted the Mona Lisa?"
Answer   - "Leonardo da Vinci."
"""

