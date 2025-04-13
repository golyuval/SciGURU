import os
import sys
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Jargon.jargon_util import analyze_text, BASE_PATH

# Test paragraphs in ascending order of complexity
test_paragraphs = [
    # Level 1: Simple, everyday language
    """
    The sun is a big, bright star in the sky. It gives us light and warmth every day. 
    Plants use sunlight to grow, and we need plants for food. The sun rises in the morning 
    and sets in the evening, making day and night.
    """,
    
    # Level 2: Basic scientific concepts
    """
    Photosynthesis is how plants make food. They use sunlight, water, and air to create 
    energy. This process happens in their leaves. Without photosynthesis, plants couldn't 
    grow, and animals wouldn't have food to eat.
    """,
    
    # Level 3: Intermediate scientific explanation
    """
    The process of photosynthesis involves the conversion of light energy into chemical 
    energy. Plants absorb sunlight through chlorophyll in their leaves, which triggers a 
    series of biochemical reactions. These reactions transform carbon dioxide and water 
    into glucose and oxygen, providing energy for plant growth and releasing oxygen into 
    the atmosphere.
    """,
    
    # Level 4: Advanced scientific explanation
    """
    Photosynthesis is a complex biochemical process occurring in the chloroplasts of 
    plant cells, specifically within the thylakoid membranes. The light-dependent 
    reactions initiate the process through photophosphorylation, generating ATP and 
    NADPH. These energy carriers then fuel the Calvin cycle, where carbon fixation 
    occurs through the enzyme RuBisCO, ultimately producing organic compounds essential 
    for cellular metabolism.
    """,
    
    # Level 5: Highly technical scientific explanation
    """
    The photosynthetic apparatus operates through a sophisticated electron transport 
    chain, where photosystem II and photosystem I work in tandem to facilitate 
    non-cyclic photophosphorylation. The Z-scheme of electron transport generates a 
    proton motive force across the thylakoid membrane, driving ATP synthase to 
    catalyze the phosphorylation of ADP. Concurrently, the reduction of NADP+ to 
    NADPH occurs through ferredoxin-NADP+ reductase, providing reducing power for 
    the subsequent carbon fixation reactions in the Calvin-Benson cycle.
    """
]

def run_jargon_tests():
    print("Testing Jargon Analyzer with paragraphs of increasing complexity\n")
    print("=" * 80)
    
    # Load required data using correct paths
    names = pd.read_csv(os.path.join(BASE_PATH, "names.csv"), header=None)[0].tolist()
    words = pd.read_csv(os.path.join(BASE_PATH, "DataUKUS2018-2021.csv"), header=None).set_index(0)[1].to_dict()
    
    for i, paragraph in enumerate(test_paragraphs, 1):
        print(f"\nLevel {i} - Complexity Test:")
        print("-" * 40)
        score = analyze_text(paragraph, words, names, verbose=True)
        print(f"Jargon Score: {score:.2f}")
        print("=" * 80)

if __name__ == "__main__":
    run_jargon_tests() 