import re

def to_snake_case(sentence: str) -> str:
    sentence = sentence.lower()
    sentence = re.sub(r'[\W_]+', '_', sentence)
    sentence = sentence.strip('_')
    
    return sentence
