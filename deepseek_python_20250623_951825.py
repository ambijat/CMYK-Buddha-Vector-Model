import numpy as np
import spacy

class EmotionEngine:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.emotion_map = {
            "Birth": ["need", "trust", "emergence"],
            "OldAge": ["memory", "loss", "wisdom"],
            "Sickness": ["pain", "confusion", "fever"],
            "Death": ["fear", "acceptance", "void"]
        }
    
    def analyze_text(self, text):
        doc = self.nlp(text)
        vector = np.zeros(4)
        
        for i, (sorrow, keywords) in enumerate(self.emotion_map.items()):
            for token in doc:
                for kw in keywords:
                    similarity = token.similarity(self.nlp(kw))
                    vector[i] += max(0, similarity)
        
        return {
            "Cyan": vector[0],
            "Magenta": vector[1],
            "Yellow": vector[2],
            "Black": vector[3]
        }