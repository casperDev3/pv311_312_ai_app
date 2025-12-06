import torch
from textblob.en import sentiment
from transformers import pipeline
import warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


class UkrainianNLP:
    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = {
            "sentiment": "cointegrated/rubert-tiny2-cedr-emotion-detection",
            "classifier": "facebook/bart-large-mnli"
        }

    def analyze_sentiment(self):
        print("Analyzing sentiment...")
        sentiments = pipeline("text-classification", model=self.model["sentiment"], device=self.device)
        texts = [
            "Цей продукт просто чудовий!",
            "Обслуговування було жахливим.",
            "Все добре, працює нормально.",
            "Я дуже розчарований якістю."
        ]
        emotion_map = {
            'neutral': '😐', 'anger': '😠', 'fear': '😨',
            'joy': '😊', 'sadness': '😢', 'surprise': '😮'
        }

        for text in texts:
            result = sentiments(text[:200])[0]
            emoji = emotion_map.get(result['label'], '❓')
            print(f"{emoji} '{text[:50]}...' → {result['score']:.0%}")


def main():
    demo = UkrainianNLP()
    demo.analyze_sentiment()


if __name__ == "__main__":
    main()
