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
            "classifier": "facebook/bart-large-mnli",
            "summarizer": "sshleifer/distilbart-cnn-12-6"
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

    def classify_text(self):
        print("Classifying text...")
        classifier = pipeline("zero-shot-classification", model=self.model["classifier"], device=self.device)

        examples = [
            {
                "text": "У Києві презентували новий технологічний стартап",
                "labels": ["технології", "спорт", "політика", "бізнес"]
            },
            {
                "text": "Футбольний матч закінчився з рахунком 3-2",
                "labels": ["спорт", "технології", "бізнес", "культура"]
            },
            {
                "text": "Акції компанії зросли на 15% сьогодні",
                "labels": ["бізнес", "спорт", "технології", "економіка"]
            }
        ]

        for example in examples:
            result = classifier(example["text"][:200], example["labels"])
            print(f"Text: '{example['text'][:50]}...'")
            for label, score in zip(result['labels'], result['scores']):
                print(f"  {label}: {score:.0%}")
            print()

    def summarize_text(self):
        print("Summarizing text...")
        summarizer = pipeline("summarization", model=self.model["summarizer"], device=self.device, min_length=30,
                              max_length=50)
        text = """
            Artificial intelligence (AI) is intelligence demonstrated by machines. 
Leading AI textbooks define the field as the study of 
intelligent agents: any system that perceives its environment 
and takes actions to achieve goals. AI is applied in 
various fields, including medicine, finance, and education.
        """
        summary = summarizer(text[:1000])[0]['summary_text']
        print(f"Original Text: {text.strip()}\n")
        print(f"Summary: {summary}")

    def simple_ner(self):
        print("Performing Named Entity Recognition...")
        ner = pipeline("ner",
                       model="dslim/bert-base-NER",
                       device=self.device,
                       aggregation_strategy="simple"
                       )
        texts = [
            "Volodymyr Zelenskyy is the president of Ukraine.",
            "The capital of Ukraine is Kyiv.",
            "Lviv is known for its beautiful architecture."
            "Apple Inc. is looking to expand its operations in Eastern Europe."
        ]

        for text in texts:
            entities = ner(text)
            print(f"Text: '{text}'")
            for entity in entities:
                print(f"  {entity['entity_group']}: {entity['word']} (Score: {entity['score']:.2%})")
            print()

    def run_all(self):
        self.analyze_sentiment()
        self.classify_text()
        self.summarize_text()
        self.simple_ner()


def main():
    demo = UkrainianNLP()
    while True:
        print("\nChoose an option:")
        choice = input(
            "1. Analyze Sentiment\n2. Classify Text\n3. Summarize Text\n4. Named Entity Recognition\n5. Run All\n6. Exit\nEnter choice: ")
        if choice == '1':
            demo.analyze_sentiment()
        elif choice == '2':
            demo.classify_text()
        elif choice == '3':
            demo.summarize_text()
        elif choice == '4':
            demo.simple_ner()
        elif choice == '5':
            demo.run_all()
        elif choice == '6':
            print("Exiting...")
            break



if __name__ == "__main__":
    main()
