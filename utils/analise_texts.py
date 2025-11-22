from textblob import TextBlob

def analyze_sentiment(text: str) -> dict:
    try:
        analysis = TextBlob(text)

        polarity = analysis.sentiment.polarity  # (-1) негативна, (1) позитивна
        subject= analysis.sentiment.subjectivity  # 0 - обʼєктивно, 1 - субʼєктивно

        if polarity > 0.3:
            sentiment = "😊 Позитивна"
            emoji = "✅"
        elif polarity <= -0.3:
            sentiment = "😔 Негативна"
            emoji = "❌"
        else:
            sentiment = "😐 Нейтральна"
            emoji = "⚪"

        confidence = min(abs(polarity) * 100, 100)
        return {
            'sentiment': sentiment,
            'polarity': round(polarity, 3),
            'subjectivity': round(subject, 3),
            'confidence': round(confidence, 1),
            'emoji': emoji
        }

    except Exception as err:
        return {
            'error': str(err)
        }
