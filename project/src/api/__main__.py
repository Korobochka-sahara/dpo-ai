import argparse
import requests


API_URL = "http://127.0.0.1:8000/predict"


def analyze_text(text: str):

    response = requests.post(
        API_URL,
        json={"text": text},
        timeout=10
    )

    response.raise_for_status()

    result = response.json()

    print("\nPrediction:")
    print(result["prediction"])

    print("\nConfidence:")
    print(round(result["confidence"], 4))


def main():

    parser = argparse.ArgumentParser(
        description="Sentiment analysis CLI"
    )

    parser.add_argument(
        "text",
        type=str,
        help="Text for sentiment analysis"
    )

    args = parser.parse_args()

    analyze_text(args.text)


if __name__ == "__main__":
    main()