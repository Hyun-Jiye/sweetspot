from src import preprocessor

from langchain.callbacks import get_openai_callback
from langchain_openai import OpenAI


def main():
    p = preprocessor.Preprocessor(
        model="ft:gpt-3.5-turbo-1106:personal::8mOmCe4Z",
        data_path="./twitter_all.csv",
        num_samples=0,
    )

    data = p.messages
    test_data = data[p.num_samples + 1000 : p.num_samples + 1010]
    test_data = [{"input": d["tweet_message"]} for d in test_data]
    with get_openai_callback() as cb:
        p.batch(test_data, verbose=True, batch_size=2)
        print(cb)


if __name__ == "__main__":
    main()
