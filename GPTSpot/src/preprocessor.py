import dotenv
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)


class Preprocessor:
    def __init__(self, model="gpt-4", data_path=None, num_samples=25):
        """
        예시 데이터를 로드하고 랭체인 파이프라인을 초기화합니다.


        Args:
            model (str, optional): 랭체인에서 사용할 openAI 모델입니다. Defaults to "gpt-4".
            data_path (str, optional): csv 파일의 경로입니다. Defaults to "/twitter_all.csv".
            num_samples (int, optional): 예시 프롬프트로 제시할 샘플의 개수입니다.. Defaults to 25.
        """
        dotenv.load_dotenv()
        examples = []
        if data_path is None:
            self.df = None
        else:
            df = pd.read_csv(data_path, sep=",", encoding="utf-8")
            self.df = df
            messages = []

            for i in range(len(df)):
                if (df["tweet"][i] is float("nan")) or (df["brand"][i] is float("nan")):
                    continue
                prompt = df["tweet"][i]
                response = df["brand"][i]
                m = {"tweet_message": prompt, "brand_name": response}
                messages.append(m)
            self.messages = messages

            self.num_samples = num_samples

            for sample in messages[:num_samples]:
                examples.append(
                    {
                        "input": sample["tweet_message"],
                        "output": sample["brand_name"],
                    }
                )

        # This is a prompt template used to format each individual example.
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt, examples=examples
        )

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You extracts brand name from the korean tweeter message. Follow the examples.",
                ),
                few_shot_prompt,
                ("human", "{input}"),
            ]
        )

        functions = [
            {
                "name": "extract_brand",
                "description": "Extract brand names from korean tweeter message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brand_name": {
                            "type": "string",
                            "description": "The brand name in the message",
                        },
                        "brand_name_with_region": {
                            "type": "string",
                            "description": "The full brand name with regional information.",
                        },
                    },
                    "required": ["brand_name"],
                },
            }
        ]
        model = ChatOpenAI(model=model)
        self.chain = (
            final_prompt
            | model.bind(function_call={"name": "extract_brand"}, functions=functions)
            | JsonOutputFunctionsParser()
        )

    def batch(self, messages, verbose=False, batch_size=10):
        """
        랭체인 파이프라인을 통해서, 인풋 메시지들을 배치 처리합니다.
        메시지의 타입은
        [{
            "input": "인풋 메시지",
        }]
        의 list of dict 형식으로 주어야 합니다. 인풋 메시지는 트윗 raw message를 넣으면 됩니다.
        output 형식은
        [{
            "brand_name": "브랜드 이름",
            "brand_name_with_region": "브랜드 이름 + 지역 정보",
        }]
        의 list of dict 형식으로 나옵니다.


        Args:
            messages (list of dict): 인풋 메시지들의 list of dict 형식입니다. 인풋 메시지는 트윗 raw message를 넣으면 됩니다.
            verbose (bool, optional): True일 경우 Input message와 output message를 동시에 출력합니다. False일 시 아무 것도 출력하지 않습니다. Defaults to False.
            batch_size (int, optional): openAI API를 통해 배치 처리할 배치 메시지의 수입니다.
            너무 크면 rate limit에 걸릴 수도 있어서 적당히 조절해주면 됩니다. 크다고 돈이 더 많이 드는 것은 아닙니다. Defaults to 10.

        Returns:
            _type_: _description_
        """
        results = self.chain.batch(messages, batch_size=batch_size)
        if verbose:
            for i in range(len(results)):
                print("Input: ", messages[i])
                print("Output: ", results[i])
                print("=====================================")
        return results

    def run(self, message, verbose=False):
        result = self.chain.invoke(message)
        if verbose:
            print("Input: ", message)
            print("Output: ", result)
        return result
