from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing
from dataflow.core import LLMServingABC
from dataflow.operators.pandas_operator import PandasOperator
from dataflow.operators.generate import QuestionDifficultyClassifier
from operators.keypoint_driven import KeyPointDriven
from prompts.kpdds import KnowledgeExtractorPrompt
import numpy as np
import pandas as pd
import torch 

class MathFusionPipeline():
    def __init__(self, llm_serving: LLMServingABC = None):
        
        self.storage = FileStorage(
            first_entry_file_name="/mnt/public/data/wongzhenhao/MathQA/dataset/math_train.jsonl",
            cache_path="./kpdds/math",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # use API server as LLM serving
        llm_serving_gpt4o= APILLMServing_request(
                    api_url="http://123.129.219.111:3000/v1/chat/completions",
                    model_name="gpt-4o",
                    max_workers=500
        )

        llm_serving_o4mini = APILLMServing_request(
                    api_url="http://123.129.219.111:3000/v1/chat/completions",
                    model_name="gpt-4o",
                    max_workers=500
        )

        self.first10 = PandasOperator([lambda df: df.head(10)])         
        
        self.difficulty_classifier = QuestionDifficultyClassifier(llm_serving = llm_serving_gpt4o)

        self.knowledge_extract = KeyPointDriven(llm_serving=llm_serving_o4mini, prompt_template=KnowledgeExtractorPrompt())


    def forward(self):
        self.first10.run(
            storage = self.storage.step(),
        )

        self.difficulty_classifier.run(
            storage = self.storage.step(),
            input_key = "question",
            output_key = "difficulty"
        )

        self.knowledge_extract.run(
            storage= self.storage.step(),
            input_question_key= "question",
            input_solution_key= "solution",
            output_keypoints_key = "key_points",
        )

if __name__ == "__main__":
    pl = MathFusionPipeline()
    pl.forward()