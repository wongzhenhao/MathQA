from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing
from dataflow.core import LLMServingABC
from dataflow.operators.pandas_operator import PandasOperator
from operators.generate_embedding import GenerateEmbedding
from operators.question_genenator import QuestionGenerator
from operators.question_evaluator import QuestionEvaluator
from prompts.mathfusion import SequentialFusionPrompt, ConditionFusionPrompt, ParallelFusionPrompt, QuestionEvaluationPrompt
import numpy as np
import pandas as pd
import torch 

class MathFusionPipeline():
    def __init__(self, llm_serving: LLMServingABC = None):
        
        self.storage = FileStorage(
            first_entry_file_name="/mnt/public/data/wongzhenhao/MathQA/dataset/gsm8k_train.jsonl",
            cache_path="./mathfusion",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # use API server as LLM serving
        llm_serving = APILLMServing_request(
                    api_url="http://123.129.219.111:3000/v1/chat/completions",
                    model_name="o4-mini",
                    max_workers=500
        )

        embedding_serving = APILLMServing_request(
                    api_url="http://123.129.219.111:3000/v1/embeddings",
                    model_name="text-embedding-3-large",
                    max_workers=500
        )

        # self.first10 = PandasOperator([lambda df: df.head(10)])             
        
        self.generate_embedding = GenerateEmbedding(embedding_serving=embedding_serving)

        def find_most_similar_questions(df):
            df = df.dropna(subset=['embeddings']).reset_index(drop=True)
            embeddings = torch.tensor(np.stack(df['embeddings'].values), dtype=torch.float32).cuda()  # shape: (n, d)
            sim_matrix = torch.matmul(embeddings, embeddings.T)  # shape: (n, n)
            sim_matrix.fill_diagonal_(-float('inf'))
            most_similar_idx = torch.argmax(sim_matrix, dim=1).cpu().numpy()
            df['most_similar_problem'] = df['question'].iloc[most_similar_idx].values

            return df
        
        self.matching_most_similar = PandasOperator([
                        find_most_similar_questions
                    ])
        
        self.sequential_fusion = QuestionGenerator(num_prompts=2, llm_serving=llm_serving, prompt_template=SequentialFusionPrompt())

        self.parallel_fusion = QuestionGenerator(num_prompts=2, llm_serving=llm_serving, prompt_template=ParallelFusionPrompt())

        self.condition_fusion = QuestionGenerator(num_prompts=2, llm_serving=llm_serving, prompt_template=ConditionFusionPrompt())

        def combined(df: pd.DataFrame) -> pd.DataFrame:
            question_columns = [
                "question",
                "condition_fusion_question_0",
                "condition_fusion_question_1",
                "parallel_fusion_question_0",
                "parallel_fusion_question_1",
                "sequential_fusion_question_0",
                "sequential_fusion_question_1"
            ]
            long_df = df[question_columns].melt(value_name="questions")[["questions"]]
            return long_df

        
        self.combined_question = PandasOperator([combined])

        self.question_evaluation = QuestionEvaluator(llm_serving=llm_serving, prompt_template=QuestionEvaluationPrompt())

    def forward(self):
        # self.first10.run(
        #     storage = self.storage.step(),
        # )

        self.generate_embedding.run(
            storage = self.storage.step(),
            question_key= "question",
            embedding_key = "embeddings",
        )

        self.matching_most_similar.run(
            storage= self.storage.step(),
        )

        self.sequential_fusion.run(
            storage= self.storage.step(),
            promblem_1= "question",
            promblem_2= "most_similar_problem",
            output_key="sequential_fusion_question",
        )

        self.parallel_fusion.run(
            storage= self.storage.step(),
            promblem_1= "question",
            promblem_2= "most_similar_problem",
            output_key="parallel_fusion_question"
        )

        self.condition_fusion.run(
            storage= self.storage.step(),
            promblem_1= "question",
            promblem_2= "most_similar_problem",
            output_key="condition_fusion_question"
        )

        self.combined_question.run(
            storage= self.storage.step()
        )

        self.question_evaluation.run(
            storage= self.storage.step(),
            problem = "questions",
            output_key= "refined_questions"
        )



if __name__ == "__main__":
    pl = MathFusionPipeline()
    pl.forward()