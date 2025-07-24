import pandas as pd
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import torch

@OPERATOR_REGISTRY.register()
class GenerateEmbedding(OperatorABC):

    def __init__(self, embedding_serving: LLMServingABC = None):
        self.embedding_serving = embedding_serving

        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def run(self,
            storage: DataFlowStorage,
            question_key: str = "origin_question",
            embedding_key: str = "embeddings",
            ):
        dataframe = storage.read("dataframe")
        self.question_key = question_key
        self.embedding_key = embedding_key

        questions = dataframe[self.question_key].tolist()
        embeddings_list = self.embedding_serving.generate_embedding_from_input(questions)
        # embeddings = torch.tensor(embeddings_list)

        dataframe[self.embedding_key] = embeddings_list

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [self.embedding_key]