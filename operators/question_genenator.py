import pandas as pd
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import torch

@OPERATOR_REGISTRY.register()
class QuestionGenerator(OperatorABC):

    def __init__(self, 
                 num_prompts: int = 1,
                 llm_serving: LLMServingABC = None,
                 prompt_template = None
                 ):
        self.prompts = prompt_template
        self.num_prompts = num_prompts
        self.llm_serving = llm_serving

        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def _reformat_prompt(self, dataframe):
        problem_1 = dataframe[self.problem_1].tolist()
        problem_2 = dataframe[self.problem_2].tolist()
        system_prompt = self.prompts.system_prompt()
        prompts = [self.prompts.prompt(p1,p2) for p1,p2 in zip(problem_1, problem_2)]

        return system_prompt, prompts

    def run(self, storage: DataFlowStorage, promblem_1: str, promblem_2: str, output_key: str):
        """
        Run the question generation process.
        """
        self.problem_1, self.problem_2 = promblem_1, promblem_2
        dataframe = storage.read("dataframe")
        formatted_prompts = self._reformat_prompt(dataframe)
        responses = self.llm_serving.generate_from_input(formatted_prompts)

        for i in range(self.num_prompt):
            formatted_prompts = self._reformat_prompt(dataframe)
            responses = self.llm_serving.generate_from_input(formatted_prompts)

            dataframe[f"{output_key}_{i}"] = responses
            self.logger.info(f"Generated questions for {output_key}_{i}")

        output_file = storage.write(dataframe)
        self.logger.info(f"Generated questions saved to {output_file}")
