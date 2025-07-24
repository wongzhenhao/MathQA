import pandas as pd
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import torch

@OPERATOR_REGISTRY.register()
class QuestionEvaluator(OperatorABC):

    def __init__(self, 
                 llm_serving: LLMServingABC = None,
                 prompt_template = None
                 ):
        self.prompts = prompt_template
        self.llm_serving = llm_serving

        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def _reformat_prompt(self, dataframe):
        problem = dataframe[self.problem].tolist()
        system_prompt = self.prompts.system_prompt()
        prompts = [self.prompts.prompt(p) for p in problem]

        return system_prompt, prompts

    def run(self, storage: DataFlowStorage, problem: str, output_key: str):
        """
        Run the question generation process.
        """
        self.problem = problem
        dataframe = storage.read("dataframe")

        sys_prompts, user_prompts = self._reformat_prompt(dataframe)
        responses = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
        dataframe[f"{output_key}"] = responses
        self.logger.info(f"Generated questions for {output_key}")

        output_file = storage.write(dataframe)
        self.logger.info(f"Generated questions saved to {output_file}")
