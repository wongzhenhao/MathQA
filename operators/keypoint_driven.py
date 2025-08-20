import pandas as pd
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from typing import List, Dict

@OPERATOR_REGISTRY.register()
class KeyPointDriven(OperatorABC):

    def __init__(self, 
                 llm_serving: LLMServingABC = None,
                 prompt_template = None
                 ):
        self.llm_serving = llm_serving
        self.prompts = prompt_template

        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def _reformat_prompt(self, dataframe):
        question = dataframe[self.input_question_key].tolist()
        solution = dataframe[self.input_solution_key].tolist()
        system_prompt = self.prompts.system_prompt()
        prompts = [self.prompts.prompt(p1,p2) for p1,p2 in enumerate(zip(question, solution))]

        return system_prompt, prompts

    def parse_extraction_result(self, result: str) -> Dict[str, List[str]]:
        """
        解析知识提取结果
        
        Args:
            result: API返回的提取结果
            
        Returns:
            {主题: [关键点列表]}
        """
        topics_kps = {}
        current_topic = None
        
        lines = result.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是主题行
            if line.startswith('主题') and '：' in line:
                topic_part = line.split('：', 1)[1].strip()
                current_topic = topic_part
                topics_kps[current_topic] = []
            # 检查是否是关键点行
            elif line.startswith('关键点') and '：' in line and current_topic:
                kp_part = line.split('：', 1)[1].strip()
                topics_kps[current_topic].append(kp_part)
        
        return topics_kps

    def run(self, storage: DataFlowStorage, 
            input_question_key: str="question", 
            input_solution_key: str="solution", 
            output_keypoints_key: str = "key_points",
            ):
        """
        Run the question generation process.
        """
        self.input_question_key, self.input_solution_key = input_question_key, input_solution_key
        dataframe = storage.read("dataframe")

        sys_prompts, user_prompts = self._reformat_prompt(dataframe)
        results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
        dataframe[f"{output_keypoints_key}"] = results

        self.logger.info(f"Generated questions for {output_keypoints_key}")
        
        # questions = dataframe[self.input_question_key].tolist()
        # solutions = dataframe[self.input_solution_key].tolist()
        # problems = [{'question': q, 'solution': s} for q, s in zip(questions, solutions)]
        # extracted_knowledge = []
        
        # for i, (problem, result) in enumerate(zip(problems, results)):
        #     if result:
        #         try:
        #             topics_kps = self.parse_extraction_result(result)
        #             extracted_knowledge.append({
        #                 'problem_id': i,
        #                 'question': problem['question'],
        #                 'solution': problem['solution'],
        #                 'topics_kps': topics_kps,
        #                 'raw_extraction': result
        #             })
        #         except Exception as e:
        #             self.logger.error(f"解析第{i+1}个问题的提取结果失败: {e}")
        #             extracted_knowledge.append({
        #                 'problem_id': i,
        #                 'question': problem['question'],
        #                 'solution': problem['solution'],
        #                 'topics_kps': {},
        #                 'raw_extraction': result,
        #                 'error': str(e)
        #             })
        #     else:
        #         self.logger.warning(f"第{i+1}个问题的知识提取失败")
        #         extracted_knowledge.append({
        #             'problem_id': i,
        #             'question': problem['question'],
        #             'solution': problem['solution'],
        #             'topics_kps': {},
        #             'raw_extraction': None,
        #             'error': "API调用失败"
        #         })
        
        # self.logger.info(f"知识提取完成，成功提取{len([x for x in extracted_knowledge if x['topics_kps']])}个问题的知识")

        output_file = storage.write(dataframe)
        self.logger.info(f"Generated questions saved to {output_file}")
