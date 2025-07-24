import json
class SequentialFusionPrompt:
    '''
    '''
    def __init__(self):
        pass

    def system_prompt(self) -> str:
        system_prompt = ""
        return system_prompt

    def prompt(self, problem_1, problem_2) -> str:
        prompt = f'''
 # Role: Mathematical Problem Merger
 ## Profile
 Your role is to merge "#Problem 1#" and "#Problem 2#" into a combined problem.
 ## Guidelines
 Step 1: Identify input and output variables in both problems. Determine mathematical relationships and constraints in each
 problem. Locate variables between "#Problem 1#" and "#Problem 2#" that can form sequential dependencies.
 Step 2: Formulate a comprehensive plan to merge the two problems by using "#Problem 1#"’s output variable to
 replace an input variable of "#Problem 2#"’s. Merge contextual elements by embedding both problems within a unified
 real-world scenario or extended narrative, aligning units and measurement systems.
 Step 3: Create a single "#Combined Problem#" where solving "#Problem 1#" is a prerequisite for "#Problem
 ## Output Format
 Please reply strictly in the following format:
 #Elements Identified#:
 #Plan#:
 #Combined Problem#:
 ## Input
 ### #Problem 1#
 {problem_1}
 ### #Problem 2#
 {problem_2}
 2#". Explicitly state variable dependencies and which variable is replaced. Adjust numerical ranges to maintain arithmetic
 consistency. The "#Combined Problem#" should contain no supplementary explanation or note.
 ## Output

'''
        return prompt