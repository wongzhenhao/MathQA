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
    
class ParallelFusionPrompt:
    '''
    '''
    def __init__(self):
        pass

    def system_prompt(self) -> str:
        system_prompt = ""
        return system_prompt

    def prompt(self, problem_1, problem_2) -> str:
        prompt = f'''
 # Role: Mathematical Problem Synthesizer
 ## Profile Your role is to organically integrate "#Problem 1#" and "#Problem 2#" to create a novel problem that
 requires advanced synthesis of their mathematical essence.
 ## Guidelines
 Step 1: Conduct deep structural analysis of both problems by identifying their fundamental mathematical operations,
 contextual frameworks, and cognitive patterns. Extract the underlying logical architectures while preserving their distinctive
 solution pathways.
 Step 2: Develop an innovative fusion mechanism by discovering non-obvious mathematical connections between
 the problems’ core concepts. Construct a multidimensional scenario that naturally embeds both original contexts through
 temporal sequencing, spatial superposition, or conceptual analogy. Engineer hybrid parameters that inherit characteristics
 from both source problems while introducing emergent properties.
 Step 3: Formulate the synthesized problem through strategic recombination of mathematical elements, ensuring
 the new problem requires concurrent application of both original solution strategies. Introduce controlled complexity
 problems’ answers.
 ## Output Format
 Please reply strictly in the following format:
 #Core Elements#:
 #Synthesis Method#:
 #New Problem#:
 ## Input
 ### #Problem 1#
 {problem_1}
 ### #Problem 2#
 {problem_2}
 through cross-domain constraints and self-verification mechanisms that establish mathematical consistency with both source
 ## Output


'''
        return prompt
    
class ConditionFusionPrompt:
    '''
    '''
    def __init__(self):
        pass

    def system_prompt(self) -> str:
        system_prompt = ""
        return system_prompt

    def prompt(self, problem_1, problem_2) -> str:
        prompt = f'''
 # Role: Problem Integrator
 ## Profile
 Create a real-world problem where the solution requires solving both "#Problem 1#" and "#Problem 2#" independently.
 **Ensure the the final answer is either from "#Problem 1#" or "#Problem 2#", depends on the "#New Question#"**.
 ## Guidelines
 Step 1: Analyze "#Problem 1#" and "#Problem 2#" and make sure that the output variables they ask about are of the same
 type. If they are different (for example, one asks about time and the other asks about price), modify one of the problem so that
 it asks about the same variable as the other.
 Step 2: Design a unified problem scenario that combines "#Problem 1#" and "#Problem 2#". Introduce a "#New Question#",
 which must be related with both "#Problem 1#" and "#Problem 2#". Ensure that final answer of the "#New Question#" must
 either come from "#Problem 1#" or "#Problem 2#". This means that the "#New Question#" should be an **comparison**
 and **selection** of the previous answers, not their **combination**. There are some examples for the "#New Question#":
 1. Who sells the most items?
 2. Howmuch money does the top earner make?
 3. Which is the cheaper plan?
 4. Someone has 200 dollor, which item can he afford?
 phrases "#Problem 1#" and "#Problem 2#" in the generated "#New Problem#".
 ## Output Format
 Please reply strictly in the following format:
 #Analysis#:
 #New Question#:
 #New Problem#:
 ## Input
 ### #Problem 1#
 {problem_1}
 ### #Problem 2#
 {problem_2}
 Step 3: Provide the "#New Problem#", which combine "#Problem 1#", "#Problem 2#", and "#New Question#" in a unified
 real-world scenario. Don’t contain solution of "#Problem 1#" and "#Problem 2#" in "#New Problem#". Avoid using the
 ## Output
'''
        return prompt
    
class QuestionEvaluationPrompt:
    '''
    '''
    def __init__(self):
        pass

    def system_prompt(self) -> str:
        system_prompt = ""
        return system_prompt

    def prompt(self, problem) -> str:
        prompt = f'''
 # Role: Mathematics Grading Teacher
 ## Profile
 You are a senior mathematics grading teacher in university, very skilled in high difficulty fields such as Intermediate Algebra,
 Precalculus, Prealgebra, Number Theory, Geometry, Counting & Probability, Algebra and so on.
 ## Guidelines
 Your task is to act as an impartial judge to evaluate the statement completeness and correctness of math problem according to
 the following rules:
 1. Assess the clarity and accuracy of the definition of each math problem. Ensure that the problem statement provides
 sufficient information, conditions, and constraints.
 2. Consider whether the problem allows for multiple interpretations or if further clarification is needed.
 3. Evaluate the clarity of mathematical notation and terminology used in the problem.
 ## Output Format
 Please reply strictly in the following format:
 #Judgement#:
 #Explanation#:
 ## Input
 {problem}
 4. Evaluate whether the math problem is solvable. If the math problem meet the rules above, output "True" in "#Judge
ment#", else "False". You should also give your explanation in "#Explanation#".
 ## Output
'''
        return prompt