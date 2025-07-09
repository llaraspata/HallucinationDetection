PROMPT_QA = """You are an helpful assistant that replies to some questions. 
You will also provided with the answer given by another agent. Don't use it while replying.

Question: {question}

Other agent answer: {answer}

Your answer:"""


PROMPT_CORRECT = """You are an helpful assistant that helps to identify hallucinations.

Given a question and an answer provided by another agent, you have to determine if the answer is correct or not.
Reply with "yes" if the answer is correct, otherwise reply with "no".

Question: {question}

Other agent answer: {answer}

Is the answer correct? (yes/no)"""
