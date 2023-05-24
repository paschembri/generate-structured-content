import json
from pprint import pprint
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


task_categories_prompt_template = """I am developing a Todo App for
{user_profile}. I need to generate 5 categories to help them
organize their tasks. Write 5 task categories and format the
results as a JSON dictionary containing the key `categories` which
is an array containing the task categories.
"""

dataset = {
    "user_profiles": [
        "Project Manager",
        "Software Developer",
        "Marketing Manager",
    ]
}

prompt = PromptTemplate(
    input_variables=["user_profile"],
    template=task_categories_prompt_template,
)

llm = OpenAI()
generate_task_categories = LLMChain(llm=llm, prompt=prompt)

task_categories = {}

for user_profile in dataset['user_profiles']:
    llm_response = generate_task_categories.run(user_profile)
    task_categories[user_profile] = json.loads(llm_response)


pprint(task_categories)
