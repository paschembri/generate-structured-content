# -*- coding: utf-8 -*-
from typing import List
from dataclasses import dataclass, fields
from functools import wraps
from contextlib import contextmanager
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0,
    max_retries=3,
    request_timeout=60,
    max_tokens=256,
)


# Some meta magic here :)
def register_chain(cls):
    @wraps(cls)
    @contextmanager
    def wrapper(llm, **kwargs):
        schema = [
            ResponseSchema(
                **{
                    'name': f.name,
                    'description': f.default,
                    'type': f.type.__name__,
                }
            )
            for f in fields(cls.model)
        ]
        parser = StructuredOutputParser.from_response_schemas(schema)
        formatting = parser.get_format_instructions()
        prompt_template = PromptTemplate.from_template(
            template=cls.prompt,
            output_parser=parser,
            partial_variables={'formatting': formatting},
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        try:
            response = chain.predict_and_parse(**kwargs)
            yield cls.model(**response)
        finally:
            pass

    return wrapper


@dataclass
class UserStoryModel:
    user_profile: str = "A user profile (superuser, user, anonymous user)"
    user_intent: str = "A specific goal"
    user_action: str = "A specific action or a set of simple actions"
    acceptance_tests: List[
        str
    ] = "A list of acceptance tests written as assertions"


@register_chain
class UserStoryChain:
    model = UserStoryModel
    prompt = """
    We are writing user stories to help product owners spend more time
    with clients. We are using a new kind of software to write user stories.

    This software uses specific metadata to help write user stories faster.

    Based on the brief description hereunder, generate metadata for a user story.

    Brief description : {description}

    {formatting}
    """


@dataclass
class TodoModel:
    name: str = "The task name"
    category: str = "A task category"
    priority: str = "Task priority (low, normal, urgent)"


@register_chain
class TodoChain:
    model = TodoModel
    prompt = """
    We are populating a database of a project management tool with sample tasks.

    Based on the project description hereunder, generate metadata for a task.

    Project description : {description}

    {formatting}
    """


if __name__ == '__main__':
    with UserStoryChain(llm, description="Reset password with 2FA") as story:
        print(story)

    with TodoChain(llm, description="Ecommerce site rebranding") as todo:
        print(todo)
