# -*- coding: utf-8 -*-
import logging
from functools import wraps
from contextlib import contextmanager
from dataclasses import fields
from langchain.chains import LLMChain, LLMRequestsChain
from langchain.chains.llm_requests import DEFAULT_HEADERS
from langchain.requests import TextRequestsWrapper
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain import PromptTemplate

logger = logging.getLogger(__name__)


def make_chain(llm, base_chain_cls, prompt_template, *args, **kwargs):
    if base_chain_cls == LLMRequestsChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        chain = LLMRequestsChain(llm_chain=llm_chain, *args, **kwargs)

        chain.requests_wrapper = TextRequestsWrapper(headers=DEFAULT_HEADERS)

    else:
        chain = base_chain_cls(
            llm=llm, prompt=prompt_template, *args, **kwargs
        )

    @wraps(chain)
    def wrapper(*args, **kwargs):
        response = chain(*args, **kwargs)
        return response[chain.output_key]

    return wrapper


def make_structured_chain(llm, cls, base_chain_cls, *args, **kwargs):
    schema = [
        ResponseSchema(
            **{
                'name': f.name,
                'description': f.default,
                'type': str(f.type),
            }
        )
        for f in fields(cls.output_cls)
    ]
    parser = StructuredOutputParser.from_response_schemas(schema)
    formatting = parser.get_format_instructions()
    prompt_template = PromptTemplate.from_template(
        template=cls.prompt,
        output_parser=parser,
        partial_variables={'formatting': formatting},
    )

    chain = make_chain(llm, base_chain_cls, prompt_template, *args, **kwargs)

    return chain, parser


def register_service(model_cls, llm_factory, base_chain_cls):
    @contextmanager
    def service(**kwargs):
        llm = llm_factory()
        chain, parser = make_structured_chain(llm, model_cls, base_chain_cls)

        try:
            response = chain(kwargs)
            logger.info(f'Running chain {chain} with {kwargs}')
            logger.info(f'Results: {response}')
            structured_resp = parser.parse(response)
            yield model_cls.output_cls(**structured_resp)
        finally:
            pass

    return service
