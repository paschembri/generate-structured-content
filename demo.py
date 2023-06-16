# -*- coding: utf-8 -*-
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, LLMRequestsChain
import meta
import catalog


if not os.getenv('OPENAI_API_KEY'):
    raise KeyError('OPENAI_API_KEY is not set in the environment')


def create_llm():
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_retries=1,
        request_timeout=60,
        max_tokens=256,
    )


business_insights = meta.register_service(
    catalog.CompanyActivity, create_llm, LLMRequestsChain
)

survey_maker = meta.register_service(catalog.Survey, create_llm, LLMChain)


def main():
    topic = "Satisfaction about Google Cloud Services"
    with survey_maker(objective=topic, lang='english') as interview:
        print('=' * 80)
        print(interview)
        print('*' * 80)
        print('\n'.join(interview.questions))
        print('=' * 80)

    url = "https://vercel.com"
    with business_insights(url=url, lang='english') as insights:
        print('=' * 80)
        print(insights)
        print('*' * 80)
        print(insights.company_name + '\n')
        print(insights.summary + '\n')
        print(insights.probable_market + '\n')
        print('=' * 80)


if __name__ == '__main__':
    main()
