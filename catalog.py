# -*- coding: utf-8 -*-
from typing import List
from dataclasses import dataclass


@dataclass
class CompanyWebsiteSummary:
    company_name: str = 'The name of the company'
    summary: str = 'A summary of the company activity'
    probable_market: str = 'The most probable market the company is targetting'


class CompanyActivity:
    output_cls = CompanyWebsiteSummary
    prompt = """
    Here is a company website content :
    ----
    {requests_result}
    ----
    We want to learn more about a company's activity and the kind of
    clients they target. Write in {lang}.

    {formatting}
    """


@dataclass
class SurveyContent:
    questions: List[str] = "List of questions for our survey"


class Survey:
    output_cls = SurveyContent
    prompt = """
    We are writing a survey about {objective}

    Write a set of questions in {lang}.

    {formatting}
    """
