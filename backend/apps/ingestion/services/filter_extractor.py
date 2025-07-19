# import os
# import json
# import openai
# from django.conf import settings

# # Make sure you’ve set OPENAI_API_KEY in your Django settings
# openai.api_key = settings.OPENAI_API_KEY

# # A single‐shot prompt that teaches the LLM to output JSON with a "filters" object.
# FILTER_PROMPT = """
# You are a smart assistant that reads a user’s question and extracts the key filter
# parameters they want to apply when searching through a report.  You MUST output
# valid JSON with exactly one top‐level key "filters", whose value is an object mapping
# filter names to their values.

# Examples:

# Q: "What methodology was used for the kinematic optimization of the exoskeleton?"
# A: {{"filters": {{"system": "exoskeleton", "aspect": "kinematic optimization", "category": "methodology"}}}}

# Q: "Which materials and fabrication techniques were selected and why?"
# A: {{"filters": {{"category": "materials and fabrication techniques"}}}}

# Then, produce the filters for this question:

# Question: "{question}"
# """

# def extract_filters_llm(question: str) -> dict:
#     """
#     Calls OpenAI to extract filter keywords from the user's question.
#     Returns the "filters" dict, or {} if nothing is found or parsing fails.
#     """
#     prompt = FILTER_PROMPT.format(question=question)

#     resp = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=prompt,
#         temperature=0,
#         max_tokens=200,
#         top_p=1.0,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=["\n\n"]
#     )

#     raw = resp.choices[0].text.strip()
#     try:
#         parsed = json.loads(raw)
#         return parsed.get("filters", {})
#     except json.JSONDecodeError:
#         # If LLM didn’t return valid JSON, bail out
#         return {}


# apps/ingestion/services/filter_extractor.py
from typing import List, Dict, Any
import openai, json
from django.conf import settings

def extract_filters_llm(query: str) -> Dict[str, Any]:
    """
    TODO: LLM-powered filter extraction to be implemented later.
    For now, always return an empty dict so downstream logic can proceed.
    """
    return {}
