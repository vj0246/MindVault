from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from security.groq_keys import get_groq_keys
from dotenv import load_dotenv
import json
import re
import os

load_dotenv()

_EXTRACTOR_LLM = None

def _get_extractor_llm():
    # Cached singleton -- was creating a new ChatGroq (+ httpx pool) on every
    # call. Upload loops this up to 10x, so 10 fresh connection pools per
    # upload. Reuse one instance instead. Chains one ChatGroq per configured
    # GROQ_API_KEY via with_fallbacks() so a rate-limited/exhausted key falls
    # through to the next (no-op with a single key).
    global _EXTRACTOR_LLM
    if _EXTRACTOR_LLM is None:
        candidates = [
            ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=key)
            for key in get_groq_keys()
        ]
        primary, *fallbacks = candidates
        _EXTRACTOR_LLM = primary.with_fallbacks(fallbacks) if fallbacks else primary
    return _EXTRACTOR_LLM

def extract_entities_and_relations(text: str, source: str) -> dict:
    llm = _get_extractor_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a knowledge graph builder. Extract information from the text below.

Return ONLY valid JSON, no explanation, no markdown:
{{
  "entities": ["concept1", "concept2"],
  "relationships": [
    {{"subject": "thing that acts", "relation": "verb phrase", "object": "thing being acted on"}}
  ]
}}

Rules for entities:
- Extract ALL technical terms, algorithms, conditions, and concepts
- Include everything mentioned even if it seems minor
- Maximum 10 entities
- Use lowercase

Rules for relationships:
- Subject is the thing DOING the action
- Object is the thing RECEIVING the action  
- Example: "Banker's Algorithm prevents Deadlock" → subject=banker's algorithm, relation=prevents, object=deadlock
- Maximum 6 relationships
- Only extract explicitly stated relationships

Text:
{text}

JSON only:
""")

    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({"text": text[:800]})
        cleaned = re.sub(r"```json|```", "", result).strip()
        data = json.loads(cleaned)

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        print(f"[Extractor] Found {len(entities)} entities, {len(relationships)} relationships")
        return {
            "entities": entities,
            "relationships": relationships,
            "source": source
        }
    except Exception as e:
        print(f"[Extractor] Failed: {e}")
        return {"entities": [], "relationships": [], "source": source}