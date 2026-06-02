from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json
import re
import os

load_dotenv()

def extract_entities_and_relations(text: str, source: str) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"]
    )

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