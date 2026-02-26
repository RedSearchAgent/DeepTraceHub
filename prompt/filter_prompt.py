import re

JUDGE_ENTITY_EQUIVALENT_PROMPT = """You are an entity matching and disambiguation system specialized in determining whether two descriptions refer to the same real-world entity.

Given:
- An answer entity (name, type, aliases, optional disambiguation hints)
- A search engine organic result (web title, snippet, URL)

Determine whether the entity described by the web page is the same entity as the answer entity.

Make the judgment only based on the web title and snippet (URL or domain may be used as weak evidence).
Do not use external knowledge or assumptions.

Answer Entity: {answer}
Search Organic Result:
- Title: {title}
- Snippet: {snippet}
- URL: {url}

Your judgement must be in the format and criteria specified below:

explanation: Briefly explain the decision using evidence from the title and/or snippet.

match: Answer 'yes' if the web page is the same entity as the answer entity, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. 
""".strip()


def parse_entity_equivalent_output(output: str):
    _entity_equivalent_re_pattern = r"^(match|explanation):\s*(.+)$"
    try:
        json_output = dict(re.findall(_entity_equivalent_re_pattern, output, re.MULTILINE))
        json_output = {k: v.strip() for k, v in json_output.items()}
        json_output["match"] = bool(json_output["match"] == "yes")
        json_output["explanation"] = json_output.get("explanation", "").strip()
        return json_output
    except Exception as e:
        return {
            "match": False,
            "explanation": None,
            "error": str(e)
        }