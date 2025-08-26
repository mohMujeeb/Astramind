from typing import Dict, Any, List
from tavily import TavilyClient

def web_search(query: str, api_key: str, k: int = 5) -> Dict[str, Any]:
    tv = TavilyClient(api_key=api_key)
    res = tv.search(query=query, max_results=k)
    # Normalize fields we care about
    results = []
    for item in res.get("results", []):
        results.append({"title": item.get("title"), "url": item.get("url"), "content": item.get("content")})
    return {"results": results}
