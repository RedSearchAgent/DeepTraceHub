"""
Google Scholar Tool Module

Implements academic paper search functionality based on Serper.dev's Google Scholar API.
Provides an asynchronous search interface with support for caching and retry mechanisms.
"""
import os
import sys
import asyncio
import aiohttp
import traceback
from typing import Any, Dict, List, Optional

from aiohttp import ClientError

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from utils.file_utils import FileUtils
from utils.logger import get_logger
from tools.types import ScholarEntry, ScholarResults
from tools.cache import ScholarCache
from tools.search import SearchToolBase, execute_with_retries

logger = get_logger(__name__)

class GoogleScholarTool(SearchToolBase):
    """
    Google Scholar search tool based on Serper.dev
    
    Example usage:
        tool = GoogleScholarTool()
        results = await tool.search("machine learning")
    """
    name = "google_scholar"
    category = "scholar"
    
    def __init__(self, config_path: str = "config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)
        self.__serper_scholar_url = "https://google.serper.dev/scholar"
        self.__serper_dev_token = os.getenv("SERPERDEV_TOKEN")
        self.cache = ScholarCache(cache_dir=f"{self.config['scholar_cache_dir']}")
    
    async def search(self, query: str) -> ScholarResults:
        """
        Search Google Scholar for academic papers
        
        Args:
            query: Search query string
            
        Returns:
            ScholarResults: Data structure containing academic paper search results
        """
        # Check cache
        cached_data = await asyncio.to_thread(self.cache.load, key=query)
        if cached_data is not None:
            logger.debug(f"[GoogleScholarTool.search] Cache hit for '{query}'")
            return cached_data
        
        serper_api_key = self.__serper_dev_token
        payload = {
            "q": query,
            "num": self.config.get("num", 10),
        }
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 30))

        async def _perform_scholar_search() -> Dict[str, Any]:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.__serper_scholar_url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            raise RuntimeError(
                                f"Failed to search with Google Scholar: {response.status} {body}"
                            )
                        return await response.json()
            except ClientError as exc:
                raise RuntimeError("Google Scholar request failed") from exc

        try:
            results = await execute_with_retries(
                _perform_scholar_search,
                context=f"google scholar search for '{query}'",
                max_attempts=self.config["retry"]["max_attempts"],
                delay=self.config["retry"]["delay"],
            )
        except Exception as exc:
            logger.error(f"[GoogleScholarTool.search] Google Scholar search failed for '{query}': {exc}")
            logger.error(traceback.format_exc())
            return ScholarResults(query=f"Failed to search for '{query}'", results=[])

        # Check if there are search results
        if "organic" not in results or not results["organic"]:
            logger.warning(f"[GoogleScholarTool.search] No results found for '{query}'")
            return ScholarResults(query=query, results=[])

        # Parse search results
        scholar_entries: List[ScholarEntry] = []
        for idx, result in enumerate(results["organic"]):
            entry = ScholarEntry(
                title=result.get("title", ""),
                link=result.get("link"),
                pdf_url=result.get("pdfUrl"),
                snippet=result.get("snippet"),
                publication_info=result.get("publicationInfo"),
                year=result.get("year"),
                cited_by=result.get("citedBy"),
                rank=idx,
                raw_organic=result
            )
            scholar_entries.append(entry)
        
        final_scholar_results = ScholarResults(query=query, results=scholar_entries)
        
        # Save to cache
        await asyncio.to_thread(self.cache.save, key=query, value=final_scholar_results)
        
        return final_scholar_results
    
    def format_results(self, results: ScholarResults) -> str:
        """
        Format search results into readable text
        
        Args:
            results: ScholarResults search results
            
        Returns:
            str: Formatted text
        """
        if not results.results:
            return f"No results found for '{results.query}'. Try with a more general query."
        
        web_snippets = []
        for idx, entry in enumerate(results.results, 1):
            # Build date info
            date_published = ""
            if entry.year:
                date_published = f"\nDate published: {entry.year}"
            
            # Build publication info
            publication_info = ""
            if entry.publication_info:
                publication_info = f"\nPublication Info: {entry.publication_info}"
            
            # Build snippet
            snippet = ""
            if entry.snippet:
                snippet = f"\nSnippet: {entry.snippet}"
            
            # Build link info
            link_info = "no available link"
            if entry.pdf_url:
                link_info = f"pdfUrl: {entry.pdf_url}"
            elif entry.link:
                link_info = f"link: {entry.link}"
            
            # Build citation info
            cited_by = ""
            if entry.cited_by:
                cited_by = f"\nCited by: {entry.cited_by}"
            
            redacted_version = f"{idx}. [{entry.title}]({link_info}){publication_info}{date_published}{cited_by}{snippet}"
            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)
        
        content = f"A Google Scholar search for '{results.query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
        return content


# Test code
if __name__ == "__main__":
    async def _test_scholar():
        tool = GoogleScholarTool()
        
        # Test single query
        query = "deep learning neural network"
        print(f"Testing Google Scholar search for: '{query}'")
        
        results = await tool.search(query)
        print(f"\nFound {len(results.results)} results:")
        
        # Print formatted results
        formatted = tool.format_results(results)
        print(formatted)
        
        # Print raw data structure
        print("\n" + "="*50)
        print("Raw ScholarResults structure:")
        for i, entry in enumerate(results.results[:3], 1):
            print(f"\n--- Entry {i} ---")
            print(f"Title: {entry.title}")
            print(f"Link: {entry.link}")
            print(f"PDF URL: {entry.pdf_url}")
            print(f"Year: {entry.year}")
            print(f"Cited by: {entry.cited_by}")
            print(f"Publication Info: {entry.publication_info}")
        
        # Test cache functionality
        print("\n" + "="*50)
        print("Testing cache (should be faster)...")
        results2 = await tool.search(query)
        print(f"Cache test completed, got {len(results2.results)} results")
    
    asyncio.run(_test_scholar())
