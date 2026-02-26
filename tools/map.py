"""
Google Maps Tool Module

Implements location search functionality based on Serper.dev's Google Maps API.
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
from tools.types import MapEntry, MapResults
from tools.cache import MapsCache
from tools.search import SearchToolBase, execute_with_retries

logger = get_logger(__name__)


class GoogleMapsTool(SearchToolBase):
    """
    Google Maps search tool based on Serper.dev
    
    Example usage:
        tool = GoogleMapsTool()
        results = await tool.search("coffee shop near me")
    """
    name = "google_maps"
    category = "maps"
    
    def __init__(self, config_path: str = "config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)
        self.__serper_maps_url = "https://google.serper.dev/maps"
        self.__serper_dev_token = os.getenv("SERPERDEV_TOKEN")
        self.cache = MapsCache(cache_dir=f"{self.config['maps_cache_dir']}")
    
    async def search(self, query: str, page: int = 1) -> MapResults:
        """
        Search for locations on Google Maps
        
        Args:
            query: Search query string
            page: Result page number, defaults to 1
            
        Returns:
            MapResults: Data structure containing location search results
        """
        # Build cache key (including page number)
        cache_key = f"{query}__page_{page}"
        
        # Check cache
        cached_data = await asyncio.to_thread(self.cache.load, key=cache_key)
        if cached_data is not None:
            logger.debug(f"[GoogleMapsTool.search] Cache hit for '{query}' page {page}")
            return cached_data
        
        serper_api_key = self.__serper_dev_token
        payload = {
            "q": query,
            "page": page,
        }
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 30))

        async def _perform_maps_search() -> Dict[str, Any]:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.__serper_maps_url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            raise RuntimeError(
                                f"Failed to search with Google Maps: {response.status} {body}"
                            )
                        return await response.json()
            except ClientError as exc:
                raise RuntimeError("Google Maps request failed") from exc

        try:
            results = await execute_with_retries(
                _perform_maps_search,
                context=f"google maps search for '{query}'",
                max_attempts=self.config["retry"]["max_attempts"],
                delay=self.config["retry"]["delay"],
            )
        except Exception as exc:
            logger.error(f"[GoogleMapsTool.search] Google Maps search failed for '{query}': {exc}")
            logger.error(traceback.format_exc())
            return MapResults(query=f"Failed to search for '{query}'", results=[])

        # Check if there are search results
        if "places" not in results or not results["places"]:
            logger.warning(f"[GoogleMapsTool.search] No results found for '{query}'")
            return MapResults(query=query, results=[])

        # Parse search results
        map_entries: List[MapEntry] = []
        for idx, result in enumerate(results["places"]):
            entry = MapEntry(
                title=result.get("title", ""),
                address=result.get("address"),
                latitude=result.get("latitude"),
                longitude=result.get("longitude"),
                rating=result.get("rating"),
                rating_count=result.get("ratingCount"),
                category=result.get("category"),
                phone_number=result.get("phoneNumber"),
                website=result.get("website"),
                cid=result.get("cid"),
                place_id=result.get("placeId"),
                rank=idx,
                raw_organic=result
            )
            map_entries.append(entry)
        
        final_map_results = MapResults(query=query, results=map_entries)
        
        # Save to cache
        await asyncio.to_thread(self.cache.save, key=cache_key, value=final_map_results)
        
        return final_map_results
    
    def format_results(self, results: MapResults) -> str:
        """
        Format search results into readable text
        
        Args:
            results: MapResults search results
            
        Returns:
            str: Formatted text
        """
        if not results.results:
            return f"No results found for '{results.query}'. Try with a different query."
        
        place_snippets = []
        for idx, entry in enumerate(results.results, 1):
            # Build basic info
            place_info = f"{idx}. **{entry.title}**"
            
            # Add category
            if entry.category:
                place_info += f"\n   Category: {entry.category}"
            
            # Add address
            if entry.address:
                place_info += f"\n   Address: {entry.address}"
            
            # Add rating
            if entry.rating is not None:
                rating_str = f"\n   Rating: {entry.rating}"
                if entry.rating_count:
                    rating_str += f" ({entry.rating_count} reviews)"
                place_info += rating_str
            
            # Add phone
            if entry.phone_number:
                place_info += f"\n   Phone: {entry.phone_number}"
            
            # Add website
            if entry.website:
                place_info += f"\n   Website: {entry.website}"
            
            # Add coordinates
            if entry.latitude is not None and entry.longitude is not None:
                place_info += f"\n   Coordinates: ({entry.latitude}, {entry.longitude})"
            
            # Add Place ID
            if entry.place_id:
                place_info += f"\n   Place ID: {entry.place_id}"
            
            place_snippets.append(place_info)
        
        content = f"A Google Maps search for '{results.query}' found {len(place_snippets)} places:\n\n## Places Results\n" + "\n\n".join(place_snippets)
        return content


# Test code
if __name__ == "__main__":
    async def _test_maps():
        tool = GoogleMapsTool()
        
        # Test single query
        query = "coffee shop in San Francisco"
        print(f"Testing Google Maps search for: '{query}'")
        
        results = await tool.search(query)
        print(f"\nFound {len(results.results)} places:")
        
        # Print formatted results
        formatted = tool.format_results(results)
        print(formatted)
        
        # Print raw data structure
        print("\n" + "="*50)
        print("Raw MapResults structure:")
        for i, entry in enumerate(results.results[:3], 1):
            print(f"\n--- Place {i} ---")
            print(f"Title: {entry.title}")
            print(f"Address: {entry.address}")
            print(f"Category: {entry.category}")
            print(f"Rating: {entry.rating} ({entry.rating_count} reviews)")
            print(f"Phone: {entry.phone_number}")
            print(f"Website: {entry.website}")
            print(f"Coordinates: ({entry.latitude}, {entry.longitude})")
            print(f"Place ID: {entry.place_id}")
        
        # Test cache functionality
        print("\n" + "="*50)
        print("Testing cache (should be faster)...")
        results2 = await tool.search(query)
        print(f"Cache test completed, got {len(results2.results)} places")
        
        # Test pagination functionality
        print("\n" + "="*50)
        print("Testing pagination (page 2)...")
        results3 = await tool.search(query, page=2)
        print(f"Page 2 results: {len(results3.results)} places")
    
    asyncio.run(_test_maps())
