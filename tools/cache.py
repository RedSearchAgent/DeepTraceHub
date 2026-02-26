import os
import sys
import json
import hashlib
from datetime import datetime


from dacite import from_dict
from dataclasses import asdict
from typing import Dict, Optional, Union


if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from utils.file_utils import FileUtils
from utils.logger import get_logger
from tools.types import WebContent, OrganicResults, WebContentV2, ScholarResults, MapResults

logger = get_logger(__name__)


class CacheBase:
    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir, exist_ok=True)
            logger.info(f"[Cache Manager] Cache directory {self._cache_dir} doesn't exist, created")
    
    def load(self, key: str):
        raise NotImplementedError("Subclass must implement load method: CacheBase.load")
    
    def save(self, key: str, value: Optional[Dict]):
        raise NotImplementedError("Subclass must implement save method: CacheBase.save")
    
    def _hash_key(self, key: str):
        return hashlib.sha256(key.encode()).hexdigest()

class WebCrawlCache(CacheBase):
    """
    Storage structure for WebCrawlCache:
    hash_key: url
    value: {
        "url": URL of the web page,
        "content": Content of the web page,
        "source": Source of the web page content,
        "timestamp": Timestamp of when the content was retrieved, represented as an ISO standard string
    }
    """
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)
    
    def load(self, key: str) -> Optional[WebContent]:
        url = key
        key = self._hash_key(key)
        try:
            data = FileUtils.load_json(os.path.join(self._cache_dir, f"{key}.json"))
            if isinstance(data, str):
                data = json.loads(data)
            return from_dict(WebContent, data)
        except Exception as e:
            return None
    
    def save(self, key: str, value: Optional[Union[Dict, WebContent]]):
        url = key
        key = self._hash_key(key)
        if isinstance(value, WebContent):
            value = asdict(value)
        value["timestamp"] = datetime.now().isoformat()
        value["url"] = url
        FileUtils.save_json(value, os.path.join(self._cache_dir, f"{key}.json"))

class WebCrawlCacheV2(CacheBase):
    """
    Storage structure for WebCrawlCacheV2:
    hash_key: url
    value: {
        'url': 'https://groups.google.com/g/meetecho-janus/c/dsyZqrq3j_Y',
        'title': 'Interest in a Janus stress testing tool?',
        'snippet': "I've been stress testing Janus a lot over the past few weeks to try to test and reproduce problems with the refcount branch.",
        'content': 'Title: Interest in a Janus stress testing tool?...',
        'rank': 0,
        'source': 'jina',
        'raw_organic': {...}
    }
    """
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)

    def load(self, key: str) -> Optional[WebContentV2]:
        url = key
        key = self._hash_key(key)
        try:
            data = FileUtils.load_json(os.path.join(self._cache_dir, f"{key}.json"))
            if isinstance(data, str):
                data = json.loads(data)
            return from_dict(WebContentV2, data)
        except Exception as e:
            return None
    
    def save(self, key: str, value: Optional[Union[Dict, WebContentV2]]):
        url = key
        key = self._hash_key(key)
        if isinstance(value, WebContentV2):
            value = asdict(value)
        value["timestamp"] = datetime.now().isoformat()
        value["url"] = url
        FileUtils.save_json(value, os.path.join(self._cache_dir, f"{key}.json"))

class SearchCache(CacheBase):
    """
    Storage structure for SearchCache:
    hash_key: query
    value: {
        "query": Search query for the web page,
        "serp_entries": List of Dict of SerpEntry(dataclass to dict),
        "source": Source of the search results,
        "timestamp": Timestamp of when the search results were retrieved, represented as an ISO standard string
    }
    """
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)

    def load(self, key: str) -> Optional[OrganicResults]:
        key = self._hash_key(key)
        try:
            data = FileUtils.load_json(os.path.join(self._cache_dir, f"{key}.json"))
            if isinstance(data, str):
                data = json.loads(data)
            return from_dict(OrganicResults, data)
        except Exception as e:
            return None
    
    def save(self, key: str, value: Optional[Union[Dict, OrganicResults]]):
        key = self._hash_key(key)
        if isinstance(value, OrganicResults):
            value = asdict(value)
        value["timestamp"] = datetime.now().isoformat()
        FileUtils.save_json(value, os.path.join(self._cache_dir, f"{key}.json"))


class ScholarCache(CacheBase):
    """
    Storage structure for ScholarCache:
    hash_key: query
    value: {
        "query": Academic search query,
        "results": List of Dict of ScholarEntry(dataclass to dict),
        "timestamp": Timestamp of when the search results were retrieved, represented as an ISO standard string
    }
    """
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)

    def load(self, key: str) -> Optional[ScholarResults]:
        key = self._hash_key(key)
        try:
            data = FileUtils.load_json(os.path.join(self._cache_dir, f"{key}.json"))
            if isinstance(data, str):
                data = json.loads(data)
            return from_dict(ScholarResults, data)
        except Exception as e:
            return None
    
    def save(self, key: str, value: Optional[Union[Dict, ScholarResults]]):
        key = self._hash_key(key)
        if isinstance(value, ScholarResults):
            value = asdict(value)
        value["timestamp"] = datetime.now().isoformat()
        FileUtils.save_json(value, os.path.join(self._cache_dir, f"{key}.json"))


class MapsCache(CacheBase):
    """
    Storage structure for MapsCache:
    hash_key: query
    value: {
        "query": Map search query,
        "results": List of Dict of MapEntry(dataclass to dict),
        "timestamp": Timestamp of when the search results were retrieved, represented as an ISO standard string
    }
    """
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)

    def load(self, key: str) -> Optional[MapResults]:
        key = self._hash_key(key)
        try:
            data = FileUtils.load_json(os.path.join(self._cache_dir, f"{key}.json"))
            if isinstance(data, str):
                data = json.loads(data)
            return from_dict(MapResults, data)
        except Exception as e:
            return None
    
    def save(self, key: str, value: Optional[Union[Dict, MapResults]]):
        key = self._hash_key(key)
        if isinstance(value, MapResults):
            value = asdict(value)
        value["timestamp"] = datetime.now().isoformat()
        FileUtils.save_json(value, os.path.join(self._cache_dir, f"{key}.json"))
