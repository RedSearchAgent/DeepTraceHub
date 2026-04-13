"""
Type definition module

This module defines various data types used in the system, including:
- WebContent: Web page content data structure
- SerpEntry: Search engine result page entry
- OrganicResults: Collection of search results
- Various enumeration types
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Union


class SerpProvider(str, Enum):
    EMPTY = "empty"
    DUCKDUCKGO = "duckduckgo"
    SERPER = "serper.dev"
    BING = "bing"
    APISERP = "api.serp"
    XHS = "xhs"
    BOCHA = "bocha"
    WANGSU = "wangsu"
    WANGSU_SMART = "wangsu_smart"
    LOCAL = "local"
    TAVILY = "tavily"

class WebProvider(str, Enum):
    SERPER = "serper.dev"
    JINA = "jina"
    GEMINI = "gemini"
    XHS = "xhs"
    LOCAL = "local"
    TAVILY = "tavily"


@dataclass
class WebContent:
    """
    Web page text content obtained from parsing via Jina or SERPER.DEV, etc.
    Currently, only the text_content field is used in the code.
    """
    text_content: Optional[str] = None
    html_content: Optional[str] = None
    markdown_content: Optional[str] = None
    source: Optional[Union[WebProvider, str]] = None

@dataclass
class WebContentV2:
    """
    Combines SerpEntry and the original WebContent to enrich the cached web page information.
    The schema of WebContentV2 corresponds to the cache of crawl_v2.
    In practice, it is largely the same as the fields in SerpEntry.
    """
    url: str
    title: str
    snippet: str
    content: Optional[Union[WebContent, str]] = None
    rank: int = 0
    source: Optional[Union[SerpProvider, str]] = None
    raw_organic: Optional[Dict] = None

    @property
    def text_content(self) -> Optional[str]:
        if isinstance(self.content, WebContent):
            return self.content.text_content
        elif isinstance(self.content, str):
            return self.content
        else:
            return None

@dataclass
class SerpEntry:
    url: str
    title: str
    snippet: str
    content: Optional[Union[WebContent, str]] = None
    rank: int = 0
    source: Optional[Union[SerpProvider, str]] = None
    raw_organic: Optional[Dict] = None


@dataclass
class OrganicResults:
    query: str
    results: List[SerpEntry]


# ===================== Google Scholar Types =====================
@dataclass
class ScholarEntry:
    """Google Scholar academic paper search result entry"""
    title: str
    link: Optional[str] = None
    pdf_url: Optional[str] = None
    snippet: Optional[str] = None
    publication_info: Optional[str] = None
    year: Optional[int] = None
    cited_by: Optional[int] = None
    rank: int = 0
    raw_organic: Optional[Dict] = None


@dataclass
class ScholarResults:
    """Google Scholar search results collection"""
    query: str
    results: List[ScholarEntry]


# ===================== Google Maps Types =====================
@dataclass
class MapEntry:
    """Google Maps location search result entry"""
    title: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    category: Optional[str] = None
    phone_number: Optional[str] = None
    website: Optional[str] = None
    cid: Optional[str] = None
    place_id: Optional[str] = None
    rank: int = 0
    raw_organic: Optional[Dict] = None


@dataclass
class MapResults:
    """Google Maps search results collection"""
    query: str
    results: List[MapEntry]
