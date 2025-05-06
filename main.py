from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, status
from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Dict, Any, Optional, Union
import asyncio
import logging
import time
from datetime import datetime
import spacy
from newspaper import Article, Config
from markdownify import markdownify as md
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake
import socials
import socid_extractor
import socialshares
from spacy import displacy
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import re
import html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nlp_api")

# Initialize FastAPI Router with more detailed description
app = APIRouter(
    prefix="/api/v1",
    tags=["nlp"],
    
)

# Initialize NLP components with lazy loading
class NLPComponents:
    _nlp = None
    _sentiment_analyzer = None
    _kw_extractor = None
    
    @property
    def nlp(self):
        if self._nlp is None:
            logger.info("Loading SpaCy model...")
            self._nlp = spacy.load("en_core_web_md")
        return self._nlp
    
    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            logger.info("Initializing sentiment analyzer...")
            self._sentiment_analyzer = SentimentIntensityAnalyzer()
        return self._sentiment_analyzer
    
    def get_keyword_extractor(self, language="en", n=1, dedup_lim=0.9, top=5):
        """Get a configured YAKE keyword extractor"""
        return yake.KeywordExtractor(lan=language, n=n, dedupLim=dedup_lim, top=top)

# Initialize global components
nlp_components = NLPComponents()

# Constants
EXCLUDED_ENTITY_TYPES = {"TIME", "DATE", "LANGUAGE", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
STRIP_TEXT_RULES = ["a"]
DEFAULT_USER_AGENT = "NLP/1.0.0 (Unix; Intel) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
SOCIAL_PLATFORMS = ["facebook", "pinterest", "linkedin", "reddit", "twitter", "instagram"]

# Cache for article processing
article_cache = {}

# Request Models with enhanced validation
class ArticleAction(BaseModel):
    link: HttpUrl = Field(..., description="URL of the article to analyze")
    cache: bool = Field(True, description="Whether to use cached results if available")
    
    class Config:
        schema_extra = {
            "example": {
                "link": "https://example.com/article",
                "cache": True
            }
        }

class SummarizeAction(BaseModel):
    text: str = Field(..., min_length=10, description="Text to summarize or extract tags from")
    max_length: Optional[int] = Field(None, description="Maximum length of summary")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This is a sample text for extracting keywords and analyzing sentiment.",
                "max_length": 100
            }
        }

class EntityFilterOptions(BaseModel):
    exclude_types: List[str] = Field(
        default_factory=lambda: list(EXCLUDED_ENTITY_TYPES),
        description="Entity types to exclude"
    )
    min_length: int = Field(1, description="Minimum length of entity text")

# Response Models
class KeywordResponse(BaseModel):
    keyword: str
    score: float

class EntityResponse(BaseModel):
    type: str
    text: str
    
class SentimentResponse(BaseModel):
    compound: float
    positive: float
    negative: float
    neutral: float
    
class ArticleResponse(BaseModel):
    title: str
    date: Optional[datetime]
    text: str
    markdown: str
    html: str
    summary: str
    keywords: List[KeywordResponse]
    authors: List[str]
    banner: Optional[str]
    images: List[str]
    entities: List[EntityResponse]
    videos: List[str]
    social_accounts: Dict[str, Any]
    sentiment: SentimentResponse
    accounts: Dict[str, Any]
    social_shares: Dict[str, Any]
    processing_time: float

# Helper Functions
def get_cache_key(url: str) -> str:
    """Generate a cache key for a URL"""
    return hashlib.md5(url.encode()).hexdigest()

async def fetch_article(link: str):
    """Fetch and parse an article using the Newspaper library."""
    try:
        config = Config()
        config.browser_user_agent = DEFAULT_USER_AGENT
        config.request_timeout = 15
        config.fetch_images = True
        config.memoize_articles = True
        config.follow_meta_refresh = True
        
        article = Article(link, config=config, keep_article_html=True)
        
        # Use asyncio to run blocking I/O operations in a separate thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, article.download)
        await loop.run_in_executor(None, article.parse)
        
        # If article text is too short, try alternative parsing
        if len(article.text) < 50:
            logger.info(f"Article text too short ({len(article.text)} chars), trying alternative parsing")
            # Try to extract text from HTML directly as fallback
            clean_text = re.sub(r'<script.*?>.*?</script>', '', article.html, flags=re.DOTALL)
            clean_text = re.sub(r'<style.*?>.*?</style>', '', clean_text, flags=re.DOTALL)
            clean_text = re.sub(r'<[^>]*>', ' ', clean_text)
            clean_text = html.unescape(clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) > len(article.text):
                article.text = clean_text
                
        # Try to extract summary if not done already
        if not article.summary:
            await loop.run_in_executor(None, article.nlp)
            
        return article
    except Exception as e:
        logger.error(f"Error fetching article {link}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not fetch article: {str(e)}"
        )

def extract_keywords(text: str, language="en", n=1, dedup_lim=0.9, top=5):
    """Extract keywords using YAKE."""
    if not text or len(text) < 10:
        return []
        
    try:
        extractor = nlp_components.get_keyword_extractor(
            language=language, n=n, dedup_lim=dedup_lim, top=top
        )
        keywords = extractor.extract_keywords(text)
        return [{"keyword": kw, "score": score} for kw, score in keywords]
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []

def filter_entities(doc, options: EntityFilterOptions = None):
    """Filter and deduplicate entities."""
    if options is None:
        options = EntityFilterOptions()
        
    entities = [
        {"type": ent.label_, "text": ent.text}
        for ent in doc.ents
        if ent.label_ not in options.exclude_types and len(ent.text) >= options.min_length
    ]
    
    # Deduplicate entities
    seen = set()
    unique_entities = []
    for entity in entities:
        key = (entity["type"], entity["text"].lower())
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
            
    return unique_entities

async def process_article_data(link: str):
    """Process article data with all NLP tasks"""
    start_time = time.time()
    
    # Fetch article
    article = await fetch_article(link)
    
    # Perform NLP processing
    doc = nlp_components.nlp(article.text)
    
    # Extract entities
    filtered_entities = filter_entities(doc)
    
    # Get social accounts
    social_accounts = socials.extract(link).get_matches_per_platform()
    
    # Get social shares
    try:
        social_shares = socialshares.fetch(link, SOCIAL_PLATFORMS)
    except Exception as e:
        logger.error(f"Error fetching social shares: {str(e)}")
        social_shares = {}
    
    # Sentiment analysis
    sentiment_scores = nlp_components.sentiment_analyzer.polarity_scores(article.text)
    
    # Generate SpaCy HTML visualization
    spacy_html = displacy.render(doc, style="ent", options={"ents": [e["type"] for e in filtered_entities]})
    
    # Extract keywords
    keywords = extract_keywords(article.text, top=5)
    
    # Extract potential accounts
    try:
        accounts = socid_extractor.extract(article.text)
    except Exception as e:
        logger.error(f"Error extracting accounts: {str(e)}")
        accounts = {}
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        "title": article.title,
        "date": article.publish_date,
        "text": article.text,
        "markdown": md(article.article_html, newline_style="BACKSLASH", strip=STRIP_TEXT_RULES, heading_style="ATX"),
        "html": article.article_html,
        "summary": article.summary,
        "keywords": keywords,
        "authors": article.authors,
        "banner": article.top_image,
        "images": list(article.images),
        "entities": filtered_entities,
        "videos": list(article.movies),
        "social_accounts": social_accounts,
        "spacy_html": spacy_html,
        "spacy_markdown": md(spacy_html, newline_style="BACKSLASH", strip=STRIP_TEXT_RULES, heading_style="ATX"),
        "sentiment": sentiment_scores,
        "accounts": accounts,
        "social_shares": social_shares,
        "processing_time": processing_time,
    }

# Helper function to get a dependency that we can override for testing
async def get_entity_filter_options(
    exclude_types: List[str] = Query(default=list(EXCLUDED_ENTITY_TYPES)),
    min_length: int = Query(default=1),
):
    return EntityFilterOptions(exclude_types=exclude_types, min_length=min_length)

# Routes
@app.post(
    "/nlp/article",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Process an article",
    description="Fetch and analyze an article using NLP techniques"
)
async def process_article(
    article: ArticleAction,
    background_tasks: BackgroundTasks,
    filter_options: EntityFilterOptions = Depends(get_entity_filter_options),
):
    try:
        cache_key = get_cache_key(str(article.link))
        
        # Check cache if enabled
        if article.cache and cache_key in article_cache:
            logger.info(f"Using cached data for {article.link}")
            result = article_cache[cache_key]
            # Update the cache in the background
            background_tasks.add_task(update_article_cache, str(article.link), cache_key)
            return {"data": result, "cached": True}
        
        # Process the article
        logger.info(f"Processing article: {article.link}")
        result = await process_article_data(str(article.link))
        
        # Save to cache
        article_cache[cache_key] = result
        
        return {"data": result, "cached": False}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing article: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing article: {str(e)}"
        )

@app.post(
    "/nlp/tags",
    response_model=Dict[str, List[KeywordResponse]],
    status_code=status.HTTP_200_OK,
    summary="Extract tags from text",
    description="Extract keywords and tags from provided text"
)
async def extract_tags(article: SummarizeAction):
    try:
        keywords = extract_keywords(article.text, n=3, top=5)
        return {"data": keywords}
    except Exception as e:
        logger.error(f"Error extracting tags: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting tags: {str(e)}"
        )

@app.post(
    "/nlp/sentiment",
    response_model=Dict[str, SentimentResponse],
    status_code=status.HTTP_200_OK,
    summary="Analyze sentiment",
    description="Analyze sentiment of provided text"
)
async def analyze_sentiment(article: SummarizeAction):
    try:
        sentiment = nlp_components.sentiment_analyzer.polarity_scores(article.text)
        return {"data": sentiment}
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

@app.post(
    "/nlp/entities",
    response_model=Dict[str, List[EntityResponse]],
    status_code=status.HTTP_200_OK,
    summary="Extract entities",
    description="Extract named entities from provided text"
)
async def extract_entities(
    article: SummarizeAction,
    filter_options: EntityFilterOptions = Depends(get_entity_filter_options),
):
    try:
        doc = nlp_components.nlp(article.text)
        entities = filter_entities(doc, filter_options)
        return {"data": entities}
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting entities: {str(e)}"
        )

@app.post(
    "/nlp/summarize",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    summary="Summarize text",
    description="Generate a summary of provided text"
)
async def summarize_text(article: SummarizeAction):
    try:
        # Create a temporary Article object for summarization
        temp_article = Article(url='')
        temp_article.set_text(article.text)
        temp_article.nlp()
        
        summary = temp_article.summary
        
        # Truncate if max_length is specified
        if article.max_length and len(summary) > article.max_length:
            summary = summary[:article.max_length].rsplit(' ', 1)[0] + '...'
            
        return {"data": summary}
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error summarizing text: {str(e)}"
        )

@app.get(
    "/nlp/health",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    summary="API health check",
    description="Check if the NLP API is running correctly"
)
async def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Background task to update article cache
async def update_article_cache(url: str, cache_key: str):
    """Update the cache for a given article URL"""
    try:
        logger.info(f"Updating cache for {url}")
        result = await process_article_data(url)
        article_cache[cache_key] = result
        logger.info(f"Cache updated for {url}")
    except Exception as e:
        logger.error(f"Error updating cache for {url}: {str(e)}")


# If this module is run directly, start the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=1098, reload=True)