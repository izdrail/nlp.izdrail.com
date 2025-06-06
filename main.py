from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Depends, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, HttpUrl, ConfigDict
from typing import List, Dict, Any, Optional, Union, Annotated
import asyncio
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime
import hashlib
import re
import html
from functools import lru_cache

# NLP Libraries
import spacy
from newspaper import Article, Config
from markdownify import markdownify as md
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake
import socials
import socid_extractor
import socialshares
from spacy import displacy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nlp_api")

# Constants
EXCLUDED_ENTITY_TYPES = {"TIME", "DATE", "LANGUAGE", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
STRIP_TEXT_RULES = ["a"]
DEFAULT_USER_AGENT = "NLP/1.0.0 (Unix; Intel) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
SOCIAL_PLATFORMS = ["facebook", "pinterest", "linkedin", "reddit", "twitter", "instagram"]
CACHE_SIZE = 100  # Number of articles to keep in cache
API_VERSION = "1.0.0"

# Cache for article processing
article_cache = {}

# Models
class KeywordResponse(BaseModel):
    keyword: str
    score: float

class EntityResponse(BaseModel):
    type: str
    text: str
    
class SentimentResponse(BaseModel):
    compound: float
    positive: float = Field(alias="pos")
    negative: float = Field(alias="neg") 
    neutral: float = Field(alias="neu")
    
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        json_schema_extra={
            "example": {
                "compound": 0.5,
                "positive": 0.6,
                "negative": 0.1,
                "neutral": 0.3
            }
        }
    )
    
class ArticleResponse(BaseModel):
    title: str
    date: Optional[datetime] = None
    text: str
    markdown: str
    html: str
    summary: str
    keywords: List[KeywordResponse]
    authors: List[str]
    banner: Optional[str] = None
    images: List[str]
    entities: List[EntityResponse]
    videos: List[str]
    social_accounts: Dict[str, Any]
    sentiment: SentimentResponse
    accounts: Dict[str, Any]
    social_shares: Dict[str, Any]
    processing_time: float
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Sample Article",
                "text": "This is the article text",
                "markdown": "# Sample Article\n\nThis is the article text",
                "html": "<h1>Sample Article</h1><p>This is the article text</p>",
                "summary": "A summary of the article",
                "keywords": [{"keyword": "sample", "score": 0.1}],
                "authors": ["John Doe"],
                "banner": "https://example.com/banner.jpg",
                "images": ["https://example.com/image1.jpg"],
                "entities": [{"type": "PERSON", "text": "John Doe"}],
                "videos": ["https://example.com/video.mp4"],
                "social_accounts": {},
                "sentiment": {"compound": 0.5, "positive": 0.6, "negative": 0.1, "neutral": 0.3},
                "accounts": {},
                "social_shares": {},
                "processing_time": 0.5
            }
        }
    )

class CachedArticleResponse(BaseModel):
    cache_key: str
    cached_at: datetime
    article: ArticleResponse
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cache_key": "abc123def456",
                "cached_at": "2023-12-01T12:00:00",
                "article": {
                    "title": "Sample Article",
                    "text": "This is the article text",
                    "processing_time": 0.5
                }
            }
        }
    )

class CachedArticlesListResponse(BaseModel):
    total_articles: int
    articles: List[CachedArticleResponse]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_articles": 10,
                "articles": [
                    {
                        "cache_key": "abc123def456",
                        "cached_at": "2023-12-01T12:00:00",
                        "article": {"title": "Sample Article"}
                    }
                ]
            }
        }
    )

class ArticleAction(BaseModel):
    link: HttpUrl = Field(..., description="URL of the article to analyze")
    cache: bool = Field(True, description="Whether to use cached results if available")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "link": "https://example.com/article",
                "cache": True
            }
        }
    )

class SummarizeAction(BaseModel):
    text: str = Field(..., min_length=10, description="Text to summarize or extract tags from")
    max_length: Optional[int] = Field(None, description="Maximum length of summary")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "This is a sample text for extracting keywords and analyzing sentiment.",
                "max_length": 100
            }
        }
    )

class EntityFilterOptions(BaseModel):
    exclude_types: List[str] = Field(
        default_factory=lambda: list(EXCLUDED_ENTITY_TYPES),
        description="Entity types to exclude"
    )
    min_length: int = Field(1, description="Minimum length of entity text")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "exclude_types": ["DATE", "TIME"],
                "min_length": 2
            }
        }
    )

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "timestamp": "2023-01-01T12:00:00"
            }
        }
    )

# NLP Components with improved lazy loading
class NLPComponents:
    _instance = None
    _nlp = None
    _sentiment_analyzer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLPComponents, cls).__new__(cls)
        return cls._instance
    
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
    
    @lru_cache(maxsize=8)
    def get_keyword_extractor(self, language="en", n=1, dedup_lim=0.9, top=5):
        """Get a configured YAKE keyword extractor with caching"""
        logger.info(f"Creating keyword extractor: lang={language}, n={n}, dedup={dedup_lim}, top={top}")
        return yake.KeywordExtractor(lan=language, n=n, dedupLim=dedup_lim, top=top)


# Lifespan manager for application startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize NLP components on startup
    logger.info("Initializing NLP components...")
    components = NLPComponents()
    # Pre-load models to avoid lazy loading during first request
    _ = components.nlp
    _ = components.sentiment_analyzer
    logger.info("NLP API started successfully")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down NLP API")
    # Clear cache
    article_cache.clear()


# Initialize FastAPI app with lifespan manager
app = FastAPI(
    title="NLP API",
    description="API for Natural Language Processing tasks including article analysis, sentiment analysis, entity extraction, and more.",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the router
router = APIRouter(
    prefix="/api/v1",
    tags=["nlp"],
)

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
        nlp_components = NLPComponents()
        extractor = nlp_components.get_keyword_extractor(
            language=language, n=n, dedup_lim=dedup_lim, top=top
        )
        keywords = extractor.extract_keywords(text)
        return [KeywordResponse(keyword=kw, score=score) for kw, score in keywords]
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []

def filter_entities(doc, options: EntityFilterOptions = None):
    """Filter and deduplicate entities."""
    if options is None:
        options = EntityFilterOptions()
        
    entities = [
        EntityResponse(type=ent.label_, text=ent.text)
        for ent in doc.ents
        if ent.label_ not in options.exclude_types and len(ent.text) >= options.min_length
    ]
    
    # Deduplicate entities
    seen = set()
    unique_entities = []
    for entity in entities:
        key = (entity.type, entity.text.lower())
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
            
    return unique_entities

async def process_article_data(link: str):
    """Process article data with all NLP tasks"""
    start_time = time.time()
    
    # Fetch article
    article = await fetch_article(link)
    
    # Get NLP components
    nlp_components = NLPComponents()
    
    # Process text in chunks if too large
    text = article.text
    doc = nlp_components.nlp(text[:500000]) if len(text) > 500000 else nlp_components.nlp(text)
    
    # Process all tasks concurrently
    tasks = []
    
    # Extract entities
    entities_task = asyncio.create_task(asyncio.to_thread(
        filter_entities, doc, EntityFilterOptions()
    ))
    tasks.append(entities_task)
    
    # Get social accounts
    social_accounts_task = asyncio.create_task(asyncio.to_thread(
        lambda: socials.extract(link).get_matches_per_platform()
    ))
    tasks.append(social_accounts_task)
    
    # Get social shares
    social_shares_task = asyncio.create_task(asyncio.to_thread(
        lambda: socialshares.fetch(link, SOCIAL_PLATFORMS)
    ))
    tasks.append(social_shares_task)
    
    # Sentiment analysis
    sentiment_task = asyncio.create_task(asyncio.to_thread(
        lambda: nlp_components.sentiment_analyzer.polarity_scores(text)
    ))
    tasks.append(sentiment_task)
    
    # Generate SpaCy HTML visualization
    spacy_html_task = asyncio.create_task(asyncio.to_thread(
        lambda: displacy.render(doc, style="ent")
    ))
    tasks.append(spacy_html_task)
    
    # Extract keywords
    keywords_task = asyncio.create_task(asyncio.to_thread(
        lambda: extract_keywords(text, top=5)
    ))
    tasks.append(keywords_task)
    
    # Extract potential accounts
    accounts_task = asyncio.create_task(asyncio.to_thread(
        lambda: socid_extractor.extract(text)
    ))
    tasks.append(accounts_task)
    
    try:
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results with error handling
        filtered_entities = results[0] if not isinstance(results[0], Exception) else []
        social_accounts = results[1] if not isinstance(results[1], Exception) else {}
        social_shares = results[2] if not isinstance(results[2], Exception) else {}
        sentiment_scores = results[3] if not isinstance(results[3], Exception) else {"compound": 0, "pos": 0, "neg": 0, "neu": 0}
        spacy_html = results[4] if not isinstance(results[4], Exception) else ""
        keywords = results[5] if not isinstance(results[5], Exception) else []
        accounts = results[6] if not isinstance(results[6], Exception) else {}
        
        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
        
    except Exception as e:
        logger.error(f"Error in concurrent processing: {str(e)}")
        # Provide default values for failed tasks
        filtered_entities = []
        social_accounts = {}
        social_shares = {}
        sentiment_scores = {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
        spacy_html = ""
        keywords = []
        accounts = {}
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return ArticleResponse(
        title=article.title,
        date=article.publish_date,
        text=article.text,
        markdown=md(article.article_html, newline_style="BACKSLASH", strip=STRIP_TEXT_RULES, heading_style="ATX"),
        html=article.article_html,
        summary=article.summary,
        keywords=keywords,
        authors=article.authors,
        banner=article.top_image,
        images=list(article.images),
        entities=filtered_entities,
        videos=list(article.movies),
        social_accounts=social_accounts,
        sentiment=SentimentResponse(**sentiment_scores),
        accounts=accounts,
        social_shares=social_shares,
        processing_time=processing_time,
    )

# Helper function to get entity filter options as a dependency
async def get_entity_filter_options(
    exclude_types: List[str] = Query(default=list(EXCLUDED_ENTITY_TYPES)),
    min_length: int = Query(default=1),
):
    return EntityFilterOptions(exclude_types=exclude_types, min_length=min_length)

# Cache management
def manage_cache_size():
    """Ensure cache doesn't exceed maximum size"""
    if len(article_cache) > CACHE_SIZE:
        # Remove oldest items
        keys_to_remove = sorted(article_cache.keys(), key=lambda k: article_cache[k].get("timestamp", 0))[:len(article_cache) - CACHE_SIZE]
        for key in keys_to_remove:
            del article_cache[key]
        logger.info(f"Cache cleaned up, removed {len(keys_to_remove)} items")

# Background task to update article cache
async def update_article_cache(url: str, cache_key: str):
    """Update the cache for a given article URL"""
    try:
        logger.info(f"Updating cache for {url}")
        result = await process_article_data(url)
        article_cache[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        manage_cache_size()
        logger.info(f"Cache updated for {url}")
    except Exception as e:
        logger.error(f"Error updating cache for {url}: {str(e)}")

# Routes
@app.post(
    "/api/v1/nlp/article",
    response_model=Dict[str, Union[ArticleResponse, bool]],
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
            cached_item = article_cache[cache_key]
            result = cached_item.get("data")
            
            # Update the cache in the background if older than 1 hour
            if time.time() - cached_item.get("timestamp", 0) > 3600:
                background_tasks.add_task(update_article_cache, str(article.link), cache_key)
                
            return {"data": result, "cached": True}
        
        # Process the article
        logger.info(f"Processing article: {article.link}")
        result = await process_article_data(str(article.link))
        
        # Save to cache
        article_cache[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        manage_cache_size()
        
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

@app.get(
    "/api/v1/nlp/articles/cached",
    response_model=CachedArticlesListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get cached articles",
    description="Retrieve all cached articles ordered by creation date (most recent first)"
)
async def get_cached_articles(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of articles to return"),
    offset: int = Query(default=0, ge=0, description="Number of articles to skip")
):
    try:
        # Convert cache to list of cached articles with metadata
        cached_articles = []
        
        for cache_key, cached_item in article_cache.items():
            article_data = cached_item.get("data")
            timestamp = cached_item.get("timestamp", 0)
            
            if article_data:
                # Use article date if available, otherwise use cache timestamp
                creation_date = article_data.date if article_data.date else datetime.fromtimestamp(timestamp)
                
                cached_articles.append({
                    "cache_key": cache_key,
                    "cached_at": datetime.fromtimestamp(timestamp),
                    "creation_date": creation_date,
                    "article": article_data
                })
        
        # Sort by creation date (article date or cache timestamp) in descending order
        cached_articles.sort(key=lambda x: x["creation_date"] or x["cached_at"], reverse=True)
        
        # Apply pagination
        total_articles = len(cached_articles)
        paginated_articles = cached_articles[offset:offset + limit]
        
        # Format response
        response_articles = [
            CachedArticleResponse(
                cache_key=item["cache_key"],
                cached_at=item["cached_at"],
                article=item["article"]
            )
            for item in paginated_articles
        ]
        
        logger.info(f"Retrieved {len(response_articles)} cached articles (total: {total_articles})")
        
        return CachedArticlesListResponse(
            total_articles=total_articles,
            articles=response_articles
        )
        
    except Exception as e:
        logger.error(f"Error retrieving cached articles: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving cached articles: {str(e)}"
        )

@app.post(
    "/api/v1/nlp/tags",
    response_model=Dict[str, List[KeywordResponse]],
    status_code=status.HTTP_200_OK,
    summary="Extract tags from text",
    description="Extract keywords and tags from provided text"
)
async def extract_tags(article: SummarizeAction):
    try:
        keywords = await asyncio.to_thread(
            extract_keywords, article.text, n=3, top=5
        )
        return {"data": keywords}
    except Exception as e:
        logger.error(f"Error extracting tags: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting tags: {str(e)}"
        )

@app.post(
    "/api/v1/nlp/sentiment",
    response_model=Dict[str, SentimentResponse],
    status_code=status.HTTP_200_OK,
    summary="Analyze sentiment",
    description="Analyze sentiment of provided text"
)
async def analyze_sentiment(article: SummarizeAction):
    try:
        nlp_components = NLPComponents()
        sentiment = await asyncio.to_thread(
            nlp_components.sentiment_analyzer.polarity_scores, article.text
        )
        # VADER returns {'neg': 0.1, 'neu': 0.2, 'pos': 0.7, 'compound': 0.5}
        # Our model expects field names: negative, neutral, positive, compound
        return {"data": SentimentResponse(**sentiment)}
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

@app.post(
    "/api/v1/nlp/entities",
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
        nlp_components = NLPComponents()
        
        # Process text in chunks if too large
        if len(article.text) > 500000:
            doc = await asyncio.to_thread(nlp_components.nlp, article.text[:500000])
            logger.warning(f"Text too large ({len(article.text)} chars), truncated to 500K chars")
        else:
            doc = await asyncio.to_thread(nlp_components.nlp, article.text)
            
        entities = await asyncio.to_thread(filter_entities, doc, filter_options)
        return {"data": entities}
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting entities: {str(e)}"
        )

@app.post(
    "/api/v1/nlp/summarize",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    summary="Summarize text",
    description="Generate a summary of provided text"
)
async def summarize_text(article: SummarizeAction):
    try:
        async def _summarize():
            # Create a temporary Article object for summarization
            temp_article = Article(url='')
            temp_article.set_text(article.text)
            temp_article.nlp()
            return temp_article.summary
            
        summary = await asyncio.to_thread(_summarize)
        
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
    "/api/v1/nlp/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="API health check",
    description="Check if the NLP API is running correctly"
)
async def health_check():
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        timestamp=datetime.now().isoformat()
    )

# Mount the router to the FastAPI app
app.include_router(router)

# If this module is run directly, start the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=1098, reload=True)