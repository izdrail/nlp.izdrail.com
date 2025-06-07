# NLP API Documentation

![](https://i.imgur.com/3otyIWl.png)

## Overview

The NLP API is a comprehensive FastAPI-based service for natural language processing tasks. It provides article analysis, sentiment analysis, entity extraction, keyword extraction, and more, with built-in caching and social media integration.

**Version:** 1.0.0  
**Base URL:** `/api/v1`

## Features

- **Article Processing**: Fetch and analyze web articles with full NLP pipeline
- **Sentiment Analysis**: VADER sentiment analysis with compound, positive, negative, and neutral scores
- **Entity Extraction**: Named Entity Recognition using spaCy with customizable filtering
- **Keyword Extraction**: YAKE-based keyword extraction with configurable parameters
- **Social Media Integration**: Extract social accounts and share counts
- **Caching System**: In-memory caching with background updates for performance
- **Concurrent Processing**: Async/await with background tasks for optimal performance

## Installation & Setup

### Prerequisites

```bash
pip install fastapi uvicorn spacy newspaper3k markdownify vaderSentiment yake socials socid-extractor socialshares
python -m spacy download en_core_web_md
```

### Environment Setup

The API uses several NLP libraries that require additional setup:

1. **spaCy Model**: Download the English medium model
   ```bash
   python -m spacy download en_core_web_md
   ```

2. **User Agent**: The API uses a default user agent for web scraping. Modify `DEFAULT_USER_AGENT` constant if needed.

### Running the API

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 1098 --workers 4
```

## API Endpoints

### 1. Process Article

**POST** `/api/v1/nlp/article`

Fetch and analyze a web article using the complete NLP pipeline.

#### Request Body

```json
{
  "link": "https://example.com/article",
  "cache": true
}
```

#### Parameters

- `link` (HttpUrl, required): URL of the article to analyze
- `cache` (bool, optional, default: true): Whether to use cached results if available

#### Response

```json
{
  "data": {
    "title": "Article Title",
    "date": "2023-12-01T12:00:00",
    "text": "Full article text...",
    "markdown": "# Article Title\n\nArticle content...",
    "html": "<h1>Article Title</h1><p>Content...</p>",
    "summary": "Article summary...",
    "keywords": [
      {
        "keyword": "machine learning",
        "score": 0.123
      }
    ],
    "authors": ["John Doe", "Jane Smith"],
    "banner": "https://example.com/banner.jpg",
    "images": ["https://example.com/image1.jpg"],
    "entities": [
      {
        "type": "PERSON",
        "text": "John Doe"
      }
    ],
    "videos": ["https://example.com/video.mp4"],
    "social_accounts": {},
    "sentiment": {
      "compound": 0.5,
      "positive": 0.6,
      "negative": 0.1,
      "neutral": 0.3
    },
    "accounts": {},
    "social_shares": {
      "facebook": 150,
      "twitter": 75
    },
    "processing_time": 2.34
  },
  "cached": false
}
```

#### Error Responses

- `422`: Unprocessable Entity - Invalid URL or article couldn't be fetched
- `500`: Internal Server Error - Processing failed

### 2. Extract Keywords/Tags

**POST** `/api/v1/nlp/tags`

Extract keywords and tags from provided text using YAKE algorithm.

#### Request Body

```json
{
  "text": "Your text content here...",
  "max_length": 100
}
```

#### Response

```json
{
  "data": [
    {
      "keyword": "natural language processing",
      "score": 0.087
    },
    {
      "keyword": "machine learning",
      "score": 0.123
    }
  ]
}
```

### 3. Sentiment Analysis

**POST** `/api/v1/nlp/sentiment`

Analyze sentiment of provided text using VADER sentiment analyzer.

#### Request Body

```json
{
  "text": "This is a great product! I love it.",
  "max_length": null
}
```

#### Response

```json
{
  "data": {
    "compound": 0.6249,
    "positive": 0.661,
    "negative": 0.0,
    "neutral": 0.339
  }
}
```

#### Sentiment Score Interpretation

- **Compound**: Overall sentiment (-1 to 1, where -1 is most negative, 1 is most positive)
- **Positive**: Proportion of positive sentiment (0 to 1)
- **Negative**: Proportion of negative sentiment (0 to 1)
- **Neutral**: Proportion of neutral sentiment (0 to 1)

### 4. Entity Extraction

**POST** `/api/v1/nlp/entities`

Extract named entities from text using spaCy NER.

#### Request Body

```json
{
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."
}
```

#### Query Parameters

- `exclude_types` (array): Entity types to exclude (default: TIME, DATE, LANGUAGE, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL)
- `min_length` (integer): Minimum length of entity text (default: 1)

#### Response

```json
{
  "data": [
    {
      "type": "ORG",
      "text": "Apple Inc."
    },
    {
      "type": "PERSON",
      "text": "Steve Jobs"
    },
    {
      "type": "GPE",
      "text": "Cupertino"
    },
    {
      "type": "GPE",
      "text": "California"
    }
  ]
}
```

#### Common Entity Types

- **PERSON**: People, including fictional
- **ORG**: Companies, agencies, institutions
- **GPE**: Countries, cities, states
- **LOC**: Non-GPE locations, mountain ranges, bodies of water
- **PRODUCT**: Objects, vehicles, foods, etc.
- **EVENT**: Named hurricanes, battles, wars, sports events
- **WORK_OF_ART**: Titles of books, songs, etc.
- **LAW**: Named documents made into laws
- **LANGUAGE**: Any named language

### 5. Text Summarization

**POST** `/api/v1/nlp/summarize`

Generate a summary of provided text using newspaper3k's summarization.

#### Request Body

```json
{
  "text": "Long article text content...",
  "max_length": 200
}
```

#### Response

```json
{
  "data": "This is the generated summary of the article content..."
}
```

### 6. Get Cached Articles

**GET** `/api/v1/nlp/articles/cached`

Retrieve all cached articles with metadata.

#### Query Parameters

- `limit` (integer, 1-500, default: 50): Maximum number of articles to return
- `offset` (integer, default: 0): Number of articles to skip

#### Response

```json
{
  "total_articles": 25,
  "articles": [
    {
      "cache_key": "abc123def456",
      "cached_at": "2023-12-01T12:00:00",
      "article": {
        "title": "Sample Article",
        "text": "Article content...",
        "processing_time": 1.23
      }
    }
  ]
}
```

### 7. Health Check

**GET** `/api/v1/nlp/health`

Check API health and status.

#### Response

```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2023-12-01T12:00:00"
}
```

## Data Models

### ArticleAction

```python
{
  "link": "https://example.com/article",  # Required HttpUrl
  "cache": true                          # Optional bool, default: true
}
```

### SummarizeAction

```python
{
  "text": "Text to analyze...",  # Required string, min_length: 10
  "max_length": 100          # Optional int
}
```

### EntityFilterOptions

```python
{
  "exclude_types": ["DATE", "TIME"],  # Optional array
  "min_length": 1                     # Optional int, default: 1
}
```

## Caching System

The API implements an intelligent caching system:

### Cache Features

- **In-Memory Storage**: Fast access to processed articles
- **Automatic Cleanup**: Maintains maximum cache size (100 articles)
- **Background Updates**: Refreshes cache entries older than 1 hour
- **Cache Keys**: MD5 hash of article URLs for consistent identification

### Cache Behavior

1. **Cache Hit**: Returns cached data immediately if available and `cache=true`
2. **Background Refresh**: Updates cached data in background if older than 1 hour
3. **Cache Miss**: Processes article and stores result in cache
4. **Size Management**: Automatically removes oldest entries when cache exceeds limit

## Performance Optimization

### Concurrent Processing

The API uses asyncio for concurrent processing of NLP tasks:

- Article fetching
- Entity extraction
- Sentiment analysis
- Keyword extraction
- Social media analysis
- HTML generation

### Lazy Loading

NLP components are loaded on-demand:

- spaCy model loaded on first use
- VADER sentiment analyzer initialized once
- YAKE extractors cached with LRU cache

### Error Handling

Robust error handling with graceful degradation:

- Individual task failures don't stop the entire pipeline
- Default values provided for failed operations
- Comprehensive logging for debugging

## Configuration

### Constants

```python
EXCLUDED_ENTITY_TYPES = {
    "TIME", "DATE", "LANGUAGE", "PERCENT", 
    "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
}
STRIP_TEXT_RULES = ["a"]
DEFAULT_USER_AGENT = "NLP/1.0.0 ..."
SOCIAL_PLATFORMS = [
    "facebook", "pinterest", "linkedin", 
    "reddit", "twitter", "instagram"
]
CACHE_SIZE = 100
API_VERSION = "1.0.0"
```

### CORS Configuration

The API is configured with permissive CORS settings for development. **Modify for production**:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

### Common Error Responses

- **400 Bad Request**: Invalid request format or parameters
- **422 Unprocessable Entity**: Valid request but unable to process (e.g., unreachable URL)
- **500 Internal Server Error**: Server-side processing error

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Logging

The API uses Python's logging module with INFO level by default:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

## Production Considerations

### Security

1. **CORS**: Configure specific allowed origins
2. **Rate Limiting**: Implement rate limiting for production use
3. **Authentication**: Add API key or OAuth authentication
4. **Input Validation**: Already implemented with Pydantic models

### Scalability

1. **Database**: Replace in-memory cache with Redis/database
2. **Workers**: Use multiple uvicorn workers
3. **Load Balancing**: Deploy behind load balancer
4. **Monitoring**: Add health checks and metrics

### Configuration

1. **Environment Variables**: Use environment variables for configuration
2. **Secrets Management**: Secure API keys and credentials
3. **Logging**: Configure structured logging for production

## Usage Examples

### Python Client Example

```python
import requests

# Process article
response = requests.post(
    "http://localhost:1098/api/v1/nlp/article",
    json={
        "link": "https://example.com/article",
        "cache": True
    }
)
article_data = response.json()

# Extract sentiment
response = requests.post(
    "http://localhost:1098/api/v1/nlp/sentiment",
    json={
        "text": "This is an amazing product!"
    }
)
sentiment = response.json()
```

### JavaScript/Fetch Example

```javascript
// Process article
const articleResponse = await fetch('/api/v1/nlp/article', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    link: 'https://example.com/article',
    cache: true
  })
});
const articleData = await articleResponse.json();

// Extract entities
const entitiesResponse = await fetch('/api/v1/nlp/entities', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Apple Inc. was founded by Steve Jobs.'
  })
});
const entities = await entitiesResponse.json();
```

## API Documentation

- **Swagger UI**: Available at `/docs`
- **ReDoc**: Available at `/redoc`
- **OpenAPI Schema**: Available at `/openapi.json`

## Support

For issues, feature requests, or questions about the NLP API, please refer to the logging output for detailed error information and processing statistics.