"""
FastAPI main application for UK Road Risk Assessment System
"""

import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from datetime import datetime

from api.endpoints import router
from api.models import ErrorResponse
from config.settings import API_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"], 
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting UK Road Risk Assessment API...")
    logger.info(f"Environment: {API_CONFIG}")
    yield
    # Shutdown
    logger.info("Shutting down UK Road Risk Assessment API...")

# Create FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # In production, specify actual hosts
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    
    error_response = ErrorResponse(
        error="Internal Server Error",
        message=str(exc),
        timestamp=datetime.now(),
        request_id=getattr(request.state, 'request_id', None)
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_response = ErrorResponse(
        error=f"HTTP {exc.status_code}",
        message=exc.detail,
        timestamp=datetime.now(),
        request_id=getattr(request.state, 'request_id', None)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Risk Assessment"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "UK Road Risk Assessment API",
        "version": API_CONFIG["version"],
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "timestamp": datetime.now().isoformat()
    }

# Additional utility endpoints
@app.get("/status")
async def status():
    """System status endpoint"""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "version": API_CONFIG["version"],
        "endpoints": {
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "model_performance": "/api/v1/model/performance",
            "data_overview": "/api/v1/data/overview",
            "feature_importance": "/api/v1/features/importance"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"]
    )