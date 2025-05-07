import os
import uuid
import shutil
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import re
from sqlalchemy import create_engine, Column, Integer, String, MetaData, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq, APIError
from dotenv import load_dotenv
import threading

global_lock = threading.Lock()
table_locks: Dict[str, threading.RLock] = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="CSV Query Assistant with Multi-Table Support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "sqlite:///./csv_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class FileUpload(Base):
    __tablename__ = "file_uploads"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    table_name = Column(String, unique=True)

Base.metadata.create_all(bind=engine)

try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not found. /generate-sql endpoint inactive.")
        groq_client = None
    else:
        groq_client = Groq(api_key=groq_api_key)
        logger.info("Groq client initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)
    groq_client = None

class SQLQueryRequest(BaseModel):
    natural_language_query: str
    primary_table_name: str
    additional_tables: List[str] = []

class TableInfo(BaseModel):
    table_name: str
    filename: str
    columns: List[str]

class TablesResponse(BaseModel):
    tables: List[TableInfo]

class SQLQueryResponse(BaseModel):
    sql_query: str
    natural_language_query: str
    tables_used: List[str]

class UploadResponse(BaseModel):
    message: str
    table_name: str
    original_filename: str
    columns: List[str]
    rows: int

class ExecuteSQLRequest(BaseModel):
    sql_query: str
    primary_table_name: str
    additional_tables: List[str] = []

class QueryResultRow(BaseModel):
    row_data: Dict[str, Any]

class QueryResultsResponse(BaseModel):
    primary_table_name: str
    additional_tables: List[str] = []
    sql_query: str
    columns: List[str]
    results: List[Dict[str, Any]]
    row_count: int
    message: Optional[str] = None

def sanitize_table_name(filename: str) -> str:
    base_name = os.path.splitext(filename)[0]
    sanitized = base_name.lower()
    sanitized = re.sub(r'[^\w]+', '_', sanitized)
    sanitized = re.sub(r'^[^a-z]+', '', sanitized)
    if not sanitized or sanitized.startswith('_'):
        sanitized = "tbl" + sanitized
    if not sanitized:
        sanitized = f"tbl_{uuid.uuid4().hex[:8]}"
    return sanitized

def get_table_schema(table_name: str, metadata=None):
    if metadata is None:
        metadata = MetaData()
        metadata.reflect(bind=engine, only=[table_name])
    if table_name not in metadata.tables:
        return []
    table_obj = metadata.tables[table_name]
    return [col.name for col in table_obj.columns]

def get_all_table_schemas():
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table_schemas = {}
    for table_name in metadata.tables:
        table_obj = metadata.tables[table_name]
        table_schemas[table_name] = [col.name for col in table_obj.columns]
    return table_schemas

def get_table_lock(table_name: str) -> threading.RLock:
    with global_lock:
        if table_name not in table_locks:
            table_locks[table_name] = threading.RLock()
        return table_locks[table_name]

@app.on_event("startup")
async def startup():
    try:
        with get_db() as db:
            logger.info("Database connection verified on startup.")
        if not groq_client:
            logger.warning("Groq client not available on startup.")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)

@app.post("/upload", response_model=UploadResponse)
async def upload_csv_file(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="File upload is required.")
    
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail=f"Invalid file type: '{file.filename}'. Only files with a .csv extension are accepted.")

    original_filename = file.filename
    table_name = sanitize_table_name(original_filename)
    temp_file_path = f"temp_{uuid.uuid4().hex}.csv"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = pd.read_csv(temp_file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty or contains only headers.")

        df.to_sql(table_name, engine, if_exists='replace', index=False)

        with get_db() as db:
            existing_upload = db.query(FileUpload).filter(FileUpload.table_name == table_name).first()
            if existing_upload:
                existing_upload.filename = original_filename
            else:
                new_upload = FileUpload(filename=original_filename, table_name=table_name)
                db.add(new_upload)
            db.commit()

        return UploadResponse(
            message="CSV uploaded and data loaded successfully",
            table_name=table_name,
            original_filename=original_filename,
            columns=df.columns.tolist(),
            rows=len(df)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/tables", response_model=TablesResponse)
async def list_tables():
    tables_details = []
    try:
        metadata = MetaData()
        metadata.reflect(bind=engine)

        with get_db() as db:
            file_uploads = db.query(FileUpload).order_by(FileUpload.id).all()
            for upload in file_uploads:
                table_name = upload.table_name
                columns = []
                if table_name in metadata.tables:
                    table_obj = metadata.tables[table_name]
                    columns = [col.name for col in table_obj.columns]
                tables_details.append(TableInfo(
                    table_name=table_name,
                    filename=upload.filename,
                    columns=columns
                ))
            return TablesResponse(tables=tables_details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-sql", response_model=SQLQueryResponse)
async def generate_sql_query(request: SQLQueryRequest):
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq AI service unavailable.")

    primary_table = request.primary_table_name
    additional_tables = request.additional_tables
    natural_query = request.natural_language_query
    all_tables = [primary_table] + additional_tables

    table_schemas = {}
    metadata = MetaData()
    metadata.reflect(bind=engine)
    
    for table_name in all_tables:
        if table_name not in metadata.tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")
        
        table_obj = metadata.tables[table_name]
        columns = [col.name for col in table_obj.columns]
        table_schemas[table_name] = columns

    table_schemas_for_prompt = []
    for table_name, columns in table_schemas.items():
        table_schemas_for_prompt.append(f"Table Name: {table_name}\nColumns: {', '.join(columns)}")
    
    prompt = f"""
You are an expert SQLite query generator.
Given the following table schemas and a user's request in natural language, generate ONLY the corresponding SQLite SQL query.

{"\n\n".join(table_schemas_for_prompt)}

User's Request: {natural_query}

Generate only the SQLite SQL query. Include proper JOIN statements if needed to query across multiple tables. Look for common column names across tables that could be used for joins. Do not include any explanation, markdown formatting (like ```sql), or introductory text.
"""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.2,
            max_tokens=1024,
        )

        generated_sql = chat_completion.choices[0].message.content.strip()

        return SQLQueryResponse(
            sql_query=generated_sql,
            natural_language_query=natural_query,
            tables_used=all_tables
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute-sql", response_model=QueryResultsResponse)
async def execute_sql(request: ExecuteSQLRequest):
    sql_query = request.sql_query.strip()
    if not sql_query.lower().startswith("select"):
        raise HTTPException(status_code=400, detail="Invalid operation: Only SELECT queries are allowed.")

    try:
        with engine.connect() as connection:
            cursor_result = connection.execute(text(sql_query))
            columns = list(cursor_result.keys())
            results = [dict(row._mapping) for row in cursor_result.fetchall()]
            row_count = len(results)

        return QueryResultsResponse(
            primary_table_name=request.primary_table_name,
            additional_tables=request.additional_tables,
            sql_query=sql_query,
            columns=columns,
            results=results,
            row_count=row_count,
            message="Query executed successfully."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)