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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
async def upload_csv_file(
    file: UploadFile = File(...)
):
    logger.info(f"Received upload request for file: {file.filename if file else 'None'}")

    if not file or not file.filename:
        logger.warning("Upload attempt failed: No file provided.")
        raise HTTPException(status_code=400, detail="File upload is required.")

    if not file.filename.lower().endswith('.csv'):
        logger.warning(f"Upload attempt failed: File '{file.filename}' does not have a .csv extension.")
        raise HTTPException(status_code=400, detail=f"Invalid file type: '{file.filename}'. Only files with a .csv extension are accepted.")

    if file.content_type not in ["text/csv", "application/vnd.ms-excel", "text/plain"]:
         logger.warning(f"File '{file.filename}' has unexpected content type: '{file.content_type}'. Proceeding based on extension.")

    original_filename = file.filename
    table_name = sanitize_table_name(original_filename)
    logger.info(f"Using sanitized table name: '{table_name}' for file '{original_filename}'")

    temp_file_path = f"temp_{uuid.uuid4().hex}.csv"

    try:
        logger.info(f"Saving uploaded file '{original_filename}' temporarily to '{temp_file_path}'")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Temporary file saved.")

        logger.info(f"Reading CSV data from '{temp_file_path}' into pandas DataFrame.")
        df = pd.read_csv(temp_file_path)
        logger.info(f"CSV read. Shape: {df.shape}. Columns: {df.columns.tolist()}")

        if df.empty:
             logger.warning(f"Uploaded CSV file '{original_filename}' is empty.")
             raise HTTPException(status_code=400, detail="The uploaded CSV file is empty or contains only headers.")

        logger.info(f"Loading DataFrame into SQLite table: '{table_name}' (if_exists=replace)")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logger.info(f"Data successfully loaded/replaced into table '{table_name}'.")

        with get_db() as db:
            existing_upload = db.query(FileUpload).filter(FileUpload.table_name == table_name).first()
            if existing_upload:
                logger.info(f"Updating existing metadata for table '{table_name}'")
                existing_upload.filename = original_filename
            else:
                logger.info(f"Creating new metadata entry for table '{table_name}'")
                new_upload = FileUpload(
                    filename=original_filename,
                    table_name=table_name
                )
                db.add(new_upload)
            db.commit()
            logger.info(f"File upload metadata saved/updated: {original_filename} -> {table_name}")

        return UploadResponse(
            message="CSV uploaded and data loaded successfully",
            table_name=table_name,
            original_filename=original_filename,
            columns=df.columns.tolist(),
            rows=len(df)
        )

    except pd.errors.ParserError as e:
         logger.error(f"Failed to parse CSV file '{original_filename}': {e}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Error parsing CSV file '{original_filename}': {e}. Please ensure it's a valid CSV format.")
    except pd.errors.EmptyDataError:
        logger.error(f"Upload failed: The CSV file '{original_filename}' seems empty (pandas EmptyDataError).")
        raise HTTPException(status_code=400, detail="The uploaded CSV file could not be read (possibly empty or invalid structure).")
    except Exception as e:
        logger.error(f"Error processing upload for '{original_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during processing: {str(e)}")

    finally:
        if file and hasattr(file, 'file') and not file.file.closed:
            try:
                 file.file.close()
                 logger.debug(f"Closed file handle for '{original_filename}'.")
            except Exception as e:
                 logger.warning(f"Could not explicitly close file handle for {original_filename}: {e}")

        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Removed temporary file '{temp_file_path}'")
            except OSError as e:
                logger.error(f"Error removing temporary file '{temp_file_path}': {e}")
def get_table_lock(table_name: str) -> threading.RLock:
    with global_lock:
        if table_name not in table_locks:
            table_locks[table_name] = threading.RLock()
        return table_locks[table_name]
@app.get("/tables", response_model=TablesResponse)
async def list_tables():
    logger.info("Request received for listing tables and columns.")
    tables_details = []
    try:
        metadata = MetaData()
        metadata.reflect(bind=engine)
        logger.info("Database schema reflected successfully.")

        with get_db() as db:
            file_uploads = db.query(FileUpload).order_by(FileUpload.id).all()
            logger.info(f"Found {len(file_uploads)} tracked file uploads in metadata.")

            for upload in file_uploads:
                table_name = upload.table_name
                columns = []
                if table_name in metadata.tables:
                    table_obj = metadata.tables[table_name]
                    columns = [col.name for col in table_obj.columns]
                    logger.debug(f"Found columns for table '{table_name}': {columns}")
                else:
                    logger.warning(f"Table '{table_name}' (from file '{upload.filename}') in metadata but not in DB schema.")

                tables_details.append(
                    TableInfo(
                        table_name=table_name,
                        filename=upload.filename,
                        columns=columns
                    )
                )

            logger.info(f"Successfully prepared list of {len(tables_details)} tables with columns.")
            return TablesResponse(tables=tables_details)

    except Exception as e:
        logger.error(f"Error listing tables or reflecting schema: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving table information: {str(e)}")

@app.post("/generate-sql", response_model=SQLQueryResponse)
async def generate_sql_query(request: SQLQueryRequest):
    logger.info(f"Received request to generate SQL for primary table '{request.primary_table_name}' and additional tables {request.additional_tables} from query: '{request.natural_language_query}'")

    if not groq_client:
        logger.error("Attempted /generate-sql but Groq client is not initialized.")
        raise HTTPException(status_code=503, detail="Groq AI service unavailable. API key may be missing or invalid.")

    primary_table = request.primary_table_name
    additional_tables = request.additional_tables
    natural_query = request.natural_language_query
    all_tables = [primary_table] + additional_tables

    table_schemas = {}
    metadata = MetaData()
    metadata.reflect(bind=engine)
    
    for table_name in all_tables:
        if table_name not in metadata.tables:
            logger.warning(f"Table '{table_name}' not found in database schema for SQL generation.")
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")
        
        table_obj = metadata.tables[table_name]
        columns = [col.name for col in table_obj.columns]
        table_schemas[table_name] = columns
        logger.info(f"Successfully retrieved columns for table '{table_name}': {columns}")

    if not table_schemas:
        logger.error(f"No schema information could be retrieved for the specified tables.")
        raise HTTPException(status_code=500, detail=f"Could not determine columns for the specified tables.")

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
    logger.debug(f"Constructed prompt for Groq API with multi-table support.")

    try:
        logger.info("Sending request to Groq API...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.2,
            max_tokens=1024,
        )
        logger.info("Received response from Groq API.")

        generated_sql = chat_completion.choices[0].message.content.strip()

        logger.info(f"Generated SQL: {generated_sql}")

        return SQLQueryResponse(
            sql_query=generated_sql,
            natural_language_query=natural_query,
            tables_used=all_tables
        )

    except APIError as e:
        logger.error(f"Groq API error: {e.status_code} - {e.message}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Groq API error: {e.message}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Groq API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the SQL query: {str(e)}")

@app.post("/execute-sql", response_model=QueryResultsResponse)
async def execute_sql(request: ExecuteSQLRequest):
    sql_query = request.sql_query.strip()
    primary_table_name = request.primary_table_name
    additional_tables = request.additional_tables

    logger.info(f"Received request to execute SQL query on primary table '{primary_table_name}' with additional tables {additional_tables}:\n{sql_query}")

    if not sql_query.lower().startswith("select"):
        logger.warning(f"Rejected execution attempt for non-SELECT query: {sql_query}")
        raise HTTPException(status_code=400, detail="Invalid operation: Only SELECT queries are allowed.")

    results = []
    columns = []
    row_count = 0
    message = "Query executed successfully."

    try:
        with engine.connect() as connection:
            logger.info("Executing SQL query...")
            cursor_result = connection.execute(text(sql_query))

            if cursor_result.returns_rows:
                columns = list(cursor_result.keys())
                fetched_rows = cursor_result.fetchall()
                row_count = len(fetched_rows)
                logger.info(f"Query returned {row_count} rows with columns: {columns}")

                results = [dict(row._mapping) for row in fetched_rows]
                logger.debug(f"First few results (max 5): {results[:5]}")
            else:
                logger.info("Query executed but did not return rows (e.g., UPDATE, INSERT - though blocked by check).")
                message = "Query executed but did not return rows."

    except SQLAlchemyError as e:
        logger.error(f"Database error executing query: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Database error executing query: {e.orig}")
    except Exception as e:
        logger.error(f"Unexpected error executing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    return QueryResultsResponse(
        primary_table_name=primary_table_name,
        additional_tables=additional_tables,
        sql_query=sql_query,
        columns=columns,
        results=results,
        row_count=row_count,
        message=message
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)