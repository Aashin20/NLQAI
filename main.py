import os
import uuid
import shutil
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any # Added Dict, Any for query results
import re

# --- Database Imports ---
from sqlalchemy import create_engine, Column, Integer, String, MetaData, text # Added text for raw SQL
from sqlalchemy.exc import SQLAlchemyError # Import specific exception
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

# --- FastAPI Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Groq Imports ---
from groq import Groq, APIError

# --- Dotenv Import ---
from dotenv import load_dotenv

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(title="CSV Upload and SQL Execution API") # Updated title

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Setup ---
DATABASE_URL = "sqlite:///./csv_data.db"
# Consider read-only connection for the query endpoint if possible/needed, though difficult with shared engine
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

# --- Database Models ---
class FileUpload(Base):
    __tablename__ = "file_uploads"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    table_name = Column(String, unique=True)

Base.metadata.create_all(bind=engine)

# --- Groq Client Initialization ---
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

# --- Pydantic Models ---
class SQLQueryRequest(BaseModel):
    natural_language_query: str
    table_name: str

class TableInfo(BaseModel):
    table_name: str
    filename: str
    columns: List[str]

class TablesResponse(BaseModel):
    tables: List[TableInfo]

class SQLQueryResponse(BaseModel):
    sql_query: str
    natural_language_query: str
    table_name: str

class UploadResponse(BaseModel):
    message: str
    table_name: str
    original_filename: str
    columns: List[str]
    rows: int

# --- New Models for SQL Execution ---
class ExecuteSQLRequest(BaseModel):
    sql_query: str
    table_name: str # Keep for context/logging, though query might access others

class QueryResultRow(BaseModel):
     # Represents a single row. Using Dict[str, Any] for flexibility.
    row_data: Dict[str, Any]

class QueryResultsResponse(BaseModel):
    table_name: str
    sql_query: str
    columns: List[str]
    results: List[Dict[str, Any]] # Return results as list of dicts
    row_count: int
    message: Optional[str] = None


# --- Utility Functions ---
def sanitize_table_name(filename: str) -> str:
    base_name = os.path.splitext(filename)[0]
    sanitized = base_name.lower()
    sanitized = re.sub(r'[^\w]+', '_', sanitized)
    sanitized = re.sub(r'^[^a-z_]+', '', sanitized)
    if not sanitized or sanitized.startswith('_'):
        sanitized = "tbl_" + sanitized
    if not sanitized:
        sanitized = f"tbl_{uuid.uuid4().hex[:8]}"
    return sanitized

# --- API Endpoints ---

@app.on_event("startup")
async def startup():
    try:
        with get_db():
           logger.info("Database connection verified on startup.")
        if not groq_client:
             logger.warning("Groq client not available on startup.")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)


@app.post("/upload", response_model=UploadResponse, summary="Upload a CSV file")
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
    logger.info(f"Received request to generate SQL for table '{request.table_name}' from query: '{request.natural_language_query}'")

    if not groq_client:
        logger.error("Attempted /generate-sql but Groq client is not initialized.")
        raise HTTPException(status_code=503, detail="Groq AI service unavailable. API key may be missing or invalid.")

    table_name = request.table_name
    natural_query = request.natural_language_query

    columns = []
    try:
        metadata = MetaData()
        metadata.reflect(bind=engine, only=[table_name])
        if table_name not in metadata.tables:
            logger.warning(f"Table '{table_name}' not found in database schema for SQL generation.")
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")

        table_obj = metadata.tables[table_name]
        columns = [col.name for col in table_obj.columns]
        logger.info(f"Successfully retrieved columns for table '{table_name}': {columns}")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error reflecting schema for table '{table_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve schema for table '{table_name}'.")

    if not columns:
         logger.error(f"No columns found for table '{table_name}'.")
         raise HTTPException(status_code=500, detail=f"Could not determine columns for table '{table_name}'.")

    prompt = f"""
    You are an expert SQLite query generator.
    Given the following table schema and a user's request in natural language, generate ONLY the corresponding SQLite SQL query.

    Table Name: {table_name}
    Columns: {', '.join(columns)}

    User's Request: {natural_query}

    Generate only the SQLite SQL query. Do not include any explanation, markdown formatting (like ```sql), or introductory text.
    """
    logger.debug(f"Constructed prompt for Groq.")

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
            table_name=table_name
        )

    except APIError as e:
        logger.error(f"Groq API error: {e.status_code} - {e.message}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Groq API error: {e.message}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Groq API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the SQL query: {str(e)}")


# --- NEW ENDPOINT for SQL Execution ---
@app.post("/execute-sql", response_model=QueryResultsResponse, summary="Execute a SQL query")
async def execute_sql(request: ExecuteSQLRequest):
    """
    Executes a provided SQL query (expected to be SELECT) against the database.

    **Security Warning:** This endpoint executes raw SQL. While intended for
    queries generated by the AI endpoint, ensure the source of the SQL query is trusted.
    Only SELECT statements are permitted.

    - **sql_query**: The SQL query string to execute.
    - **table_name**: The primary table associated with the query (for context).
    """
    sql_query = request.sql_query.strip()
    table_name = request.table_name

    logger.info(f"Received request to execute SQL query on table context '{table_name}':\n{sql_query}")

    # ** Basic Security Check: Only allow SELECT statements **
    # This is NOT foolproof against complex attacks but prevents simple misuse.
    if not sql_query.lower().startswith("select"):
        logger.warning(f"Rejected execution attempt for non-SELECT query: {sql_query}")
        raise HTTPException(status_code=400, detail="Invalid operation: Only SELECT queries are allowed.")

    results = []
    columns = []
    row_count = 0
    message = "Query executed successfully."

    try:
        # Use engine.connect() for executing raw SQL directly
        with engine.connect() as connection:
            logger.info("Executing SQL query...")
            # Use text() for clarity and potential future parameterization benefits
            cursor_result = connection.execute(text(sql_query))

            # Check if the query returns rows (SELECT) or not (e.g., some CTEs might not)
            if cursor_result.returns_rows:
                columns = list(cursor_result.keys()) # Get column names
                fetched_rows = cursor_result.fetchall() # Fetch all rows
                row_count = len(fetched_rows)
                logger.info(f"Query returned {row_count} rows with columns: {columns}")

                # Convert rows (which are SQLAlchemy Row objects) to dictionaries
                results = [dict(row._mapping) for row in fetched_rows]
                logger.debug(f"First few results (max 5): {results[:5]}")
            else:
                logger.info("Query executed but did not return rows (e.g., UPDATE, INSERT - though blocked by check).")
                message = "Query executed but did not return rows."
                # row_count remains 0, columns and results remain empty

    except SQLAlchemyError as e:
        logger.error(f"Database error executing query: {e}", exc_info=True)
        # Provide a generic error message but include specifics in the log
        raise HTTPException(status_code=400, detail=f"Database error executing query: {e.orig}") # Show original error type if available
    except Exception as e:
        logger.error(f"Unexpected error executing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    return QueryResultsResponse(
        table_name=table_name,
        sql_query=sql_query,
        columns=columns,
        results=results,
        row_count=row_count,
        message=message
    )


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)