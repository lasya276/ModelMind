"""
Data Engine - Loader Module
Handles CSV loading, file validation, and extracting basic structural metadata.
"""

from typing import Dict, Any
import io
import pandas as pd


def read_csv_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """
    Reads CSV bytes using Pandas and returns a validated DataFrame.

    Args:
        file_bytes: The raw bytes of the uploaded CSV file.

    Returns:
        pd.DataFrame: Parsed DataFrame.

    Raises:
        ValueError: If file is empty, malformed, or cannot be parsed as CSV.
    """
    # Check if the uploaded file has empty content
    if not file_bytes or len(file_bytes.strip()) == 0:
        raise ValueError("The uploaded CSV file is empty.")

    try:
        # Wrap bytes in a BytesIO stream for Pandas to read
        stream = io.BytesIO(file_bytes)
        df = pd.read_csv(stream)

        # Check if the dataframe contains no columns
        if df.columns.empty:
            raise ValueError("The CSV file does not contain any valid columns.")

        return df

    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV file contains no data.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")
    except UnicodeDecodeError:
        raise ValueError("The file encoding is not supported. Please upload a UTF-8 encoded CSV file.")
    except Exception as e:
        # Catch other unexpected reading errors
        raise ValueError(f"An error occurred while processing the CSV file: {str(e)}")


def load_csv(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Reads CSV bytes using Pandas and returns basic structural information.

    Args:
        file_bytes: The raw bytes of the uploaded CSV file.
        filename: Name of the uploaded file.

    Returns:
        Dict containing filename, row_count, column_count, and columns.

    Raises:
        ValueError: If file is empty, malformed, or cannot be parsed as CSV.
    """
    df = read_csv_dataframe(file_bytes)

    row_count = int(len(df))
    column_count = int(len(df.columns))
    columns = [str(col) for col in df.columns.tolist()]

    return {
        "filename": filename,
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
    }
