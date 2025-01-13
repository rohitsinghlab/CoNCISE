import typer
from omegaconf import OmegaConf
import logging
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def get_top_codes_and_smiles(conn: sqlite3.Connection, num_codes_per_protein, num_smiles_per_code) -> pd.DataFrame:
    """
    Get top 10 codes per ID and 20 random SMILES per code.
    
    Args:
        conn: SQLite connection object
    
    Returns:
        DataFrame with columns: ID, Code, Score, SMILES_List
    """
    query = f"""
    WITH TopCodes AS (
      SELECT *
      FROM (
        SELECT 
          ps.ID,
          ps.Code,
          ps.Score,
          ROW_NUMBER() OVER (PARTITION BY ps.ID ORDER BY ps.Score DESC) as rank
        FROM protein_scores ps
        INNER JOIN codes c ON ps.Code = c.Code
      )
      WHERE rank <= {num_codes_per_protein}
    ),
    RandomSmiles AS (
      SELECT 
        t.ID,
        t.Code,
        t.Score,
        c.SMILES
      FROM TopCodes t
      JOIN codes c ON t.Code = c.Code
      WHERE (
        SELECT COUNT(*) 
        FROM codes c2 
        WHERE c2.Code = t.Code 
        AND c2.ROWID <= c.ROWID
      ) <= {num_smiles_per_code}
      AND RANDOM() % (
        SELECT COUNT(*) 
        FROM codes c3 
        WHERE c3.Code = t.Code
      ) < {num_smiles_per_code}
    )
    SELECT 
      ID,
      Code,
      Score,
      GROUP_CONCAT(SMILES, ';') as SMILES_List
    FROM RandomSmiles
    GROUP BY ID, Code, Score
    ORDER BY ID, Score DESC;
    """
    
    try:
        results = pd.read_sql_query(query, conn)
        logger.info(f"Retrieved {len(results)} rows from database")
        return results
    except sqlite3.Error as e:
        logger.error(f"Error executing SQL query: {e}")
        raise

def process_results(df: pd.DataFrame) -> Dict[str, List[Tuple[str, float, List[str]]]]:
    """
    Process the results DataFrame into a more useful format.
    
    Args:
        df: DataFrame with columns ID, Code, Score, SMILES_List
    
    Returns:
        Dictionary mapping ID to list of (Code, Score, SMILES_list) tuples
    """
    results_dict = {}
    
    for _, row in df.iterrows():
        id_val = row['ID']
        code = row['Code']
        score = row['Score']
        smiles_list = row['SMILES_List'].split(';') if pd.notna(row['SMILES_List']) else []
        
        if id_val not in results_dict:
            results_dict[id_val] = []
        
        results_dict[id_val].append((code, score, smiles_list))
    
    return results_dict

def main(config_path: Path = typer.Option(..., help="Path to config file")):
    # Load configuration
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Process codes file
    codes_file = Path(cfg["codes_file"])
    if not codes_file.exists():
        raise FileNotFoundError(f"Codes file not found at {codes_file}")
    
    # Create SQLite database if needed
    if codes_file.suffix != ".sqlite":
        logger.info("Converting codes file to SQLite database")
        if codes_file.suffix != ".csv":
            raise ValueError("Codes file must be CSV or SQLite")
        
        codes = pd.read_csv(codes_file)
        sqlite_file = codes_file.with_suffix(".sqlite")
        conn = sqlite3.connect(sqlite_file)
        codes.to_sql("codes", conn, index=False, if_exists="replace")
        logger.info(f"Created SQLite database at {sqlite_file}")
    else:
        conn = sqlite3.connect(codes_file)

    # Process protein scores
    protein_scores_file = Path(cfg["protein_scores_file"])
    if not protein_scores_file.exists():
        raise FileNotFoundError(f"Protein scores file not found at {protein_scores_file}")
    if protein_scores_file.suffix != ".csv":
        raise ValueError("Protein scores file must be CSV")
    
    protein_scores = pd.read_csv(protein_scores_file)
    required_columns = ["ID", "Score", "Code"]
    if not all(col in protein_scores.columns for col in required_columns):
        raise ValueError(f"Protein scores file must have columns: {required_columns}")
    
    # Add protein scores to database
    protein_scores.to_sql("protein_scores", conn, index=False, if_exists="replace")
    logger.info("Added protein scores to database")


    try:
        results_df = get_top_codes_and_smiles(conn, cfg["num_codes_per_protein"], cfg["num_smiles_per_code"])
        save_path = Path(cfg["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)
        
    finally:
        conn.close()

