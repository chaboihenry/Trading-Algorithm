import os

# Quant data root — set QUANT_DATA_DIR env var to relocate vault to an external SSD
# or alternate location. Default keeps it on the local home directory.
QUANT_DATA_DIR = os.environ.get(
    "QUANT_DATA_DIR",
    os.path.expanduser("~/quant_data"),
)
VAULT_ROOT = os.path.join(QUANT_DATA_DIR, "tick_data_storage")
SPLIT_FACTORS_PATH = os.path.join(QUANT_DATA_DIR, "split_factors.parquet")

LOGS_DIR = "logs"
MODELS_DIR = "the_models"
EXECUTION_DATA_DIR = "the_execution_node/data"
RAW_MACRO_CSV = f"{EXECUTION_DATA_DIR}/raw_macro_data.csv"
TRADE_HISTORY_CSV = f"{EXECUTION_DATA_DIR}/trade_history.csv"
CURATED_UNIVERSE_JSON = f"{MODELS_DIR}/curated_universe.json"
META_LABELER_JSON = f"{MODELS_DIR}/meta_labeler_v3.json"
ACTIVE_MODEL_VERSION = f"{MODELS_DIR}/active_model_version.txt"
OPEN_POSITIONS_JSON = f"{LOGS_DIR}/open_positions.json"
DISCOVERY_LEDGER_JSONL = f"{LOGS_DIR}/cluster_discovery_ledger.jsonl"
EXECUTION_LOG = f"{LOGS_DIR}/execution_node.log"
ORCHESTRATOR_LOG = f"{LOGS_DIR}/orchestrator.log"