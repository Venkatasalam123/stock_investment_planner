"""Script to delete all database runs with unknown universe_name."""

import sys
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from storage.database import delete_unknown_runs

if __name__ == "__main__":
    print("Searching for runs with unknown universe_name...")
    deleted_count = delete_unknown_runs()
    
    if deleted_count > 0:
        print(f"✅ Successfully deleted {deleted_count} run(s) with unknown universe_name.")
    else:
        print("ℹ️ No runs with unknown universe_name found in the database.")

