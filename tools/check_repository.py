from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from release import check_public
root=Path(__file__).resolve().parents[1]
print(check_public(root))
