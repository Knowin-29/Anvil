"""Execute the sealed original trainer with quota-resilient append transport."""
from pathlib import Path
import runpy
from quota_io import install
V=Path(__file__).resolve().parent.parent/'visual_geometry_candidate_20261104'
install(V/'runs')
runpy.run_path(str(V/'train.py'),run_name='__main__')
