from pathlib import Path
import importlib.util
p=Path(__file__).resolve().parent.parent/'visual_resume_storage_20261105/storage_budget.py'
s=importlib.util.spec_from_file_location('original_storage_reserve',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
release=m.release
before_checkpoint=m.before_checkpoint
