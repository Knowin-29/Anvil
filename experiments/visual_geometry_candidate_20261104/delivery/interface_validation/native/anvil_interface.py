"""Native Anvil prompt/output adapter; no scoring code or reference answers."""
import importlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
MODE=json.loads((ROOT/'anvil_representation.json').read_text())['mode']
MODULE=importlib.import_module({'rotation128':'dof_codebook_interface','rotation6d':'dof_codec_interface'}[MODE]) if MODE!='native' else None

def prepare_native_prompt(form,prompt):
    return MODULE.model_prompt(prompt) if form=='dof' and MODULE else prompt

def decode_native_response(form,response,original_prompt):
    # Invalid latent outputs raise an error; no answer is invented.
    return MODULE.decode_response(response,original_prompt) if form=='dof' and MODULE else response
