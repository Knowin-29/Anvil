from pathlib import Path
import ast,json,hashlib,shutil
D=Path(__file__).resolve().parent.parent/'diagnosis_32gpu'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def copy_inference_interface(output,mode):
    """Keep exact inference functions; omit the codebook fitting dependency."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files=['dof_codec_interface.py','dof_codec_prototype.py'] if mode!='native' else []
    for name in files:shutil.copyfile(D/name,output/name)
    if mode=='rotation128':
        source=(D/'dof_codebook_interface.py').read_text()
        tree=ast.parse(source);lines=source.splitlines(keepends=True);remove=set()
        for node in tree.body:
            if (isinstance(node,ast.ImportFrom) and node.module=='prepare_dof_rotation_codebook') or (
                isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in {'encode_target','training_row'}):
                remove.update(range(node.lineno-1,node.end_lineno))
        (output/'dof_codebook_interface.py').write_text(''.join(line for i,line in enumerate(lines) if i not in remove))
        shutil.copyfile(D/'dof_rotation_codebook.json',output/'dof_rotation_codebook.json')
    save(output/'anvil_representation.json',dict(mode=mode,decoder_required=mode!='native',
        source_inference_files_sha256={name:sha(D/name) for name in files+(['dof_codebook_interface.py','dof_rotation_codebook.json'] if mode=='rotation128' else [])}))
    (output/'anvil_interface.py').write_text('''"""Native Anvil prompt/output adapter; no scoring code or reference answers."""
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
''')
