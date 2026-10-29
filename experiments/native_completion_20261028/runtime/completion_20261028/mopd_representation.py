"""Explicit DoF model representations for the unchanged atomic MOPD objective.

Source records remain independently validated. Derived model messages have a
new contract and cannot impersonate the immutable source serialization. Actual
sampled token IDs are preserved, including malformed generated responses.
"""
from copy import deepcopy
from bisect import bisect_right
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
DIAG = ROOT / 'diagnosis_32gpu'
sys.path.insert(0, str(DIAG))
from trains.flywheel.native_training import validate_training_record, canonical, require
from trains.mopd.anvil_runtime import (AnvilBatchRouter, AnvilTokenPlan,
    ContractV2BatchRouter, structural_ranges)

SCHEMA = 'anvil-derived-dof-model-training-v1'
MODES = {'rotation6d': 'dof_codec_interface', 'rotation128': 'dof_codebook_interface'}


def install_linear_bytelevel_spans():
    """Exact spans for valid UTF-8 ByteLevel streams; conservative fallback.

    The original decoder computes every full prefix, which is quadratic for
    long mask/point responses. This path maps the ACTUAL token bytes to character
    offsets without re-tokenizing. Unsupported decoders, invalid UTF-8 and the
    replacement character use the original implementation unchanged.
    """
    from trains.mopd import anvil_runtime
    if getattr(anvil_runtime.decoded_token_spans, '_anvil_linear_bytelevel', False):
        return
    original=anvil_runtime.decoded_token_spans
    base=list(range(33,127))+list(range(161,173))+list(range(174,256))
    chars=list(base);extra=0
    for byte in range(256):
        if byte not in base:
            base.append(byte);chars.append(256+extra);extra+=1
    byte_map={chr(char):byte for byte,char in zip(base,chars)}

    def decoded(tokenizer,ids):
        try:
            state=json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
            if state.get('type')!='ByteLevel':
                return original(tokenizer,ids)
            pieces=[bytes(byte_map[c] for c in token) for token in tokenizer.convert_ids_to_tokens(ids)]
            raw=b''.join(pieces)
            text=raw.decode('utf-8')
            if '\ufffd' in text or text!=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False):
                return original(tokenizer,ids)
            ends=[];offset=0
            for char in text:
                offset+=len(char.encode('utf-8'));ends.append(offset)
            boundaries=[0];offset=0
            for piece in pieces[:-1]:
                offset+=len(piece);boundaries.append(bisect_right(ends,offset))
            boundaries.append(len(text))
            spans=[(left,right+int(left==right and right<len(text))) for left,right in zip(boundaries,boundaries[1:])]
            return text,spans if ids else []
        except (AttributeError,TypeError,KeyError,UnicodeDecodeError):
            return original(tokenizer,ids)
    decoded._anvil_linear_bytelevel=True
    decoded._anvil_original=original
    anvil_runtime.decoded_token_spans=decoded


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Representation:
    def __init__(self, mode):
        require(mode in {'native', *MODES}, 'Unknown model representation')
        self.mode = mode
        self.module = importlib.import_module(MODES[mode]) if mode != 'native' else None
        names = [] if mode == 'native' else ['dof_codec_prototype.py', 'dof_codec_interface.py']
        if mode == 'rotation128':
            names += ['dof_codebook_interface.py', 'dof_rotation_codebook.json', 'prepare_dof_rotation_codebook.py']
        self.descriptor = dict(mode=mode, version=self.module.VERSION if self.module else 'source-native',
            derived_contract=SCHEMA if self.module else None,
            decoder_required=self.module is not None,
            files_sha256={str(DIAG/n):sha(DIAG/n) for n in names})
        self.digest = hashlib.sha256(canonical(self.descriptor).encode()).hexdigest()

    def row(self, source):
        if self.module is None or source['gsa_form'] != 'dof':
            return source
        validate_training_record(source)
        transformed = self.module.training_row(source)
        row = {k:deepcopy(source[k]) for k in ('id', 'canonical_uid', 'images', 'gsa_form', 'gsa_family')}
        row.update(gsa_contract_version=SCHEMA, messages=transformed['messages'],
            gsa_meta=dict(split='train', admission='explicit_derived_model_representation',
                          binding=deepcopy(source['gsa_meta']['binding']),
                          representation_sha256=self.digest),
            anvil_source_record=deepcopy(source))
        row['gsa_meta']['derived_content_sha256'] = hashlib.sha256(canonical({k:v for k,v in row.items() if k!='gsa_meta'}).encode()).hexdigest()
        return row

    def validate(self, row):
        require(self.module is not None and row.get('gsa_contract_version') == SCHEMA,
                'Derived response must use an explicit represented model')
        source = row.get('anvil_source_record')
        require(isinstance(source, dict), 'Missing immutable native source record')
        expected = self.row(source)
        require(expected == row, 'Derived row or its native source binding changed')
        return source

    def validate_adapters(self, adapters, student):
        result = {}
        for role, adapter in [*adapters.items(), ('student', student)]:
            path = Path(adapter).parent / 'protocol.json'
            protocol = json.loads(path.read_text()) if path.exists() else {}
            declared = protocol.get('model_representation')
            needs_decoder = protocol.get('decode_required_for_native_inference', False)
            if role in ('G','S'):
                require(not needs_decoder, 'G/S adapter cannot silently require a DoF decoder')
            elif role == 'A':
                require(needs_decoder == (self.mode != 'native'), 'A teacher model interface mismatch')
                if needs_decoder:
                    require(declared == self.descriptor['version'], 'A teacher decoder version mismatch')
            elif needs_decoder:
                require(declared == self.descriptor['version'], 'Student decoder version mismatch')
            if needs_decoder:
                recorded=protocol.get('representation',{}).get('files_sha256',{})
                legacy=protocol.get('code_sha256',{})
                for name,digest in self.descriptor['files_sha256'].items():
                    expected=recorded.get(name)
                    if expected is None and isinstance(legacy,dict):
                        expected=legacy.get(Path(name).name)
                    # Older codebook receipts include all train-decoder code;
                    # the inherited context-only helper is already bound by its
                    # versioned importing interface and separately frozen here.
                    required=Path(name).name in {'dof_codec_prototype.py',MODES[self.mode]+'.py','dof_rotation_codebook.json'}
                    if expected is not None or required:
                        require(expected==digest, 'Adapter was trained with different decoder bytes: '+Path(name).name)
            result[role] = dict(protocol=str(path) if path.exists() else None,
                protocol_sha256=sha(path) if path.exists() else None,
                declared_representation=declared,
                student_requires_representation_adaptation=(role=='student' and self.mode!='native' and not needs_decoder))
        return result


class RepresentationRouter:
    def __init__(self, representation):
        self.representation = representation
        self.native = AnvilBatchRouter()

    def plan(self, rows, inputs, tokenizer, *, reference, qualified, reference_prefix=''):
        # Runtime microbatches contain one example; support mixed batches for
        # correctness tests and future batching without mixing identity fields.
        require(len(rows) == inputs['input_ids'].shape[0], 'Batch row mismatch')
        tokens = inputs['input_ids'][:,1:].detach().clone()
        stages = torch.full_like(tokens, -1)
        structural = torch.zeros_like(tokens, dtype=torch.bool)
        selected = torch.full((len(rows),), -1, device=tokens.device, dtype=torch.long)
        decisions = []
        for i, row in enumerate(rows):
            batch = {k:inputs[k][i:i+1] for k in ('input_ids','labels','attention_mask')}
            if row.get('gsa_contract_version') != SCHEMA:
                plan = self.native.plan([row], batch, tokenizer, reference=reference,
                                        qualified=qualified, reference_prefix=reference_prefix)
                require(torch.equal(plan.token_ids[0], tokens[i]), 'Native token identity changed')
                stages[i], structural[i], selected[i] = plan.stage_ids[0], plan.structural_mask[0], plan.selected_stages[0]
                decisions.extend(plan.decisions)
                continue
            self.representation.validate(row)
            text, offsets, positions = ContractV2BatchRouter._response(
                tokenizer, batch, 0, reference_prefix if reference else '')
            if reference:
                require(text == row['messages'][-1]['content'], 'Encoded derived reference changed')
            ranges = structural_ranges(text, coordinate_fields={'latent','rotation_bin','position'}, literal_fields=frozenset())
            for (left,right), position in zip(offsets,positions):
                require(position>0 and bool(batch['attention_mask'][0,position]) and
                        batch['labels'][0,position] == batch['input_ids'][0,position],
                        'Derived labels are not attended next-token targets')
                stages[i,position-1] = 2
                structural[i,position-1] = any(left<end and right>start for start,end in ranges)
            nonempty = bool((stages[i]==2).any())
            if reference:
                require(nonempty, 'Empty represented reference')
            else:
                eligible = nonempty and qualified.get('A') is True
                if eligible:
                    selected[i] = 2
                decisions.append(dict(uid=row['canonical_uid'], contract=SCHEMA,
                    representation_sha256=self.representation.digest,
                    rollout_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    stage='A', state='atomic', supervision_span='atomic_response',
                    mtd_action='apply_mtd' if eligible else 'skip_mtd_retain_sft',
                    reason='declared_represented_native_family' if eligible else 'empty_rollout_or_unqualified_teacher'))
        return AnvilTokenPlan(tokens, stages, selected, structural, decisions)


class ExactCycleStop:
    """Halt only the same conservative sustained cycle used by evaluation.

No replacement token or answer repair. The sampled prefix is retained for MTD
and the stop reason is recorded separately from EOS and budget exhaustion.
"""
    def __init__(self, prompt_length):
        self.prompt_length = prompt_length
        self.cycle = None

    def __call__(self, input_ids, scores, **kwargs):
        from eval_core import repeat_period
        require(len(input_ids)==1, 'Cycle stop currently requires one sampled example')
        generated = input_ids.shape[1]-self.prompt_length
        if generated>=512 and generated%32==0:
            self.cycle = repeat_period(input_ids[0,self.prompt_length:].tolist())
        return torch.tensor([self.cycle is not None], device=input_ids.device, dtype=torch.bool)
