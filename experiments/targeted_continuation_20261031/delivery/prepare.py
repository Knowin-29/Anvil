"""Seal the exact completed MOPD model and evaluator dependencies before inference."""
import argparse
from pathlib import Path
from common import C, D, R, HERE, EXPORT, bound, formal_training, read, require, save, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--training-run', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--native-shards', type=int, default=32)
    ap.add_argument('--public-shards', type=int, default=8)
    args = ap.parse_args()
    require(not args.output.exists(), 'Use a new immutable evaluation spec')
    complete, protocol, checkpoint, receipt = formal_training(args.training_run)
    model = args.model.resolve()
    merged = read(model/'merge_receipt.json')
    require(merged['state'] == 'verified_derived_inference_copy', 'Incomplete merge')
    require(Path(merged['original_adapter']).resolve() == checkpoint, 'Merge uses another checkpoint')
    require(merged['adapter_sha256'] == receipt['files']['adapter_model.safetensors']['sha256'],
            'Merge uses different adapter bytes')
    index = read(model/'model.safetensors.index.json')
    require(set(index['weight_map'].values()) == set(merged['shards']), 'Incomplete model shards')
    for name, info in merged['shards'].items():
        require(sha(model/name) == info['derived_sha256'], 'Changed merged shard: ' + name)
    files = {p.name: sha(p) for p in sorted(model.iterdir()) if p.is_file()}
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json']:
        require(name in files, 'Missing inference asset: ' + name)
    for file, digest in protocol['representation']['files_sha256'].items():
        require(sha(file) == digest, 'Training representation changed: ' + file)
    native = read(EXPORT/'manifest.json')
    public = read(D/'inputs/manifest.json')
    require(native['records'] == 580000 and sum(public['retention']['counts'].values()) == 10105,
            'Unexpected complete evaluation sets')
    required = list(HERE.glob('*.py')) + list((C/'eval_code').rglob('*.py'))
    required += list((D/'code').rglob('*.py'))
    required += [D/n for n in ['dof_codec_interface.py','dof_codec_prototype.py',
                              'dof_codebook_interface.py','dof_rotation_codebook.json',
                              'prepare_dof_rotation_codebook.py']]
    required += [Path(public['retention']['vqa_reference'])]
    required += [D/'retention_node.py', R/'cloud_a800/checkpoints.py']
    spec = dict(schema='anvil-final-evaluation-spec-v1', training_run=str(args.training_run.resolve()),
                checkpoint=str(checkpoint), model=str(model), model_files=files,
                representation=protocol['representation']['mode'],
                native_records=580000, general_records=10105,
                public_records={k:v['n'] for k,v in public['public'].items()},
                native_shards=args.native_shards, public_shards=args.public_shards,
                evidence={key:bound(path) for key,path in {
                    'training_complete':args.training_run/'complete.json',
                    'training_protocol':args.training_run/'protocol.json',
                    'checkpoint':checkpoint/'checkpoint_complete.json',
                    'native_inputs':EXPORT/'manifest.json', 'public_inputs':D/'inputs/manifest.json',
                    'paper_protocol':D/'paper_protocol_reconciliation.json'}.items()},
                frozen_files={str(p.resolve()):sha(p) for p in sorted(set(required))},
                final_quality_accepted=False, weights_uploaded=False,
                protocol_note='Same checkpoint on full native, seven public and general sets. RefSpatial paper comparison uses Location100; full277 is also reported. Historical execution equivalence is not assumed.')
    require(args.native_shards > 0 and args.public_shards > 0, 'Invalid shard counts')
    save(args.output,spec)
    print(args.output)


if __name__ == '__main__':
    main()
