"""Build an isolated release candidate from an authorized local engineering tree.

This utility only copies an explicit allowlist. It never uploads or licenses files.
Run once into a new candidate directory; existing destination files are refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_file(source, destination):
    if source.is_symlink() or destination.exists():
        raise ValueError(f'Refusing symlink or existing destination: {destination.name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--gmr-license', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    files = ['run.py', 'run.sh', 'render.py', 'bootstrap_env.py', 'requirements.txt',
             'requirements-render.txt', 'requirements-lock-cu128.txt']
    files += [str(p.relative_to(source)) for p in sorted((source/'trace_core').glob('*.py'))]
    files += [str(p.relative_to(source)) for p in sorted((source/'assets/unitree_g1').rglob('*')) if p.is_file()]
    files += ['bootstrap/pip-26.2.1-py3-none-any.whl']
    files += ['tests/'+n for n in ['test_pipeline.py', 'test_render.py', 'test_bootstrap.py',
        'test_foot_labels.py', 'test_contact_window.py', 'test_contact_multipoint.py']]
    for name in files:
        copy_file(source/name, ROOT/name)
    copy_file(args.gmr_license, ROOT/'LICENSES/GMR-MIT.txt')
    artifacts = []
    for repo_type, names in [
        ('model', ['generator.pt', 'generator_boundary.pt', 'planner_adapter/adapter_config.json',
                   'planner_adapter/adapter_model.safetensors']),
        ('dataset', ['generation_stats.npz', 'standing_pose35.npy']),
    ]:
        folder = ROOT/'publish'/('model' if repo_type == 'model' else 'data')
        for name in names:
            copy_file(source/'models'/name, folder/name)
            artifacts.append(dict(repo_type=repo_type, path=name, destination='models/'+name,
                                  bytes=(folder/name).stat().st_size, sha256=sha256(folder/name)))
    name='example_prompts.jsonl'
    prompts=ROOT/'publish/data'/name
    artifacts.append(dict(repo_type='dataset',path=name,destination='data/'+name,
                          bytes=prompts.stat().st_size,sha256=sha256(prompts)))
    manifest = dict(schema_version=1, public_release_approved=False,
        approval_notes='Owner license decision and training-data/model release permission must be confirmed.',
        artifacts=artifacts, upstream=[
            dict(repo_id='Qwen/Qwen3-4B-Instruct-2507',destination='models/planner_base',revision='master'),
            dict(repo_id='Qwen/Qwen3-Embedding-4B',destination='models/text_embedding',revision='master')])
    # Only hashes and portable relative names enter the public candidate.
    (ROOT/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    provenance={name:sha256(source/name) for name in files}
    (ROOT/'SOURCE_HASHES.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'runtime_files':len(files),'artifact_bytes':sum(x['bytes'] for x in artifacts),
                      'public_release_approved':False}))


if __name__ == '__main__':
    main()
