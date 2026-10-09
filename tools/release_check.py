"""Check a local release candidate and optionally build a code-only ZIP.

No upload, credential access, license grant, or mutation of source files occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '.cache', '__pycache__', '.pytest_cache', 'models', 'data', 'outputs', 'publish', 'dist'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def code_files(root):
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if any(part in EXCLUDED for part in rel.parts):
            continue
        if p.is_symlink():
            raise ValueError('Symlink in code candidate: '+str(rel))
        if p.is_file() and p.suffix not in ('.pyc','.log'):
            yield p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',action='store_true')
    a=parser.parse_args()
    manifest=json.loads((ROOT/'release_manifest.json').read_text())
    problems=[];checked=[]
    for row in manifest['artifacts']:
        p=ROOT/'publish'/('model' if row['repo_type']=='model' else 'data')/row['path']
        if not p.is_file() or p.is_symlink() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
            problems.append('Artifact missing or changed: '+row['path'])
        else:checked.append(row['path'])
    files=list(code_files(ROOT))
    # Only matched filenames are reported, never a candidate secret value.
    sensitive=re.compile(r'hf_[A-Za-z0-9]{25,}|glpat-[A-Za-z0-9_-]{15,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')
    private_hosts=re.compile(r'git[.]xmov[.]ai|/mnt/nas/|/home/xmov/|192[.]168[.]\d+[.]\d+')
    for p in files:
        if p.suffix in ('.pt','.safetensors'):
            problems.append('Weight in code candidate: '+str(p.relative_to(ROOT)))
        if p.suffix in ('.py','.md','.json','.sh','.txt') or p.name.startswith('.'):
            text=p.read_text(encoding='utf-8')
            if sensitive.search(text) or (p.name!='release_check.py' and private_hosts.search(text)):
                problems.append('Potential private information: '+str(p.relative_to(ROOT)))
    if problems:
        raise SystemExit('\n'.join(problems))
    report={'artifact_files_verified':len(checked),'artifact_bytes':sum(x['bytes'] for x in manifest['artifacts']),
            'code_files':len(files),'code_bytes':sum(p.stat().st_size for p in files),
            'public_release_approved':manifest['public_release_approved'],
            'note':'Integrity checks passed; publication permission is a separate owner decision.'}
    if a.archive:
        target=ROOT/'dist/trace_open_source_code_candidate.zip';target.parent.mkdir(exist_ok=True)
        if target.exists():raise FileExistsError('Archive exists; preserve it or use a new release folder.')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,'TRACE/'+str(p.relative_to(ROOT)))
        report.update(archive=str(target.relative_to(ROOT)),archive_bytes=target.stat().st_size,archive_sha256=digest(target))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
