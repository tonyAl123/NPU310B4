"""Read-only verification and reuse of a failed build; new files use new attempt paths."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def link_file(source, target):
    assert source.is_file() and not source.is_symlink() and not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def copy_onnx_file(source, target, digest):
    """ONNX rejects external tensors with st_nlink > 1, even for trusted reuse."""
    assert source.is_file() and not source.is_symlink() and not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Always create a separate inode. Existing sources may already have hard links
    # from an earlier failed recovery; neither they nor their links are modified.
    with source.open('rb') as reader, target.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
    shutil.copystat(source, target)
    assert target.stat().st_nlink == 1 and not os.path.samefile(source, target)
    assert target.stat().st_size == source.stat().st_size and sha(target) == digest


def check_attempt(shared, staging, build_root, request, package):
    shared = Path(shared).resolve()
    assert shared.parent == (staging / 'builds' / request['job_id'] / 'attempts').resolve()
    previous = read(shared / 'build-manifest.json')
    old_request = read(shared / 'request.json')
    assert previous['status'] == 'failed' and previous['stage'] == 'numeric_validation'
    for key in ('job_id', 'config', 'files', 'validation', 'versions', 'project_commit', 'context',
                'image_size', 'container_image_id', 'model_relative_path', 'precision_mode'):
        assert old_request[key] == request[key], 'Reuse contract changed: ' + key
    local = Path(previous['linux_work_directory']).resolve()
    assert local.parent == (build_root / 'conversions' / request['job_id']).resolve()
    assert local.name == shared.name and local.is_dir()
    for stage in ('adapter_selftest', 'native_reference', 'export_vision', 'export_decoder'):
        record = previous['stages'][stage]
        visited = set()
        while True:
            assert record['status'] == 'passed'
            if 'exit_code' in record:
                assert record['exit_code'] == 0
                break
            assert stage in ('native_reference', 'export_decoder')
            assert record['execution'] == 'reused_after_hash_verification'
            ancestor = Path(record['source_attempt']).resolve()
            assert ancestor.parent == shared.parent and ancestor not in visited
            visited.add(ancestor)
            ancestor_manifest = read(ancestor / 'build-manifest.json')
            ancestor_request = read(ancestor / 'request.json')
            for key in ('files', 'config', 'validation', 'versions', 'project_commit'):
                assert ancestor_request[key] == request[key]
            record = ancestor_manifest['stages'][stage]
    for name in ('native.py', 'contract.py'):
        assert sha(package / name) == sha(shared / 'source' / name), 'Reference semantics changed: ' + name
    def decoder_code(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = {'FixedAttention', 'delta_step', 'Decoder', 'decoder_examples', 'load_model', 'export_graph'}
        nodes = [ast.dump(node, include_attributes=False) for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        assert len(nodes) == len(names)
        return nodes
    assert decoder_code(package / 'adapter.py') == decoder_code(shared / 'source/adapter.py'), 'Decoder implementation changed'
    assert read(local / 'reference/reference-status.json')['status'] == 'passed'
    return shared, local, previous


def reuse_model(old_local, snapshot, request):
    snapshot.mkdir()
    for record in request['files']:
        source = old_local / 'model' / record['name']
        assert source.stat().st_size == record['bytes'] and sha(source) == record['sha256']
        link_file(source, snapshot / record['name'])


def reuse_products(old_shared, old_local, local):
    inventory = []
    for folder in ('reference', 'onnx/vision', 'onnx/decoder'):
        origin = old_local / folder
        assert origin.is_dir() and not origin.is_symlink()
        sources = sorted(p for p in origin.rglob('*') if p.is_file())
        assert sources, folder
        for source in sources:
            relative = source.relative_to(old_local)
            archive = old_shared / relative
            assert not source.is_symlink() and source.resolve().is_relative_to(old_local)
            digest = sha(source)
            assert archive.is_file() and archive.stat().st_size == source.stat().st_size and sha(archive) == digest, str(relative)
            onnx_file = relative.parts[0] == 'onnx'
            if onnx_file:
                copy_onnx_file(source, local / relative, digest)
            else:
                link_file(source, local / relative)
            inventory.append({'path': relative.as_posix(), 'bytes': source.stat().st_size, 'sha256': digest,
                              'reuse_method': 'independent_copy' if onnx_file else 'link_or_copy'})
    return inventory


def reuse_isolation(old_shared, destination, request):
    origin = old_shared / 'diagnostics/native-vision'
    if not (origin / 'validation.json').is_file():
        return None
    report = read(origin / 'validation.json')
    assert report['status'] == 'passed' and report['purpose'] == 'diagnostic_native_vision'
    assert report['eligible_for_atc'] is False and report['policy'] == request['validation']
    assert not report['failures'] and report['cases'] and all(case['passed'] for case in report['cases'])
    assert {case['name'] for case in report['cases']} == {
        'text-zh', 'text-en', 'image-shapes', 'boundary-511', 'boundary-1023', 'boundary-2046'}
    assert {case['fixture'] for case in report['vision']} == {'shapes', 'checker'}
    assert all(case['passed'] for case in report['vision'])
    previous = read(old_shared / 'build-manifest.json')
    assert previous['stages']['isolate_decoder']['status'] == 'passed'
    assert not destination.exists()
    shutil.copytree(origin, destination)
    for source in origin.rglob('*'):
        if source.is_file():
            assert sha(source) == sha(destination / source.relative_to(origin))
    return {'status': 'passed', 'execution': 'reused_after_hash_verification',
            'source_attempt': str(old_shared), 'report_sha256': sha(destination / 'validation.json'),
            'eligible_for_atc': False}
