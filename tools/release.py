"""Review, verify, and explicitly publish a prepared release kit.

plan/check/verify are offline. Only publish-* with --upload contacts services.
No authentication secrets are accepted as command-line arguments or written.
"""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def under(root, path):
    value = path.resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError('Path is outside its release root')
    return value


def repo_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe repository path: ' + value)
    return path.as_posix()


def valid_repo(owner, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', owner or ''):
        raise ValueError('Fill in a valid account/organization namespace in release-config.json')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name or ''):
        raise ValueError('Fill in a valid repository name')
    return owner + '/' + name


SECRET_PATTERNS = {
    'private Windows home': re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+(?!PUBLIC|USER|<)', re.I),
    'private WSL mount': re.compile(r'/mnt/[a-z]/Users/(?!PUBLIC|USER|<)', re.I),
    'private Linux home': re.compile(r'/home/(?!USER|user|<|\$)[A-Za-z0-9_.-]+/'),
    'access token': re.compile(r'\b(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,})\b'),
    'SSH private key': re.compile(r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----'),
    'embedded SSH password': re.compile(r'sshpass\s+-p\s+'),
}
FORBIDDEN = {'.om', '.onnx', '.safetensors', '.pt', '.pth', '.vhdx', '.run', '.rpm', '.zip', '.tar', '.gz', '.data', '.bin'}


def check_public(root, large_allowed=False):
    count = size = 0
    errors = []
    for path in root.rglob('*'):
        if '.git' in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            errors.append('Symlink: ' + str(path.relative_to(root)))
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        count += 1
        size += path.stat().st_size
        if path.name in {'auth.json', '.env', 'local-sources.json', 'release-config.json'}:
            errors.append('Private file: ' + relative)
        if not large_allowed and (path.suffix.lower() in FORBIDDEN or path.stat().st_size > 25 * 1024**2):
            errors.append('Model/binary payload is forbidden in GitHub staging: ' + relative)
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(encoding='utf-8'))
            except Exception as exc:
                errors.append('Python syntax: ' + relative + ': ' + str(exc))
        if path.stat().st_size <= 1024**2 and path.suffix.lower() in {'.py', '.sh', '.ps1', '.md', '.txt', '.json', '.html', '.yaml', '.yml', '.patch', '.toml', '.jinja', '.log', ''}:
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeError:
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(label + ': ' + relative)
    if errors:
        raise RuntimeError('Public-tree checks failed:\n' + '\n'.join(errors[:30]))
    return {'files': count, 'bytes': size, 'syntax_and_private_data_scan': 'passed'}


def inspect_onnx_graph(path):
    """Inspect references without loading external weights or executing the graph."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError('ONNX reference verification requires tools/requirements-publish.txt') from exc
    model = onnx.load(str(path), load_external_data=False)
    external = {}
    def visit(message):
        if isinstance(message, onnx.TensorProto) and message.data_location == onnx.TensorProto.EXTERNAL:
            fields = {item.key: item.value for item in message.external_data}
            location = repo_path(fields.get('location', ''))
            if not location or location == '.':
                raise RuntimeError('ONNX external tensor has no location')
            target = under(path.parent, path.parent / location)
            if not target.is_file() or (path.parent / location).is_symlink():
                raise RuntimeError('Missing ONNX external data: ' + location)
            size = target.stat().st_size
            offset = int(fields.get('offset', 0))
            length = int(fields.get('length', size - offset))
            if offset < 0 or length < 0 or offset + length > size:
                raise RuntimeError('ONNX external tensor range exceeds file: ' + location)
            record = external.setdefault(location, {'bytes': size, 'tensor_count': 0, 'max_end': 0})
            record['tensor_count'] += 1
            record['max_end'] = max(record['max_end'], offset + length)
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                for child in value if field.is_repeated else [value]:
                    visit(child)
    visit(model)
    result = {'ir_version': model.ir_version,
              'opsets': [{'domain': item.domain, 'version': item.version} for item in model.opset_import],
              'external_data': [{'location': key, **value} for key, value in sorted(external.items())]}
    del model
    gc.collect()
    return result


def verify_onnx_bundles(kit, family, entries, inspect_graphs=False):
    by_path = {entry['path']: entry for entry in entries}
    declared = set()
    graph_count = external_count = 0
    root = kit / 'huggingface' / 'onnx' / family
    for variant in read(root / 'model-index.json')['variants']:
        prefix = repo_path(variant['variant']) + '/'
        report = read(root / variant['variant'] / 'onnx-manifest.json')
        if report.get('status') != 'complete' or not report.get('graphs'):
            raise RuntimeError('Incomplete ONNX bundle: ' + family + '/' + prefix)
        for graph in report['graphs']:
            remote = prefix + repo_path(graph['path'])
            entry = by_path.get(remote)
            if not entry or entry['sha256'] != graph['sha256'] or entry['bytes'] != graph['bytes']:
                raise RuntimeError('Missing or mismatched ONNX graph: ' + remote)
            declared.add(remote)
            graph_count += 1
            if inspect_graphs:
                actual = inspect_onnx_graph(Path(entry['source']))
                if actual != graph['inspection']:
                    raise RuntimeError('ONNX external references changed: ' + remote)
                print('ONNX REFERENCES VERIFIED:', family + '/' + remote, flush=True)
            data_by_location = {item['location']: item for item in graph['external_files']}
            if len(data_by_location) != len(graph['external_files']):
                raise RuntimeError('Duplicate external data declaration: ' + remote)
            locations = {item['location'] for item in graph['inspection']['external_data']}
            if set(data_by_location) != locations:
                raise RuntimeError('Incomplete external data declaration: ' + remote)
            for reference in graph['inspection']['external_data']:
                location = repo_path(reference['location'])
                target = (PurePosixPath(remote).parent / location).as_posix()
                data_entry = by_path.get(target)
                saved = data_by_location[location]
                if not data_entry or data_entry['bytes'] != reference['bytes'] or data_entry['sha256'] != saved['sha256']:
                    raise RuntimeError('ONNX weight file missing from upload: ' + target)
                if saved['path'] != (PurePosixPath(graph['path']).parent / location).as_posix():
                    raise RuntimeError('ONNX weight layout changed: ' + target)
                declared.add(target)
                external_count += 1
    actual_graphs = {item['path'] for item in entries if item['path'].endswith('.onnx')}
    if actual_graphs != {item for item in declared if item.endswith('.onnx')}:
        raise RuntimeError('Unlisted ONNX graph in payload map: ' + family)
    return {'onnx_files': graph_count, 'external_data_files': external_count,
            'onnx_references': 'parsed and complete' if inspect_graphs else 'complete per prepared manifest; run verify to reparse'}


def sources_for(kit):
    mapping = read(kit / '.local' / 'local-sources.json')
    if mapping.get('schema_version') != 2:
        raise RuntimeError('Use the split OM/ONNX release kit (source map schema_version 2)')
    return mapping['formats']


def checked_entries(kit, artifact_format, family, entries, hash_large):
    seen = set()
    for entry in entries:
        remote = repo_path(entry['path'])
        if remote in seen:
            raise RuntimeError('Duplicate upload path: ' + family + '/' + remote)
        seen.add(remote)
        if artifact_format == 'onnx' and remote.endswith('.om'):
            raise RuntimeError('OM payload cannot be uploaded to an ONNX repository')
        if artifact_format == 'om' and (remote.endswith('.onnx') or '/onnx/' in remote or entry.get('kind','').startswith('onnx_')):
            raise RuntimeError('ONNX payload cannot be uploaded to an OM repository')
        source = Path(entry['source'])
        if not source.is_file() or source.is_symlink():
            raise RuntimeError('Missing/redirected payload: ' + family + '/' + remote)
        if source.stat().st_size != entry['bytes']:
            raise RuntimeError('Payload size changed: ' + family + '/' + remote)
        if hash_large and sha(source) != entry['sha256']:
            raise RuntimeError('Payload SHA256 mismatch: ' + family + '/' + remote)
        if hash_large and source.suffix in {'.om','.onnx'}:
            print('SHA256 VERIFIED:', artifact_format + '/' + family + '/' + remote, flush=True)
    onnx_report = verify_onnx_bundles(kit, family, entries, inspect_graphs=hash_large) if artifact_format == 'onnx' else {
        'onnx_files':0, 'external_data_files':0, 'onnx_references':'not included in this OM repository'}
    return {'files':len(entries), 'bytes':sum(item['bytes'] for item in entries), **onnx_report,
            'om_files':sum(item['path'].endswith('.om') for item in entries),
            'payload_sha256':'verified' if hash_large else 'not read; run verify'}


def verified_entries(kit, families=None, hash_large=False, formats=None):
    mapping = sources_for(kit)
    selected = set(formats or mapping)
    result = {}
    for artifact_format in selected:
        if artifact_format not in mapping:
            raise ValueError('Unknown format')
        available = mapping[artifact_format]
        if families and not set(families).issubset(available):
            raise ValueError('Unknown family')
        result[artifact_format] = {family:checked_entries(kit,artifact_format,family,entries,hash_large)
                                   for family,entries in available.items() if not families or family in families}
    if selected != set(result):
        raise ValueError('Unknown format')
    return result


def config_for_upload(kit):
    path = kit / 'release-config.json'
    if not path.is_file():
        raise RuntimeError('Copy release-config.example.json to release-config.json and complete it first')
    config = read(path)
    if config.get('schema_version') != 2:
        raise RuntimeError('Update release-config.json using the split-repository schema_version 2 template')
    names = [repo['repo_name'] for model in config['models'].values() for repo in model['repositories'].values()]
    if len(names) != len(set(names)):
        raise RuntimeError('OM and ONNX must use separate, unique repository names')
    return config


def metadata_operations(kit, family, config, artifact_format):
    # Metadata is read only from the reviewed, small staging root. ONNX/OM/tokenizer
    # payloads come from the explicit private allowlist, never a recursive drive upload.
    from huggingface_hub import CommitOperationAdd
    root = kit / 'huggingface' / artifact_format / family
    settings = config['models'][family]
    for key in ['source_and_rights_reviewed', 'license_reviewed']:
        if settings.get(key) is not True:
            raise RuntimeError(f'{family}: complete the source/license review fields before upload')
    if not settings.get('checkpoint_source', '').strip():
        raise RuntimeError(f'{family}: checkpoint_source is required; do not infer identity from architecture')
    license_id = settings.get('license_id', '')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*', license_id):
        raise RuntimeError(f'{family}: set the actual license identifier')
    if not (root / 'LICENSE').is_file():
        raise RuntimeError(f'{family}: add the applicable full model LICENSE to the staging root')
    if family == 'gemma3' and not (root / 'NOTICE').is_file():
        raise RuntimeError('Gemma redistribution requires the applicable terms/notices; read LICENSE_STATUS.md')
    check_public(root, large_allowed=True)
    card = (root / 'README.md').read_text(encoding='utf-8')
    frontmatter = card.split('---', 2)
    if len(frontmatter) != 3:
        raise RuntimeError('Invalid model-card front matter')
    extra = '\nlicense: ' + json.dumps(license_id)
    if settings.get('base_model'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', settings['base_model']):
            raise ValueError('Invalid base_model identifier')
        extra += '\nbase_model: ' + json.dumps(settings['base_model'])
    provenance = read(root / 'PROVENANCE.json')
    provenance['publisher_review'] = {k: settings.get(k) for k in ['checkpoint_source', 'upstream_revision', 'base_model', 'license_id', 'source_and_rights_reviewed', 'license_reviewed']}
    provenance['publisher_review']['artifact_format'] = artifact_format
    operations = []
    for file in sorted(root.rglob('*')):
        if not file.is_file():
            continue
        if ((artifact_format == 'onnx' and file.suffix == '.om') or
                (artifact_format == 'om' and (file.suffix in {'.onnx','.data'} or 'onnx' in file.relative_to(root).parts))):
            raise RuntimeError('Wrong-format binary in staged metadata: ' + file.name)
        relative = repo_path(file.relative_to(root).as_posix())
        if relative == 'README.md':
            content = ('---' + frontmatter[1] + extra + '\n---' + frontmatter[2]).encode('utf-8')
        elif relative == 'PROVENANCE.json':
            content = (json.dumps(provenance, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        else:
            content = file.read_bytes()
        # Verify rendered metadata too, not only its template.
        for label, pattern in SECRET_PATTERNS.items():
            if file.suffix in {'.md', '.json', '.txt'} and pattern.search(content.decode('utf-8')):
                raise RuntimeError(label + ' in rendered metadata ' + relative)
        operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=content))
    return operations


def publish_hf(kit, family, upload, artifact_format):
    if artifact_format not in {'om','onnx'}:
        raise ValueError('Select exactly one format: om or onnx')
    if not upload:
        print(json.dumps(verified_entries(kit, [family], formats=[artifact_format]), indent=2))
        print('Preview only. No login, repository creation or upload was attempted.')
        return
    config = config_for_upload(kit)
    namespace = config['huggingface']['namespace']
    settings = config['models'][family]['repositories'][artifact_format]
    repo_id = valid_repo(namespace, settings['repo_name'])
    from huggingface_hub import CommitOperationAdd, HfApi
    operations = metadata_operations(kit, family, config, artifact_format)
    verified_entries(kit, [family], hash_large=True, formats=[artifact_format])
    mapping = sources_for(kit)[artifact_format][family]
    metadata_paths = {operation.path_in_repo for operation in operations}
    for item in mapping:
        remote = repo_path(item['path'])
        if remote in metadata_paths:
            raise RuntimeError('Metadata/payload name collision: ' + remote)
        operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=item['source']))
    api = HfApi()
    # Authentication is provided by `hf auth login` / the user's normal environment.
    api.whoami()
    visibility = settings.get('private', True)
    if type(visibility) is not bool:
        raise ValueError('private must be a JSON boolean')
    # Deliberately fail for an existing repo rather than overwrite a user's release.
    api.create_repo(repo_id=repo_id, repo_type='model', private=visibility, exist_ok=False)
    commit = api.create_commit(repo_id=repo_id, repo_type='model', operations=operations,
                               commit_message='Add verified ' + artifact_format.upper() + ' conversion packages')
    receipt = {'repo_id': repo_id, 'commit_url': commit.commit_url, 'commit_sha': commit.oid,
               'files': len(operations), 'private': visibility, 'artifact_format':artifact_format}
    write(kit / '.local' / ('uploaded-' + family + '-' + artifact_format + '.json'), receipt)
    print(json.dumps(receipt, indent=2))


def publish_github(kit, upload):
    directory = kit / 'github' / 'ascend-om-conversion-lab'
    check_public(directory)
    if not upload:
        print('Preview only. GitHub source: github/ascend-om-conversion-lab; no repository created.')
        return
    config = config_for_upload(kit)
    settings = config['github']
    if settings.get('code_license_reviewed') is not True or settings.get('public_files_reviewed') is not True:
        raise RuntimeError('Review the code license, notices and public files first')
    repo_id = valid_repo(settings['namespace'], settings['repo_name'])
    if type(settings.get('private', True)) is not bool:
        raise ValueError('private must be a JSON boolean')
    def run(*argv, check=True):
        return subprocess.run(argv, cwd=directory, check=check, text=True, capture_output=True)
    run('gh', 'auth', 'status')
    existing = run('gh', 'repo', 'view', repo_id, '--json', 'nameWithOwner', check=False)
    if existing.returncode == 0:
        raise RuntimeError('Repository already exists; refusing automatic overwrite. Use the documented manual update workflow.')
    if not (directory / '.git').exists():
        run('git', 'init', '-b', 'main')
    for key in ['user.name', 'user.email']:
        if not run('git', 'config', '--get', key, check=False).stdout.strip():
            raise RuntimeError('Set your intended Git commit identity locally before publishing: ' + key)
    if run('git', 'remote', check=False).stdout.strip():
        raise RuntimeError('A Git remote already exists; refusing to replace it')
    run('git', 'add', '--', '.')
    if run('git', 'diff', '--cached', '--quiet', check=False).returncode:
        run('git', 'commit', '-m', 'Prepare Ascend OM conversion tools and beginner guide')
    args = ['gh', 'repo', 'create', repo_id, '--source', str(directory), '--remote', 'origin', '--push',
            '--private' if settings.get('private', True) else '--public',
            '--description', 'Ascend310B4 OM conversion workflows, numerical validation and Chinese beginner guide']
    run(*args)
    write(kit / '.local' / 'uploaded-github.json', {'repo_id': repo_id, 'url': 'https://github.com/' + repo_id})
    print('Created: https://github.com/' + repo_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'check', 'verify', 'publish-github', 'publish-hf'])
    parser.add_argument('--kit-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--family', choices=['gemma3', 'tinyllama', 'qwen35'])
    parser.add_argument('--format', dest='artifact_format', choices=['om','onnx'], help='Required for publish-hf; optional filter for verify/check')
    parser.add_argument('--upload', action='store_true', help='Explicit network write; accepted only for publish-*')
    args = parser.parse_args()
    kit = args.kit_root.resolve()
    if args.upload and not args.command.startswith('publish-'):
        parser.error('--upload is accepted only with publish-*')
    if args.artifact_format and args.command not in {'check','verify','publish-hf'}:
        parser.error('--format is accepted only with check, verify or publish-hf')
    if args.command == 'plan':
        print(json.dumps(read(kit / 'release-catalog.json'), ensure_ascii=False, indent=2))
    elif args.command == 'check':
        reports = {'github': check_public(kit / 'github' / 'ascend-om-conversion-lab')}
        for artifact_format in [args.artifact_format] if args.artifact_format else ['om','onnx']:
            for root in (kit / 'huggingface' / artifact_format).iterdir():
                if root.is_dir():
                    reports[artifact_format+'/'+root.name] = check_public(root, large_allowed=True)
        reports['payloads'] = verified_entries(kit, [args.family] if args.family else None, formats=[args.artifact_format] if args.artifact_format else None)
        write(kit / '.local' / 'check-report.json', reports)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    elif args.command == 'verify':
        result = verified_entries(kit, [args.family] if args.family else None, hash_large=True, formats=[args.artifact_format] if args.artifact_format else None)
        report_path = kit / '.local' / 'payload-verification.json'
        saved = read(report_path) if report_path.is_file() else {}
        if 'om' not in saved and 'onnx' not in saved:saved = {}
        for artifact_format,values in result.items():saved.setdefault(artifact_format,{}).update(values)
        write(report_path, saved)
        print(json.dumps(result, indent=2))
    elif args.command == 'publish-github':
        publish_github(kit, args.upload)
    else:
        if not args.family or not args.artifact_format:
            parser.error('publish-hf requires --family and --format om|onnx; each format has a separate repository')
        publish_hf(kit, args.family, args.upload, args.artifact_format)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Avoid printing command output or tokens when auth/upload commands fail.
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(1)
