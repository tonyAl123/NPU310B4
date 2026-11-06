"""Frozen local Qwen3.5A multimodal build: independent reference -> ONNX gates -> two ATC jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import traceback
import uuid


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(args):
    staging, package = Path(args.staging).resolve(), Path(__file__).resolve().parent
    build_root = (Path.home() / 'ascend-model-build')
    assert build_root.resolve() == build_root
    request = read(package / 'request.json')
    assert (request['context'], request['image_size'], request['visual_tokens']) == (2048, 768, 576)
    reuse = None
    if args.reuse_attempt:
        from reuse import check_attempt
        reuse = check_attempt(args.reuse_attempt, staging, build_root, request, package)
    job = staging / 'builds' / request['job_id']
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    shared, local = job / 'attempts' / attempt, build_root / 'conversions' / request['job_id'] / attempt
    assert local.resolve().is_relative_to(build_root / 'conversions')
    shared.mkdir(parents=True)
    local.mkdir(parents=True)
    for folder in ('logs', 'source', 'deliverables'):
        (shared / folder).mkdir()
    manifest = {'build_id': request['job_id'] + '/' + attempt, 'created_at': datetime.now(timezone.utc).isoformat(),
                'status': 'running', 'stage': 'preflight', 'request': request, 'commands': [], 'stages': {},
                'artifacts': [], 'board_runtime_verified': False, 'linux_work_directory': str(local),
                'board_validation': 'not_run'}
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'})

    def phase(name):
        manifest['stage'] = name
        write(shared / 'build-manifest.json', manifest)
        write(job / 'conversion-status.json', {'status': manifest['status'], 'stage': name,
                                               'attempt': str(shared), 'local': str(local)})
        print('PHASE:', name, '| Windows-visible output:', shared, flush=True)

    def command(name, arguments, cwd=None):
        phase(name)
        manifest['commands'].append({'stage': name, 'argv': list(map(str, arguments)),
                                     'environment': 'ATC container' if name.startswith('atc_') else 'WSL/Linux'})
        print('COMMAND:', shlex.join(map(str, arguments)), flush=True)
        with (shared / 'logs' / (name + '.log')).open('w', encoding='utf-8') as log:
            with subprocess.Popen(list(map(str, arguments)), cwd=cwd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors='replace') as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
        manifest['stages'][name] = {'status': 'passed' if code == 0 else 'failed', 'exit_code': code,
                                    'log': 'logs/' + name + '.log'}
        write(shared / 'build-manifest.json', manifest)
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}')

    def archive():
        for folder in ('onnx', 'reference', 'adapter-self-test'):
            if (local / folder).exists() and not (shared / folder).exists():
                shutil.copytree(local / folder, shared / folder)
        if (local / 'om').exists():
            for graph in ('vision', 'decoder'):
                if (local / 'om' / graph / 'ascend-logs').exists() and not (shared / 'logs' / (graph + '-ascend')).exists():
                    shutil.copytree(local / 'om' / graph / 'ascend-logs', shared / 'logs' / (graph + '-ascend'))

    def report():
        text = f"""# Qwen3.5A 2B 图文 OM 构建记录
状态：{manifest['status']}
目标：Ascend310B4，batch1，总上下文2048，单图768×768；图片占576 token。
原始本地模型及文件SHA256见request.json；不保证与官方未修改权重一致。
当前/最后阶段：{manifest['stage']}
错误：{manifest.get('error', '无')}
Linux构建：{local}
Windows记录：{shared}
导出包：{manifest.get('export_directory', '尚未生成')}

vision_768.om：输入预处理后的2304个16×16时空patch，输出576×2048图像特征。
decoder_ctx2048.om：每次输入一个token、KV写入位置、三轴MRoPE坐标、图像向量与模态标记。
循环状态S保持FP32，卷积状态及KV为FP16。编译采用must_keep_origin_dtype保留图内混合精度。
视觉精度配置：{manifest.get('vision_profile', 'FP16内部计算及接口')}。
所有状态、输入输出名称、dtype和shape见两个interface.json。不能直接使用旧Qwen3/0.8B板端程序。
prepare_input.py处理原图、保持比例缩放补边、生成图文token及位置，未提供已验证的板端推理程序。

先做小模型原生/ONNX两子图测试，再独立原生FP32参考，最后做两个全量ONNX子图与图文串联验证。
原生完整图文forward另与拆分后的原生视觉+分块语言forward对比。
ONNX图文prefill从空状态逐token运行，并校验greedy生成；高位置使用各自真实前缀种子，
不把后续循环状态截断冒充较早状态。包含位置511、1023、2046、2047。
这些数值测试不等于真实照片识别质量评测或开发板推理验证。
board_runtime_verified=false。未执行驱动升级、远程登录、开发板加载、NPU精度或速度测试。
"""
        (shared / 'BUILD_REPORT.md').write_text(text, encoding='utf-8')
        shutil.copy2(shared / 'BUILD_REPORT.md', shared / 'deliverables/BUILD_REPORT.md')

    try:
        phase('preflight')
        environment = read(staging / 'environment-status.json')
        assert environment['environment_ready'] and environment['smoke_compile'] == 'passed'
        project = build_root / 'qwen-atlas-slm'
        assert subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip() == request['project_commit']
        assert not subprocess.check_output(['git', '-C', str(project), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available_ram = int(meminfo['MemAvailable'].split()[0]) * 1024
        assert available_ram > 22 * 1024**3, 'Need more than 22 GiB currently available WSL RAM for this multimodal 2B build'
        assert shutil.disk_usage(local).free > 35 * 1024**3, 'Need 35 GiB free Linux disk'
        assert shutil.disk_usage(staging).free > 20 * 1024**3, 'Need 20 GiB free Windows disk for build records and export'
        versions = {name: importlib.metadata.version(name) for name in request['versions']}
        assert versions == request['versions'] == environment['versions'], 'Locked dependency versions changed'
        manifest['host'] = {'machine': platform.machine(), 'kernel': platform.release(),
                             'available_ram_bytes': available_ram, 'free_linux_bytes': shutil.disk_usage(local).free,
                             'free_windows_bytes': shutil.disk_usage(staging).free}
        manifest['versions'] = versions
        podman = ['podman', '--cgroup-manager=cgroupfs', '--events-backend=file']
        image = request['container_image_id']
        write(shared / 'container-image.json', json.loads(subprocess.check_output([*podman, 'image', 'inspect', image], text=True)))
        phase('snapshot_source_and_model')
        source = local / 'source'
        source.mkdir()
        for record in request['source_files']:
            relative = Path(record['name'])
            assert not relative.is_absolute() and '..' not in relative.parts
            original, destination = package / relative, source / relative
            assert original.is_file() and original.stat().st_size == record['bytes'] and sha(original) == record['sha256'], 'Prepared source changed: ' + str(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            assert sha(destination) == record['sha256']
        shutil.copytree(source, shared / 'source', dirs_exist_ok=True)
        write(source / 'request.json', request)
        write(local / 'request.json', request)
        write(shared / 'request.json', request)
        write(shared / 'validation-policy.json', request['validation'])
        write(shared / 'memory-estimate.json', request['memory_estimate'])
        model_source = (staging / request['model_relative_path']).resolve()
        assert model_source.is_relative_to((staging / 'models').resolve())
        snapshot = local / 'model'
        if reuse:
            from reuse import reuse_model
            reuse_model(reuse[1], snapshot, request)
        else:
            snapshot.mkdir()
            for record in request['files']:
                original = model_source / record['name']
                assert original.resolve().parent == model_source
                target = snapshot / record['name']
                shutil.copy2(original, target)
                assert target.stat().st_size == record['bytes'] and sha(target) == record['sha256'], 'Checkpoint changed: ' + record['name']
        from importlib.util import find_spec
        module_root = Path(find_spec('transformers').origin).parent
        manifest['source'] = {'project_commit': request['project_commit'], 'lock_sha256': sha(project / 'pixi.lock'),
            'native_sources': {name: sha(module_root / name) for name in
                ('models/qwen3_5/modeling_qwen3_5.py', 'cache_utils.py', 'masking_utils.py', 'vision_utils.py')},
            'adapter_files': request['source_files'], 'documentation': request['source_verification']}
        manifest['vision_profile'] = 'FP32图像输入、内部计算及权重，FP16特征输出；语言解码器不变'
        memory = read(shared / 'memory-estimate.json')
        memory.update(vision_weight_dtype='float32', vision_parameter_count=331416576,
                      additional_vision_weight_bytes=331416576 * 2,
                      vision_input_dtype='float32', vision_output_dtype='float16')
        write(shared / 'memory-estimate.json', memory)
        command('adapter_selftest', [sys.executable, '-u', source / 'adapter.py', 'selftest', '--output', local / 'adapter-self-test'], project)
        if reuse:
            from reuse import reuse_products, reuse_isolation
            phase('verify_reused_artifacts')
            assert manifest['source']['native_sources'] == reuse[2]['source']['native_sources'], 'Installed native implementation changed'
            manifest['reuse'] = {'source_attempt': str(reuse[0]), 'source_local': str(reuse[1]),
                                 'files': reuse_products(reuse[0], reuse[1], local)}
            for name in ('native_reference', 'export_decoder'):
                manifest['stages'][name] = {'status': 'passed', 'execution': 'reused_after_hash_verification',
                                           'source_attempt': str(reuse[0])}
            diagnostics = shared / 'diagnostics/native-vision'
            previous_isolation = reuse_isolation(reuse[0], diagnostics, request)
            if previous_isolation:
                manifest['stages']['isolate_decoder'] = previous_isolation
                print('Reused passed native-vision diagnostic; this diagnostic cannot unlock ATC.', flush=True)
            else:
                diagnostics.mkdir(parents=True)
                command('isolate_decoder', [sys.executable, '-u', source / 'validate.py', '--local', local,
                        '--shared', diagnostics, '--package', source, '--reference-vision'], project)
            isolated = read(diagnostics / 'validation.json')
            assert isolated['status'] == 'passed' and not isolated['eligible_for_atc']
            # Preserve the prior graph before creating the new input precision profile.
            (local / 'onnx/vision').rename(local / 'onnx/vision-original')
            command('export_vision', [sys.executable, '-u', source / 'adapter.py', 'vision', '--model', snapshot,
                    '--output', local / 'onnx/vision', '--vision-fp32'], project)
        else:
            command('native_reference', [sys.executable, '-u', source / 'native.py', '--local', local, '--package', source], project)
            for graph in ('vision', 'decoder'):
                precision = ['--vision-fp32'] if graph == 'vision' else []
                command('export_' + graph, [sys.executable, '-u', source / 'adapter.py', graph, '--model', snapshot,
                                            '--output', local / 'onnx' / graph, *precision], project)
        validation_precision = ['--vision-input-dtype', 'float32']
        command('numeric_validation', [sys.executable, '-u', source / 'validate.py', '--local', local,
                                       '--shared', shared, '--package', source, *validation_precision], project)
        manifest['validation'] = read(shared / 'validation.json')
        assert manifest['validation']['status'] == 'passed' and manifest['validation']['eligible_for_atc']
        manifest['compile'] = {}
        for graph in ('vision', 'decoder'):
            directory = local / 'om' / graph
            (directory / 'ascend-logs').mkdir(parents=True)
            stem = 'vision_768' if graph == 'vision' else 'decoder_ctx2048'
            shape = (shared / f'{graph}-input-shape.txt').read_text().strip()
            atc = ['--model=/model/model.onnx', '--framework=5', '--output=/output/' + stem,
                   '--input_format=ND', '--input_shape=' + shape, '--soc_version=Ascend310B4',
                   '--precision_mode=must_keep_origin_dtype', '--log=info']
            argv = [*podman, 'run', '--rm', '--cgroups=disabled', '--network=none',
                    '-e', 'TE_PARALLEL_COMPILER=1', '-e', 'MAX_COMPILE_CORE_NUMBER=1',
                    '-e', 'ASCEND_PROCESS_LOG_PATH=/output/ascend-logs',
                    '-v', str(local / 'onnx' / graph) + ':/model:ro', '-v', str(directory) + ':/output:Z',
                    '-w', '/output', image, 'bash', '-c', 'atc "$@"', 'atc-job', *atc]
            manifest['compile'][graph] = {'image_id': image, 'arguments': atc, 'parallelism': 1}
            command('atc_' + graph, argv, local)
            om = directory / (stem + '.om')
            assert om.is_file() and om.stat().st_size > 0
            assert 'ATC run success' in (shared / 'logs' / ('atc_' + graph + '.log')).read_text()
            manifest['compile'][graph]['om'] = {'bytes': om.stat().st_size, 'sha256': sha(om)}
        phase('deliver')
        delivery = shared / 'deliverables'
        for graph, stem in (('vision', 'vision_768'), ('decoder', 'decoder_ctx2048')):
            original, target = local / 'om' / graph / (stem + '.om'), delivery / (stem + '.om')
            shutil.copy2(original, target)
            assert sha(target) == manifest['compile'][graph]['om']['sha256']
            for suffix in ('interface.json', 'input-shape.txt'):
                shutil.copy2(shared / (graph + '-' + suffix), delivery / (graph + '-' + suffix))
            shutil.copy2(shared / 'logs' / ('atc_' + graph + '.log'), delivery / ('atc_' + graph + '.log'))
        tokenizer = delivery / 'tokenizer'
        tokenizer.mkdir()
        for record in request['files']:
            if not record['name'].endswith('.safetensors'):
                shutil.copy2(snapshot / record['name'], tokenizer / record['name'])
        for name in ('contract.py', 'prepare_input.py', 'README.md', 'preprocess-self-test.json'):
            shutil.copy2(source / name, delivery / name)
        usage = (delivery / 'README.md').read_text(encoding='utf-8')
        usage = usage.replace('当前状态：转换流程已准备，尚未执行本次模型导出、数值验证或ATC。',
                              '本目录由数值验证和两个ATC编译步骤通过后生成。实际构建状态见build-manifest.json；尚未执行NPU推理。')
        usage += '\n本次视觉模块使用FP32图像输入、内部权重与计算、FP16特征输出；语言模块与缓存接口不变。详细验证见validation.json。\n'
        (delivery / 'README.md').write_text(usage, encoding='utf-8')
        shutil.copytree(source / 'assets', delivery / 'fixtures')
        shutil.copytree(shared / 'reference-vectors', delivery / 'reference-vectors')
        shutil.copy2(local / 'adapter-self-test/self-test.json', delivery / 'adapter-self-test.json')
        for name in ('request.json', 'validation.json', 'validation-policy.json', 'memory-estimate.json'):
            shutil.copy2(shared / name, delivery / name)
        phase('archive')
        archive()
        manifest['artifacts'] = [{'path': str(p.relative_to(shared)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                                  for p in sorted(shared.rglob('*')) if p.is_file() and p.name != 'build-manifest.json']
        exports = staging / 'exports'
        exports.mkdir(exist_ok=True)
        target = exports / 'Qwen3.5A-2B-Ascend310B4-ctx2048-img768'
        if target.exists():
            target = exports / (target.name + '-' + attempt)
        assert not target.exists() and target.resolve().parent == exports.resolve()
        needed = sum(p.stat().st_size for p in delivery.rglob('*') if p.is_file())
        assert shutil.disk_usage(exports).free > needed + 256 * 1024**2, 'Need more free Windows disk space to finish export'
        manifest.update(status='compiled', stage='complete', export_directory=str(target),
                        finished_at=datetime.now(timezone.utc).isoformat())
        report()
        write(shared / 'build-manifest.json', manifest)
        write(delivery / 'build-manifest.json', manifest)
        temporary = exports / ('.qwen35a-mm2048-partial-' + uuid.uuid4().hex[:8])
        temporary.mkdir()
        package_files = []
        for original in sorted(delivery.rglob('*')):
            if original.is_file():
                relative = original.relative_to(delivery)
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
                digest = sha(original)
                assert destination.stat().st_size == original.stat().st_size and sha(destination) == digest
                package_files.append({'path': str(relative), 'bytes': destination.stat().st_size, 'sha256': digest})
        write(temporary / 'package-manifest.json', {'status': 'exported', 'source_build_id': manifest['build_id'],
              'context': 2048, 'image_size': 768, 'visual_tokens': 576, 'board_runtime_verified': False,
              'files': package_files})
        temporary.rename(target)
        phase('complete')
        print('EXPORT SUCCESS:', target, flush=True)
    except Exception as error:
        if manifest['status'] == 'compiled':
            manifest['stage'] = 'export_failed'
        manifest.update(status='failed', error=str(error), finished_at=datetime.now(timezone.utc).isoformat())
        (shared / 'logs/failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(traceback.format_exc(), flush=True)
        try:
            archive()
        except Exception as archive_error:
            manifest['archive_error'] = str(archive_error)
        report()
        phase(manifest['stage'])
    print('FINAL STATUS:', manifest['status'], '| Output:', shared, flush=True)
    return 0 if manifest['status'] == 'compiled' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--staging', required=True)
    parser.add_argument('--reuse-attempt', help='Reuse a failed numerical-gate attempt, isolate decoder, and trial FP32 vision')
    sys.exit(run(parser.parse_args()))
