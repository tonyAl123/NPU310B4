"""Configure a historical conversion profile for a user's own local checkpoint.

This hashes local model files and refreshes request provenance. It does not
download weights, export graphs, compile models or publish anything.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def records(folder,exclude=()):
    result=[]
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:continue
        if path.name in exclude:continue
        if path.is_symlink():raise ValueError('Use regular files in the local checkpoint/profile')
        result.append({'name':path.relative_to(folder).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1]/'workspace')
    parser.add_argument('--profile',required=True)
    parser.add_argument('--model-dir',required=True,type=Path)
    parser.add_argument('--checkpoint-source',required=True,help='Actual download/derived checkpoint source; do not infer from architecture')
    parser.add_argument('--upstream-revision',default=None)
    args=parser.parse_args()
    workspace=args.workspace.resolve()
    profiles=load(workspace/'profiles.json')
    profile=profiles[args.profile]
    request_path=workspace/profile['request']
    request=load(request_path)
    model=args.model_dir.resolve()
    if not model.is_relative_to((workspace/'models').resolve()):
        raise ValueError('Place the complete checkpoint in workspace/models/<folder>')
    cfg=load(model/'config.json')
    original=request['config']
    for branch in ['', 'text_config', 'vision_config']:
        old=original if not branch else original.get(branch)
        new=cfg if not branch else cfg.get(branch)
        if old is None:continue
        if new is None:raise ValueError('Missing configuration branch: '+branch)
        for key in ['model_type','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','vocab_size','layer_types','sliding_window','rope_parameters','patch_size','temporal_patch_size','spatial_merge_size','out_hidden_size','depth','linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim','linear_conv_kernel_dim']:
            if key in old and old[key]!=new.get(key):
                raise ValueError('Architecture differs from this adapter: '+branch+'.'+key)
    if cfg.get('quantization_config'):
        raise ValueError('These profiles are not INT8/INT4 checkpoint importers')
    headers=[]
    for path in sorted(model.glob('*.safetensors')):
        with path.open('rb') as stream:
            n=struct.unpack('<Q',stream.read(8))[0]
            if n>20_000_000:raise ValueError('Unexpected safetensors header size')
            header=json.loads(stream.read(n))
        types=Counter()
        for name,item in header.items():
            if name!='__metadata__':types[item['dtype']]+=math.prod(item['shape'])
        headers.append({'file':path.name,'elements_by_dtype':dict(types),'stored_elements':sum(types.values())})
    if not headers:raise ValueError('A complete safetensors checkpoint is required')
    environment=load(workspace/'environment-status.json')
    if environment.get('environment_ready') is not True:raise ValueError('Run compiler setup and verify the environment first')
    if environment['project_commit']!=request['project_commit'] or environment['versions']!=request.get('versions',environment['versions']):
        raise ValueError('Environment differs from the pinned conversion implementation')
    request['versions']=environment['versions']
    request['config']=cfg
    request['model_relative_path']=model.relative_to(workspace).as_posix()
    request['model_source']=args.checkpoint_source
    request['upstream_revision']=args.upstream_revision
    request['files']=records(model)
    request['safetensors_header']={'files':headers,'stored_elements':sum(item['stored_elements'] for item in headers)}
    request['container_image_id']=environment['container_image_id']
    request['publication_reconfiguration']={'reference':'Historical adapter profile; fresh model identity and numerical validation are required','profile':args.profile}
    if profile.get('exporter'):
        digest=sha(workspace/profile['exporter'])
        request['adapter_sha256']=digest
    if profile['kind']=='multimodal':
        package=request_path.parent
        request['source_files']=records(package,exclude=('request.json','SHA256SUMS'))
    save(request_path,request)
    if profile['kind'] in ['multimodal','tinyllama']:
        package=request_path.parent
        if profile['kind']=='tinyllama':request['adapter_sha256']=sha(package/'export_tinyllama_fixed.py');save(request_path,request)
        sums=records(package,exclude=('SHA256SUMS',))
        (package/'SHA256SUMS').write_text(''.join(f"{item['sha256']}  {item['name']}\n" for item in sums),encoding='utf-8')
    save(workspace/'.configured'/f'{args.profile}.json',{'request_sha256':sha(request_path),'profile':args.profile})
    print('Configured:',args.profile)
    print('Request:',request_path)
    print('This only prepared the request. The conversion launcher will generate fresh validation and compile results.')


if __name__=='__main__':main()
