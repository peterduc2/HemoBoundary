#!/usr/bin/env python3
"""Isolated entry point: never upgrades the active Colab pandas/NumPy installation.
The notebook/standalone bundle supplies this directory before calling launch().
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, sys, venv
from pathlib import Path

def launch(mode='full',datasets='SHIN,BALINT,KHAN',root=None,output=None,venv_path=None):
    base=Path(__file__).resolve().parent
    if not (3,11)<=sys.version_info[:2]<=(3,13):
        raise RuntimeError('This pinned environment targets Python 3.11, 3.12 and 3.13. Do not silently upgrade to a different numerical stack.')
    try:
        from google.colab import drive
        if not any((Path(m)/'MyDrive').exists() for m in ('/content/gdrive','/content/drive')):
            drive.mount('/content/gdrive',force_remount=False)
    except ImportError:pass
    env_root=Path(venv_path) if venv_path else (Path('/content') if Path('/content').exists() else base)/'hb_v8_isolated_env'
    requirement=(base/'requirements.txt').read_bytes()
    key=hashlib.sha256(requirement+platform.python_version().encode()).hexdigest()
    marker=env_root/'HB_ENVIRONMENT.json'
    python=env_root/'bin/python'
    ready=False
    if python.is_file() and marker.is_file():
        try:ready=json.loads(marker.read_text()).get('key')==key
        except (OSError,ValueError):pass
    if not ready:
        print('Creating/using a dedicated environment without ensurepip. Colab runtime packages are not changed.',flush=True)
        # Colab Python 3.13 may ship without working venv.ensurepip. The old
        # with_pip=True launcher aborts during environment creation even when
        # its existing notebook pip works. Building WITHOUT bundled pip and
        # using the host's modern `pip --python <venv>` installs only INTO the
        # isolated environment, including previously half-created environments.
        venv.EnvBuilder(with_pip=False,clear=False).create(env_root)
        host_pip=[sys.executable,'-m','pip','--python',str(python)]
        pip_probe=subprocess.run(host_pip+['--version'],capture_output=True,text=True)
        if pip_probe.returncode != 0:
            raise RuntimeError(
                'The active runtime needs pip >=22.3 with --python support. '
                'Isolated environment was created, but no benchmark dependencies '
                'were installed. Host pip diagnostic: '
                + (pip_probe.stderr or pip_probe.stdout)[-1200:]
            )
        print('Installing pinned packages through host pip --python into',python,flush=True)
        subprocess.run(host_pip+['install','--disable-pip-version-check',
                                 '-r',str(base/'requirements.txt')],check=True)
        # These checks also target the isolated environment, not Colab's kernel.
        subprocess.run(host_pip+['check'],check=True)
        resolved=subprocess.check_output(host_pip+['freeze'],text=True)
        (env_root/'resolved_requirements.txt').write_text(resolved)
        marker.write_text(json.dumps({'key':key,'python':platform.python_version()},indent=2))
    child_env=os.environ.copy()
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        child_env[variable]='1'
    child_env['MPLBACKEND']='Agg';child_env['PYTHONUNBUFFERED']='1';child_env['PYTHONPATH']=str(base)
    # Tests use only synthetic software fixtures in temporary directories.
    test=subprocess.run([str(python),'-m','pytest','tests','-q','--disable-warnings'],cwd=base,env=child_env,check=True,text=True)
    cmd=[str(python),str(base/'hb_benchmark.py'),'--mode',mode,'--datasets',datasets]
    if root:cmd+=['--root',str(root)]
    if output:cmd+=['--output',str(output)]
    print('\nStarting the declared benchmark. Original HemoBoundary files remain read-only.',flush=True)
    subprocess.run(cmd,cwd=base,env=child_env,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',default='full',choices=('full','seed1','pilot','preflight','summarize'))
    p.add_argument('--datasets',default='SHIN,BALINT,KHAN');p.add_argument('--root');p.add_argument('--output');p.add_argument('--venv-path')
    a=p.parse_args();launch(a.mode,a.datasets,a.root,a.output,a.venv_path)
