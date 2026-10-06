from pathlib import Path
import os, sys, json, shutil, zipfile
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.sites-runtime' / 'jupyter')
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True, exist_ok=True)
report = []
for path in sorted((root / 'notebooks').glob('*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    print('Executing', path.name, flush=True)
    NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': str(root)}}).execute()
    nbformat.write(nb, path)
    shutil.copy(path, root / 'dist' / 'notebooks' / path.name)
    images = sum('image/png' in out.get('data', {}) for c in nb.cells for out in c.get('outputs', []))
    errors = [out for c in nb.cells for out in c.get('outputs', []) if out.output_type == 'error']
    assert images >= 2 and not errors
    row = {'notebook': path.name, 'code_cells': sum(c.cell_type=='code' for c in nb.cells), 'graphs': images, 'errors': len(errors)}
    report.append(row)
    print(json.dumps(row), flush=True)
with zipfile.ZipFile(root / 'dist' / 'ml-weekly-notebooks.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in (root / 'notebooks').glob('*.ipynb'): archive.write(path, 'notebooks/' + path.name)
    for name in ['geyser.csv', 'geyser_original.csv', 'metadata.json']: archive.write(root / 'data' / name, 'data/' + name)
    for name in ['README.md', 'requirements.txt', 'environment-tested.txt', 'scripts/build_notebooks.py', 'scripts/execute_notebooks.py']: archive.write(root / name, name)
(root / 'verification.json').write_text(json.dumps(report, indent=2))
