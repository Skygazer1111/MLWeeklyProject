let python, namespace, initialized;
let executionCount = 0;
const status = message => self.postMessage({type: 'status', message});
async function initialize() {
  status('Downloading Python…');
  const {loadPyodide} = await import('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs');
  python = await loadPyodide({indexURL: 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'});
  status('Loading scientific packages…');
  await python.loadPackage(['numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn', 'ipython']);
  const response = await fetch('/data/traffic.csv');
  if (!response.ok) throw new Error('The dataset could not be loaded. Please reload and try again.');
  python.FS.mkdirTree('/data');
  python.FS.writeFile('/data/traffic.csv', await response.text());
  await python.runPythonAsync(`
import os, io, json, ast, base64, traceback, contextlib
os.chdir('/')
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
import IPython.display as _ipdisplay

_lab_outputs = []
def _lab_display(*objects, **kwargs):
    for obj in objects:
        data = {'text/plain': str(obj)}
        if hasattr(obj, '_repr_html_'):
            html = obj._repr_html_()
            if html is not None: data['text/html'] = html
        _lab_outputs.append({'output_type': 'display_data', 'data': data, 'metadata': {}})

def _lab_show(*args, **kwargs):
    for number in _plt.get_fignums():
        fig = _plt.figure(number)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=115, bbox_inches='tight')
        _lab_outputs.append({'output_type': 'display_data', 'data': {'image/png': base64.b64encode(buf.getvalue()).decode('ascii')}, 'metadata': {}})
    _plt.close('all')

_ipdisplay.display = _lab_display
_plt.show = _lab_show
def _lab_execute(source, namespace):
    _lab_outputs.clear()
    stdout, stderr = io.StringIO(), io.StringIO()
    failure = False
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            tree = ast.parse(source, filename='<notebook cell>')
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = tree.body.pop()
                exec(compile(tree, '<notebook cell>', 'exec'), namespace)
                result = eval(compile(ast.Expression(last.value), '<notebook cell>', 'eval'), namespace)
                if result is not None: _lab_display(result)
            else:
                exec(compile(tree, '<notebook cell>', 'exec'), namespace)
            if _plt.get_fignums(): _lab_show()
        except Exception as exc:
            failure = True
            _lab_outputs.append({'output_type': 'error', 'ename': type(exc).__name__, 'evalue': str(exc), 'traceback': traceback.format_exc().splitlines()})
            _plt.close('all')
    outputs = []
    if stdout.getvalue(): outputs.append({'output_type':'stream', 'name':'stdout', 'text':stdout.getvalue()})
    if stderr.getvalue(): outputs.append({'output_type':'stream', 'name':'stderr', 'text':stderr.getvalue()})
    outputs.extend(_lab_outputs)
    return json.dumps({'outputs': outputs, 'failed': failure})
`);
  reset();
  status('Python ready');
}
function reset() {
  namespace?.destroy();
  namespace = python.runPython("{'__name__': '__main__'}");
  executionCount = 0;
}
self.onmessage = async ({data}) => {
  const {id, action, code} = data;
  try {
    initialized ??= initialize();
    await initialized;
    if (action === 'reset') {
      reset();
      self.postMessage({id, result: {reset: true}});
      return;
    }
    python.globals.set('_lab_source', code);
    python.globals.set('_lab_namespace', namespace);
    const raw = await python.runPythonAsync('_lab_execute(_lab_source, _lab_namespace)');
    const result = JSON.parse(raw);
    result.execution_count = ++executionCount;
    self.postMessage({id, result});
  } catch (error) {
    self.postMessage({id, error: error.message || String(error)});
  }
};
