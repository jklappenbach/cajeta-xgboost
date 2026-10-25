#!/usr/bin/env python3
"""Generate the cajeta.dev kernel catalog page for dev.cajeta.xgboost.

    python3 tools/catalog/gen-catalog.py ../../cpp/cajeta/docs/kernels/catalog-xgboost.md

Every @Kernel in src/main/cajeta/dev/cajeta/xgboost/gpu, in source order by
file: the training stage its file serves, whether it declares a 32-lane wave,
and its role from the kernel's own documentation. The page is markdown whose
table is raw HTML in a shape that cannot wrap at the site's prose width: three
grid columns (kernel linked to its source line, stage, @Wave(32)) and the role
on its own full-width row underneath.
"""
import argparse
import datetime
import html
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GPU = 'src/main/cajeta/dev/cajeta/xgboost/gpu'

# The training stage each kernel file serves, in the page's order.
STAGES = [
    ('GpuBinner.cajeta', 'Binning'),
    ('GpuHistogram.cajeta', 'Histograms'),
    ('GpuLeaves.cajeta', 'Leaves'),
    ('GpuObjective.cajeta', 'Gradients'),
    ('GpuPartition.cajeta', 'Partition'),
    ('GpuQuantiser.cajeta', 'Gradient quantisation'),
    ('GpuSplitFinder.cajeta', 'Split finding'),
]
ROLES = os.path.join(HERE, 'roles.tsv')   # the role of a kernel that carries no doc comment


def clean_doc(lines):
    """A comment block's text as one line: markers, {@link}, {@code} and backticks removed."""
    text = []
    for ln in lines:
        s = ln.strip()
        s = re.sub(r'^/\*+', '', s)
        s = re.sub(r'\*+/$', '', s)
        s = re.sub(r'^\*+(?!/)', '', s)
        text.append(s.strip())
    t = ' '.join(x for x in text if x)
    t = re.sub(r'\{@link(?:plain)?\s+([^}\s]+)(?:\s+([^}]*))?\}', lambda m: m.group(2) or m.group(1), t)
    t = re.sub(r'\{@code\s+([^}]*)\}', r'\1', t)
    t = t.replace('`', '')
    return re.sub(r'\s+', ' ', t).strip()


def doc_before(lines, i):
    """The block comment directly above the annotations of the declaration at line i."""
    j = i - 1
    top = i
    while j >= 0:
        s = lines[j].strip()
        if s.startswith('@'):
            top = j
        elif not (s.startswith('//') or s == ''):
            break
        j -= 1
    j = top - 1
    if j < 0 or not lines[j].strip().endswith('*/'):
        return ''
    end = j
    while j >= 0 and '/*' not in lines[j]:
        j -= 1
    return clean_doc(lines[j:end + 1]) if j >= 0 else ''


def first_sentence(text):
    m = re.match(r'(.*?[.!?])(?=\s|$)', text)
    # whole, however long: the role has a full-width row of its own
    return m.group(1) if m else text


def read_roles(path):
    roles = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f.read().split('\n')[1:]:
                if line.strip():
                    k, role = line.split('\t', 1)
                    roles[k] = role
    return roles


def scan(repo, roles=None):
    roles = roles or {}
    rows = []
    stage_of = dict(STAGES)
    files = sorted(f for f in os.listdir(os.path.join(repo, GPU)) if f.endswith('.cajeta'))
    decl = re.compile(r'\bvoid\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
    for fn in files:
        with open(os.path.join(repo, GPU, fn), encoding='utf-8') as f:
            lines = f.read().split('\n')
        for i, ln in enumerate(lines):
            if not re.match(r'\s*@Kernel\b', ln):
                continue
            for k in range(i, min(i + 20, len(lines))):
                m = decl.search(lines[k])
                if m:
                    break
            else:
                continue
            if fn not in stage_of:
                raise SystemExit('a kernel in %s, which has no stage: add it to STAGES' % fn)
            annotations = ' '.join(lines[i:k])
            j = i - 1
            while j >= 0 and lines[j].strip().startswith(('@', '//')):
                annotations += ' ' + lines[j]
                j -= 1
            wave = re.search(r'@Wave\s*\(\s*(?:width\s*=\s*)?32\s*\)', annotations) is not None
            rows.append({'kernel': m.group(1), 'file': fn, 'stage': stage_of[fn],
                         'path': '%s/%s' % (GPU, fn), 'line': k + 1, 'wave': wave,
                         'role': first_sentence(doc_before(lines, k)) or roles.get(m.group(1), '')})
    order = [f for f, _ in STAGES]
    rows.sort(key=lambda r: order.index(r['file']))     # stable: source order within a file
    return rows


def github_base(repo):
    url = subprocess.run(['git', '-C', repo, 'remote', 'get-url', 'origin'],
                         capture_output=True, text=True, check=True).stdout.strip()
    m = re.match(r'(?:git@github\.com:|https://github\.com/)([^/]+/[^/]+?)(?:\.git)?$', url)
    if not m:
        raise SystemExit('origin is not a GitHub remote: ' + url)
    return 'https://github.com/%s/blob/main/' % m.group(1)


def version(repo):
    with open(os.path.join(repo, 'cajeta.json'), encoding='utf-8') as f:
        text = f.read()
    m = re.search(r'"version"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else '?'


def esc(s):
    return html.escape(s, quote=True)


def render(rows, base, ver, date):
    out = ['# Catalog: every cajeta-xgboost kernel', '',
           'Generated from the sources of `dev.cajeta.xgboost` %s on %s by `tools/catalog/gen-catalog.py`: '
           'every `@Kernel` in the `gpu` package, its name linked to its source line, the training stage it '
           'serves, whether it declares a 32-lane wave, and under it its role from the kernel\'s own '
           'documentation. The stages, decompositions and the one measured adaptation are described in '
           '[Gradient boosting on the device](xgboost.md).' % (ver, date), '',
           '%d kernels. Every one is integer-exact across backends except the gradient and leaf kernels, '
           'whose float32 results are per row and need no reduction.' % len(rows), '',
           '<table class="kernel-catalog">',
           '<thead><tr><th>kernel</th><th>stage</th><th class="census"><code>@Wave(32)</code></th></tr></thead>',
           '<tbody>']
    for r in rows:
        title = 'gpu/%s:%d' % (r['file'], r['line'])
        out.append('<tr class="kernel"><td class="kernel-name"><a href="%s%s#L%d" title="%s"><code>%s</code></a></td>'
                   '<td>%s</td><td class="census">%s</td></tr>' % (
                       base, esc(r['path']), r['line'], esc(title), esc(r['kernel']), esc(r['stage']),
                       'yes' if r['wave'] else ''))
        out.append('<tr class="kernel-role"><td colspan="3">%s</td></tr>' % esc(r['role']))
    out += ['</tbody>', '</table>', '']
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('output', help='the page to write, e.g. ../../cpp/cajeta/docs/kernels/catalog-xgboost.md')
    ap.add_argument('--repo', default=REPO, help='the cajeta-xgboost checkout (default: this one)')
    ap.add_argument('--date', default=datetime.date.today().isoformat())
    a = ap.parse_args(argv)
    rows = scan(a.repo, read_roles(ROLES))
    undocumented = [r['kernel'] for r in rows if not r['role']]
    if undocumented:
        print('no doc comment and no tools/catalog/roles.tsv row: ' + ', '.join(undocumented), file=sys.stderr)
    page = render(rows, github_base(a.repo), version(a.repo), a.date)
    with open(a.output, 'w', encoding='utf-8') as f:
        f.write(page)
    print('%s: %d kernels' % (a.output, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
