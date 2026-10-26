"""Tests for gen-catalog.py.

    python3 -m unittest discover -s tools/catalog

The row test regenerates the page from this checkout's sources and compares
every entry (kernel, stage, source line, @Wave(32), role) with the committed
page in the cajeta repository (CATALOG_PAGE, default
../../cpp/cajeta/docs/xpu/catalog-xgboost.md). It reads both the old
markdown-table page and the HTML-table page the generator writes now.
"""
import html
import importlib.util
import os
import re
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PAGE = os.environ.get('CATALOG_PAGE', os.path.normpath(
    os.path.join(REPO, '..', '..', 'cpp', 'cajeta', 'docs', 'xpu', 'catalog-xgboost.md')))

spec = importlib.util.spec_from_file_location('gen_catalog', os.path.join(HERE, 'gen-catalog.py'))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def unhtml(s):
    return html.unescape(re.sub(r'<[^>]+>', '', s))


def parse_page(text):
    entries = []
    for line in text.split('\n'):
        if line.startswith('| `'):
            c = [x.strip() for x in re.split(r'(?<!\\)\|', line.strip())[1:-1]]
            entries.append(dict(kernel=c[0].strip('`'), stage=c[1], source=c[2].strip('`'),
                                wave=c[3] == 'yes', role=c[4]))
    rows = re.findall(r'<tr class="(kernel|kernel-role)">(.*?)</tr>', text, re.S)
    for (cls, tr), (ncls, nxt) in zip(rows[0::2], rows[1::2]):
        assert cls == 'kernel' and ncls == 'kernel-role', tr
        tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)
        entries.append(dict(kernel=unhtml(tds[0]), stage=unhtml(tds[1]),
                            source=html.unescape(re.search(r'title="([^"]*)"', tds[0]).group(1)),
                            wave=unhtml(tds[2]) == 'yes',
                            role=unhtml(re.search(r'<td[^>]*>(.*?)</td>', nxt, re.S).group(1))))
    return entries


class CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = os.path.join(cls.tmp.name, 'catalog-xgboost.md')
        gen.main([out, '--date', '2026-10-24'])
        with open(out, encoding='utf-8') as f:
            cls.page = f.read()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rows_match_committed_page(self):
        if not os.path.exists(PAGE):
            self.skipTest('no committed page at ' + PAGE)
        with open(PAGE, encoding='utf-8') as f:
            want = parse_page(f.read())
        got = parse_page(self.page)
        self.assertEqual([e['kernel'] for e in want], [e['kernel'] for e in got])
        for w, g in zip(want, got):
            self.assertEqual(w, g, w['kernel'])

    def test_committed_page_is_at_the_xpu_path(self):
        if not os.path.isdir(os.path.join(REPO, '..', '..', 'cpp', 'cajeta', 'docs')):
            self.skipTest('no sibling cajeta checkout')
        self.assertTrue(os.path.exists(PAGE), 'no committed page at ' + PAGE)

    def test_default_output_is_the_xpu_page(self):
        self.assertEqual(os.path.normpath(gen.DEFAULT_OUTPUT), os.path.normpath(
            os.path.join(REPO, '..', '..', 'cpp', 'cajeta', 'docs', 'xpu', 'catalog-xgboost.md')))

    def test_preamble_links_the_xpu_page(self):
        self.assertIn('[Gradient boosting on the device](kernels-xgboost.md)', self.page)
        self.assertNotIn('](xgboost.md)', self.page)

    def test_no_markdown_table(self):
        self.assertEqual([l for l in self.page.split('\n') if re.match(r'\s*\|.*\|\s*$', l)], [])

    def test_three_grid_columns_and_a_role_row_per_kernel(self):
        heads = re.findall(r'<thead>(.*?)</thead>', self.page, re.S)
        self.assertEqual(len(heads), 1)
        self.assertEqual([unhtml(h) for h in re.findall(r'<th[^>]*>(.*?)</th>', heads[0])],
                         ['kernel', 'stage', '@Wave(32)'])
        rows = re.findall(r'<tr class="([^"]*)"', self.page)
        self.assertEqual(rows, ['kernel', 'kernel-role'] * (len(rows) // 2))
        self.assertIn('colspan="3"', self.page)

    def test_no_blank_line_inside_the_table(self):
        for t in re.findall(r'<table.*?</table>', self.page, re.S):
            self.assertNotIn('\n\n', t)

    def test_kernel_links_point_at_a_source_line(self):
        hrefs = re.findall(r'href="([^"]*)"', self.page)
        self.assertTrue(hrefs)
        for href in hrefs:
            if 'github.com' in href:
                self.assertRegex(href, r'^https://github\.com/[^/]+/cajeta-xgboost/blob/main/src/main/cajeta/dev/cajeta/xgboost/gpu/\w+\.cajeta#L\d+$')


class DocTest(unittest.TestCase):
    def test_first_sentence_is_whole_however_long(self):
        self.assertEqual(gen.first_sentence('One. Two.'), 'One.')
        long = 'x' * 400 + '.'
        self.assertEqual(gen.first_sentence(long + ' More.'), long)


if __name__ == '__main__':
    unittest.main()
