"""Offline colour token checks: python3 tests/test_tokens_color.py."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gptkit import tokens, system


def model():
    colors = []
    for name, h in [('Screen background', '#FFFFFF'), ('Surface', '#FFFFFF'),
                    ('Primary', '#663399'), ('Primary text', '#999999')]:
        ident = tokens._ident(name, 'camel')
        colors.append({'name': name, 'usage': '', 'hex': h, 'alpha': 1.0,
                       'kt': tokens._ident(name), 'sw': ident, 'dart': ident,
                       'css': tokens._ident(name, 'kebab')})
    return {'colors': colors, 'types': [], 'radii': [], 'spacing': [], 'families': [], 'px_to_dp': 1}


def test_color_emitters():
    tm = model()
    mode, measured, schemes = tokens._color_roles(tm)
    assert mode == 'light' and schemes[mode]['primary'] == measured['primary'] == '#663399'
    assert all(schemes['light'][r] != schemes['dark'][r] for r in ('surface', 'primary', 'onSurface'))
    issues = system.check_pairs(schemes[mode], 'web', mode=mode)
    assert any(i['fg'] == 'onSurface' and i['bg'] == 'surface' and i['ratio'] < 4.5 for i in issues)
    a = SimpleNamespace(prefix='Acme', package='com.acme.ui.theme', r_package=None, author=None)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tokens._emit_compose(tm, a, root / 'compose')
        tokens._emit_swift(tm, a, root / 'swift')
        tokens._emit_flutter(tm, a, root / 'flutter')
        tokens._emit_web(tm, a, root / 'web')
        tokens._emit_dtcg(tm, root)
        compose = (root / 'compose' / 'Color.kt').read_text()
        assert 'lightColorScheme(' in compose and 'darkColorScheme(' in compose and 'AcmeColors.Primary' in compose
        css = (root / 'web' / 'tokens.css').read_text()
        assert 'prefers-color-scheme: dark' in css and ':root[data-theme=dark]' in css
        asset = json.loads((root / 'swift' / 'Colors.xcassets' / 'primary.colorset' / 'Contents.json').read_text())
        assert len(asset['colors']) == 2 and asset['colors'][1]['appearances'][0]['value'] == 'dark'
        assert 'brightness: Brightness.dark' in (root / 'flutter' / 'acme_tokens.dart').read_text()
        doc = json.loads((root / 'tokens.tokens.json').read_text())
        assert doc['color']['light']['primary']['$value'] == '#663399' and 'dark' in doc['color']


def test_snap_composited_alpha():
    spec = {'palette': [{'name': 'Screen background', 'hex': '#808080', 'alpha': 0.5}], 'components': []}
    result = tokens._snap_palette(spec, Image.new('RGB', (20, 20), '#808080'), {})[0]
    assert result['snapped'] and result['alpha'] == 1.0 and result['note'] == 'composited: sampled over its background'
    spec['palette'][0]['hex'] = '#FF0000'
    result = tokens._snap_palette(spec, Image.new('RGB', (20, 20), '#808080'), {})[0]
    assert not result['snapped'] and result['alpha'] == 0.5


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
