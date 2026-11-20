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
        assert doc['color']['role']['light']['primary']['$value'] == '#663399' and 'dark' in doc['color']['role']


def test_non_ui_colors_do_not_become_primary():
    tm = model()
    tm['colors'] = [dict(tm['colors'][2], name='Avatar', usage='illustration', hex='#F4A261')]
    assert 'primary' not in tokens._color_roles(tm)[1]


def test_specific_roles_win():
    tm = model()
    tm['colors'] = [dict(tm['colors'][2], name='On primary container', hex='#123456'),
                    dict(tm['colors'][2], name='Primary container', hex='#ABCDEF')]
    measured = tokens._color_roles(tm)[1]
    assert measured['onPrimaryContainer'] == '#123456' and measured['primaryContainer'] == '#ABCDEF'
    assert 'onPrimary' not in measured


def test_role_alpha_emitters_and_audit():
    tm = model()
    tm['colors'].append(dict(tm['colors'][2], name='On primary', hex='#FFFFFF', alpha=0.5, kt='OnPrimary', sw='onPrimary', dart='onPrimary', css='on-primary'))
    args = SimpleNamespace(prefix='Acme', package='com.acme.ui.theme', r_package=None, author=None)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tokens._emit_compose(tm, args, root / 'compose')
        tokens._emit_swift(tm, args, root / 'swift')
        tokens._emit_flutter(tm, args, root / 'flutter')
        tokens._emit_web(tm, args, root / 'web')
        assert 'Color(0x80FFFFFF)' in (root / 'compose/Color.kt').read_text()
        assert '0x80FFFFFF' in (root / 'flutter/acme_tokens.dart').read_text()
        assert '#FFFFFF80' in (root / 'web/tokens.css').read_text()
        asset = json.loads((root / 'swift/Colors.xcassets/onPrimary.colorset/Contents.json').read_text())
        assert asset['colors'][0]['color']['components']['alpha'] == '0.502'
        assert system.contrast_ratio(tm['schemes']['light']['onPrimary'], '#FFFFFF') == 1


def test_text_styles_follow_roles():
    spec = {'typography': [{'role': 'body', 'size_px': 14, 'color': '#663399'}]}
    palette = [dict(c, hex_guess=c['hex']) for c in model()['colors']]
    tm = tokens._token_model(spec, palette, {}, None)
    args = SimpleNamespace(prefix='Acme', package='com.acme.ui.theme', r_package=None, author=None)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tokens._emit_swift(tm, args, root / 'swift')
        tokens._emit_flutter(tm, args, root / 'flutter')
        tokens._emit_web(tm, args, root / 'web')
        assert 'color: AcmeColorRoles.primary' in (root / 'swift/AcmeTokens.swift').read_text()
        assert 'copyWith(color: scheme.primary)' in (root / 'flutter/acme_tokens.dart').read_text()
        assert 'color: var(--color-primary)' in (root / 'web/tokens.css').read_text()


def test_flutter_preserves_background_and_surface():
    tm = model()
    tm['colors'][1]['hex'] = '#EEEEEE'
    args = SimpleNamespace(prefix='Acme')
    with tempfile.TemporaryDirectory() as tmp:
        tokens._emit_flutter(tm, args, Path(tmp))
        dart = (Path(tmp) / 'acme_tokens.dart').read_text()
        assert 'surface: Color(0xFFFFFFFF)' in dart and 'surfaceContainer: Color(0xFFEEEEEE)' in dart


def test_apple_bold_contrast():
    tm = model()
    tm['colors'][3]['hex'] = '#777777'
    tm['types'] = [{'size': 14, 'weight': 700, 'role': 'body', 'line_height': 20, 'tracking': 0, 'color': tm['colors'][3]}]
    _, audit = tokens._platform_model(tm, 'ios', False)
    assert not any(i['fg'] == 'onSurface' and i['bg'] == 'surface' and i['mode'] == 'light' for i in audit['contrast'])


def test_dtcg_light_primitive_survives():
    tm = model()
    tm['colors'].append(dict(tm['colors'][0], name='Light', css='light'))
    with tempfile.TemporaryDirectory() as tmp:
        tokens._emit_dtcg(tm, Path(tmp))
        doc = json.loads((Path(tmp) / 'tokens.tokens.json').read_text())
        assert doc['color']['primitive']['light']['$value'] == '#FFFFFF'
        assert 'primary' in doc['color']['role']['light']


def test_snap_composited_alpha():
    spec = {'palette': [{'name': 'Screen background', 'hex': '#808080', 'alpha': 0.5}], 'components': []}
    result = tokens._snap_palette(spec, Image.new('RGB', (20, 20), '#808080'), {})[0]
    assert result['snapped'] and result['alpha'] == 1.0 and result['note'] == 'composited: sampled over its background'
    spec['palette'][0]['hex'] = '#FF0000'
    result = tokens._snap_palette(spec, Image.new('RGB', (20, 20), '#808080'), {})[0]
    assert not result['snapped'] and result['alpha'] == 0.5


def test_secondary_text_and_label_roles():
    tm = model()
    for name, h in [('Secondary text', '#555555'), ('Button label', '#FFFFFF')]:
        ident = tokens._ident(name, 'camel')
        tm['colors'].append({'name': name, 'usage': '', 'hex': h, 'alpha': 1.0, 'kt': tokens._ident(name),
                             'sw': ident, 'dart': ident, 'css': tokens._ident(name, 'kebab')})
    _, measured, _ = tokens._color_roles(tm)
    assert measured.get('onSurfaceVariant') == '#555555' and measured.get('secondary') != '#555555'
    label = next(c for c in tm['colors'] if c['name'] == 'Button label')
    assert tokens._text_role(tm, {'color': label}) == 'onPrimary'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
