/**
 * @author Shakib
 * @since 18/10/26
 */
import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import {
  composite,
  gptkitLabel,
  groupOf,
  isTransparent,
  nextBackdrop,
  parseBmp,
  printedPictures,
  ratioLabel,
  sizeLabel,
  togglePick,
} from '../hooks/register'

const PANE_PROPS = {
  title: 'Attachments',
  isFocused: true,
  bodyColumns: 120,
  placement: 'dock' as const,
  scroll: { offset: 0, bodyRows: 40 },
  view: {},
}

const MOCKUP_OUTPUT = [
  'plan: ~2 image calls (gpt-image-2.5-flare) ≈ $0.30 list price',
  '/work/design/home/mockup_v1.png  quality=medium size=1152x2496 aspect=ok',
  '/work/design/home/mockup_v2.png  quality=medium size=853x1844 aspect=fitted 853x1844 -> 853x1848 (added 4 px)',
  'design QA (critique):',
  'mockup_v1.png: 1 issues, weight 2',
  '-> /work/design/home/mockup_v1_critique.json, overlay /work/design/home/mockup_v1_critique.png',
].join('\n')

const CRITIQUE_JSON = JSON.stringify({ rounds: [{ issues: [{ severity: 'medium' }], weight: 2 }] })

/** A 2x1 32-bit BMP: one opaque red pixel, one fully transparent pixel. */
function alphaBmp(): Uint8Array {
  const bytes = new Uint8Array(54 + 8)
  const view = new DataView(bytes.buffer)
  bytes[0] = 0x42
  bytes[1] = 0x4d
  view.setUint32(2, bytes.length, true)
  view.setUint32(10, 54, true)
  view.setUint32(14, 40, true)
  view.setInt32(18, 2, true)
  view.setInt32(22, -1, true)
  view.setUint16(26, 1, true)
  view.setUint16(28, 32, true)
  bytes.set([0, 0, 255, 255, 0, 0, 0, 0], 54)

  return bytes
}

function fakeHost(on: On, size = { width: 1152, height: 2496 }) {
  const opened: string[] = []
  const toasts: string[] = []
  on('session.id', () => ({ value: 'test-session' }))
  on('session.cwd', () => ({ value: '/work' }))
  on('session.messages', () => ({ value: [] }) as never)
  mock.env(on, { HOME: '/Users/me' })
  mock.clock(on, { now: 1_000 })
  on('command.register', () => ({ value: { command: 'attachments' } }))
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('fs.write', () => ({ value: undefined }))
  on('fs.exists', ($, e) => ({ value: !/_raw\.png$/.test(e.path) }))
  on('fs.stat', ($, e) => ({
    value: /\.[a-z]+$/i.test(e.path)
      ? { kind: 'file', size: 2048, mtimeMs: 1, isLink: false }
      : { kind: 'dir', size: 0, mtimeMs: 1, isLink: false },
  }) as never)
  on('fs.list', () => ({
    value: [
      { name: 'home.svg', kind: 'file', size: 300, mtimeMs: 1, isLink: false },
      { name: 'search.svg', kind: 'file', size: 300, mtimeMs: 1, isLink: false },
      { name: 'notes.txt', kind: 'file', size: 30, mtimeMs: 1, isLink: false },
      { name: 'old', kind: 'dir', size: 0, mtimeMs: 1, isLink: false },
    ],
  }) as never)
  on('fs.read', ($, e) =>
    /\.json$/.test(e.path)
      ? ({ value: CRITIQUE_JSON } as never)
      : ({ value: { base64: alphaBmp().toBase64() } } as never),
  )
  on('ui.toast', ($, e) => {
    toasts.push(String((e as { text?: string }).text ?? e))

    return { value: undefined }
  })
  on('ui.log', ($, e) => {
    throw new Error(`plugin logged: ${JSON.stringify(e)}`)
  })
  on('process.run', ($, e) => ({
    value: {
      exitCode: 0,
      stdout: e.argv.includes('-g') ? `x\n  pixelWidth: ${size.width}\n  pixelHeight: ${size.height}\n` : '',
      stderr: '',
      isStdoutTruncated: false,
      isStderrTruncated: false,
    },
  }))
  let notify: () => void = () => {}
  on('ui.open', ($, e) => {
    opened.push(e.id)
    notify()

    return { value: { isOpen: true } } as never
  })
  const nextOpen = () => new Promise<void>(resolve => (notify = resolve))

  return { opened, toasts, nextOpen }
}

function mountPane($: Parameters<Parameters<typeof test>[1]>[0]) {
  return $.ui.mount({
    plugin: 'attachment-preview',
    surface: 'terminal',
    component: 'Pane',
    requestId: 'attachments',
    props: PANE_PROPS,
  })
}

test('labels aspect ratios, sizes and gptkit commands', async () => {
  expect(ratioLabel(1152, 2496)).toBe('9:19.5')
  expect(ratioLabel(803, 1740)).toBe('9:19.5')
  expect(ratioLabel(1024, 1536)).toBe('2:3')
  expect(ratioLabel(1536, 1024)).toBe('3:2')
  expect(ratioLabel(1000, 1100)).toBe('0.909')
  expect(sizeLabel(512)).toBe('512 B')
  expect(sizeLabel(4096)).toBe('4 KB')
  expect(sizeLabel(3 * 1024 * 1024)).toBe('3.0 MB')
  expect(gptkitLabel('python3 "$K" mockup "home" --out d')).toBe('gptkit')
  expect(gptkitLabel('python3 ~/skills/gpt-ui-assets/scripts/gptkit.py forge build/home')).toBe('gptkit forge')
})

test('finds the pictures a gptkit command printed, review sheets included', async () => {
  expect(printedPictures(MOCKUP_OUTPUT)).toEqual([
    '/work/design/home/mockup_v1.png',
    '/work/design/home/mockup_v2.png',
    '/work/design/home/mockup_v1_critique.png',
  ])
})

test('keeps at most four picks and cycles backdrops', async () => {
  let keys: string[] = []
  for (const key of ['a', 'b', 'c', 'd', 'e']) keys = togglePick(keys, key)
  expect(keys).toEqual(['b', 'c', 'd', 'e'])
  expect(togglePick(keys, 'c')).toEqual(['b', 'd', 'e'])
  expect(nextBackdrop('checker')).toBe('light')
  expect(nextBackdrop('dark')).toBe('checker')
})

test('reads alpha and lays a transparent asset over each backdrop', async () => {
  const pixels = parseBmp(alphaBmp())!
  expect(Array.from(pixels.alpha ?? [])).toEqual([255, 0])
  expect(isTransparent(pixels)).toBe(true)
  expect(isTransparent({ ...pixels, alpha: new Uint8Array([0, 0]) })).toBe(false)
  expect(Array.from(composite(pixels, 'dark').rgb)).toEqual([255, 0, 0, 28, 28, 28])
  expect(Array.from(composite(pixels, 'light').rgb.slice(3))).toEqual([245, 245, 245])
  expect(Array.from(composite(pixels, 'checker').rgb.slice(3))).toEqual([255, 255, 255])
  const wide = { width: 48, height: 1, rgb: new Uint8Array(48 * 3), alpha: new Uint8Array(48) }
  expect(Array.from(composite(wide, 'checker').rgb.slice(2 * 3, 2 * 3 + 3))).toEqual([204, 204, 204])
})

test('groups a batch for the grid, else the neighbours', async () => {
  const one = (name: string, batch?: string) => ({
    key: name, from: 'claude' as const, name, openPath: `/w/${name}`, png: `/w/${name}`, width: 10, height: 20, batch, at: 0,
  })
  const list = [one('a'), one('b', 'run1'), one('c', 'run1'), one('d')]
  expect(groupOf(list, 1).map(x => x.name)).toEqual(['b', 'c'])
  expect(groupOf(list, 3).map(x => x.name)).toEqual(['a', 'b', 'c', 'd'])
})

test('a gptkit mockup run opens as a grid of its variants with QA', async ($, on) => {
  const host = fakeHost(on)
  on('tool.call', () => ({ result: { stdout: MOCKUP_OUTPUT, stderr: '' } }) as never)
  const opened = host.nextOpen()
  await $.tool.call({
    tool: 'Bash',
    command: 'python3 ~/.claude/skills/gpt-ui-assets/scripts/gptkit.py mockup "home" --out design/home --variants 2',
  } as never)
  await opened

  expect(host.toasts.some(text => /3 pictures from gptkit mockup/.test(text))).toBe(true)
  const pane = await mountPane($)
  expect((await pane.find({ key: 'grid' }))?.text).toBe('[grid]')
  expect(await pane.find({ type: 'Text', text: /gptkit mockup · \d+ · 3 pictures/ })).toBeDefined()
  expect(await pane.find({ type: 'Text', text: /mockup_v1\.png 1\/3 · 1152×2496 · 9:19\.5 · 2 KB · alpha · QA 1 · w2/ })).toBeDefined()
  expect((await pane.find({ key: 'preview' }))?.type).toBe('Raster')
  expect((await pane.find({ key: 'tile1' }))?.type).toBe('Raster')
  expect((await pane.find({ key: 'tile2' }))?.type).toBe('Raster')
  expect(await pane.find({ key: 'list' })).toBeUndefined()
  await pane.unmount()
})

test('picked pictures compare side by side, A to D', async ($, on) => {
  const host = fakeHost(on)
  on('tool.call', () => ({ result: { stdout: MOCKUP_OUTPUT, stderr: '' } }) as never)
  const opened = host.nextOpen()
  await $.tool.call({ tool: 'Bash', command: 'python3 gptkit.py mockup x --out d' } as never)
  await opened

  const pane = await mountPane($)
  await pane.press({ key: 'pick' })
  await pane.press({ key: 'next' })
  await pane.press({ key: 'pick' })
  await pane.press({ key: 'compare' })
  expect((await pane.find({ key: 'compare' }))?.text).toBe('[compare]')
  expect(await pane.find({ type: 'Text', text: /A · mockup_v1\.png/ })).toBeDefined()
  expect(await pane.find({ type: 'Text', text: /B · mockup_v2\.png/ })).toBeDefined()
  expect((await pane.find({ key: 'tile0' }))?.type).toBe('Raster')
  expect((await pane.find({ key: 'tile1' }))?.type).toBe('Raster')
  await pane.press({ key: 'backdrop' })
  expect((await pane.find({ key: 'backdrop' }))?.text).toBe('bg light')
  await pane.unmount()
})

test('a folder of icons is added as one batch and shown as a contact sheet', async ($, on) => {
  const host = fakeHost(on, { width: 24, height: 24 })
  const opened = host.nextOpen()
  const ran = await $.command.run({
    command: 'attachments',
    args: 'add build/home/svg',
    origin: { kind: 'person' },
    presentation: { isFullscreen: true, columns: 160 },
  } as never)
  expect(ran.text).toBe('Added 2 pictures from svg.')
  await opened

  const pane = await mountPane($)
  expect(await pane.find({ type: 'Text', text: /svg · 2 · 2 icons/ })).toBeDefined()
  expect(await pane.find({ type: 'Text', text: /▸1 home\.svg/ })).toBeDefined()
  expect(await pane.find({ type: 'Text', text: / 2 search\.svg/ })).toBeDefined()
  await pane.unmount()

  const cleared = await $.command.run({
    command: 'attachments',
    args: 'clear',
    origin: { kind: 'person' },
    presentation: { isFullscreen: true, columns: 160 },
  } as never)
  expect(cleared.text).toBe('Cleared the attachments.')
})
