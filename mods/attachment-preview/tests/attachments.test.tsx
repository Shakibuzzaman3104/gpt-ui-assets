/**
 * @author Shakib
 * @since 04/10/26
 */
import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

import { fitBox, halfBlockCells, jobsIn, parseBmp } from '../hooks/register'

const RED_BLUE_BMP = tinyBmp()

const PANE_PROPS = {
  title: 'Attachments',
  isFocused: true,
  bodyColumns: 60,
  placement: 'dock' as const,
  scroll: { offset: 0, bodyRows: 30 },
  view: {},
}

function tinyBmp(): Uint8Array {
  // 2x2, 24-bit, top-down: red red / blue blue
  const stride = 8
  const bytes = new Uint8Array(54 + stride * 2)
  const view = new DataView(bytes.buffer)
  bytes[0] = 0x42
  bytes[1] = 0x4d
  view.setUint32(2, bytes.length, true)
  view.setUint32(10, 54, true)
  view.setUint32(14, 40, true)
  view.setInt32(18, 2, true)
  view.setInt32(22, -2, true)
  view.setUint16(26, 1, true)
  view.setUint16(28, 24, true)
  bytes.set([0, 0, 255, 0, 0, 255], 54)
  bytes.set([255, 0, 0, 255, 0, 0], 54 + stride)

  return bytes
}

const PASTED = { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: '/9j/AAAA' } }

function fakeHost(on: On, history: unknown[] = []) {
  const opened: string[] = []
  let notify: () => void = () => {}
  on('session.id', () => ({ value: 'test-session' }))
  on('session.cwd', () => ({ value: '/work' }))
  on('session.messages', () => ({ value: history }) as never)
  mock.env(on, { HOME: '/Users/me' })
  mock.clock(on, { now: 1_000 })
  on('command.register', () => ({ value: { command: 'attachments' } }))
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('fs.write', () => ({ value: undefined }))
  on('fs.exists', () => ({ value: true }))
  on('fs.read', () => ({ value: { base64: RED_BLUE_BMP.toBase64() } }))
  on('ui.toast', () => ({ value: undefined }))
  on('ui.log', ($, e) => {
    throw new Error(`plugin logged: ${JSON.stringify(e)}`)
  })
  on('process.run', ($, e) => {
    const isProbe = e.argv.includes('-g')

    return {
      value: {
      exitCode: 0,
      stdout: isProbe ? 'x.png\n  pixelWidth: 1024\n  pixelHeight: 512\n' : '',
      stderr: '',
      isStdoutTruncated: false,
      isStderrTruncated: false,
      },
    }
  })
  on('ui.open', ($, e) => {
    opened.push(e.id)
    notify()

    return { value: { isOpen: true } } as never
  })
  const nextOpen = () => new Promise<void>(resolve => (notify = resolve))

  return { opened, nextOpen }
}

test('parses a top-down BMP and packs half-block cells', async () => {
  const pixels = parseBmp(RED_BLUE_BMP)
  expect(pixels?.width).toBe(2)
  expect(pixels?.height).toBe(2)
  expect(Array.from(pixels?.rgb.slice(0, 3) ?? [])).toEqual([255, 0, 0])
  expect(Array.from(pixels?.rgb.slice(6, 9) ?? [])).toEqual([0, 0, 255])

  const cells = Uint8Array.fromBase64(halfBlockCells(pixels!, 1, 1))
  const words = new Uint32Array(cells.buffer)
  expect(Array.from(words)).toEqual([0x2580, 0xff0000, 0x0000ff])
})

test('fits a picture into the pane keeping its aspect', async () => {
  expect(fitBox(1024, 512, 80, 30)).toEqual({ columns: 80, rows: 20 })
  expect(fitBox(1024, 512, 200, 30)).toEqual({ columns: 120, rows: 30 })
})

test('splits pasted and tool-result media out of a row', async () => {
  const fromPrompt = jobsIn([{ type: 'text', text: 'hi' }, PASTED], 'you')
  expect(fromPrompt).toHaveLength(1)
  expect(fromPrompt[0]).toMatchObject({ from: 'you', mediaType: 'image/jpeg', base64: '/9j/AAAA' })

  const fromTool = jobsIn([{ type: 'tool_result', tool_use_id: 't1', content: [PASTED] }], 'claude', 'mcp__shot')
  expect(fromTool[0]).toMatchObject({ from: 'claude', via: 'mcp__shot' })
  expect(fromTool[0]?.key).toBe(fromPrompt[0]?.key)
})

test('a pasted image is captured and previewed on every surface', async ($, on) => {
  const host = fakeHost(on, [{ role: 'user', content: [{ type: 'text', text: 'look' }, PASTED] }])
  const opened = host.nextOpen()
  await $.session.start({ cwd: '/work', surface: 'terminal', isInteractive: true })
  await opened

  expect(host.opened).toContain('attachments')

  const terminal = await $.ui.mount({
    plugin: 'attachment-preview',
    surface: 'terminal',
    component: 'Pane',
    requestId: 'attachments',
    props: PANE_PROPS,
  })
  expect(await terminal.find({ type: 'Text', text: /1\/1 · 1024×512/ })).toBeDefined()
  expect((await terminal.find({ key: 'preview' }))?.type).toBe('Raster')
  expect((await terminal.find({ key: 'list' }))?.text).toMatch(/You/)
  await terminal.press({ key: 'mode' })
  expect((await terminal.find({ key: 'preview' }))?.type).toBe('Image')
  await terminal.unmount()

  for (const surface of ['desktop', 'mobile'] as const) {
    const remote = await $.ui.mount({
      plugin: 'attachment-preview',
      surface,
      component: 'Pane',
      requestId: 'attachments',
      props: PANE_PROPS,
    })
    expect((await remote.find({ key: 'list' }))?.text).toMatch(/pasted/)
    await remote.unmount()
  }
})

test('a file Claude sends is listed as a clickable link', async ($, on) => {
  const host = fakeHost(on)
  on('tool.call', () => ({
    result: { attachments: [{ path: '/work/report.pdf', size: 10, isImage: false }] },
  }) as never)
  const opened = host.nextOpen()
  await $.tool.call({ tool: 'SendUserFile', files: ['report.pdf'], status: 'normal' } as never)
  await opened

  const pane = await $.ui.mount({
    plugin: 'attachment-preview',
    surface: 'terminal',
    component: 'Pane',
    requestId: 'attachments',
    props: PANE_PROPS,
  })
  expect(await pane.find({ type: 'Text', text: /Claude · SendUserFile/ })).toBeDefined()
  expect((await pane.find({ key: 'list' }))?.props.text).toMatch(/\[report\.pdf\]\(file:\/\/\/work\/report\.pdf\)/)
  await pane.press({ key: 'list', link: { href: 'file:///work/report.pdf' } })
  await pane.unmount()
})

test('other plugins add captures, and two pictures compare side by side', async ($, on) => {
  const host = fakeHost(on)
  for (const name of ['before.png', 'after.png']) {
    const opened = host.nextOpen()
    const ran = await $.command.run({
      command: 'attachments',
      args: `add /Users/me/shots/${name}`,
      origin: { kind: 'plugin', name: 'device-deck' },
      presentation: { isFullscreen: true, columns: 160 },
    } as never)
    expect(ran.text).toBe(`Added ${name} to attachments.`)
    await opened
  }

  const pane = await $.ui.mount({
    plugin: 'attachment-preview',
    surface: 'terminal',
    component: 'Pane',
    requestId: 'attachments',
    props: PANE_PROPS,
  })
  expect(await pane.find({ type: 'Text', text: /You · device-deck/ })).toBeDefined()
  expect(await pane.find({ type: 'Text', text: /after\.png/ })).toBeDefined()
  expect(await pane.find({ key: 'before' })).toBeUndefined()

  await pane.press({ key: 'compare' })
  expect((await pane.find({ key: 'before' }))?.type).toBe('Raster')
  expect((await pane.find({ key: 'preview' }))?.type).toBe('Raster')
  expect(await pane.find({ type: 'Text', text: /before · before\.png/ })).toBeDefined()
  await pane.unmount()
})
