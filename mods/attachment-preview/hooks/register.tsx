/**
 * @author Shakib
 * @since 04/10/26
 */
import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { Attachment, AttachmentFrom, Backdrop, PreviewMode, QaSummary, ViewMode } from '../types'

const PANE = 'attachments'
const TITLE = 'Attachments'
const items = atom({ plugin: 'attachment-preview', key: 'items' } as const, [] as Attachment[])
const selected = atom({ plugin: 'attachment-preview', key: 'selected' } as const, 0)
const mode = atom({ plugin: 'attachment-preview', key: 'mode' } as const, 'auto' as PreviewMode)
const view = atom({ plugin: 'attachment-preview', key: 'view' } as const, 'single' as ViewMode)
const picked = atom({ plugin: 'attachment-preview', key: 'picked' } as const, [] as string[])
const backdrop = atom({ plugin: 'attachment-preview', key: 'backdrop' } as const, 'checker' as Backdrop)

const IMAGE_EXT = /\.(png|jpe?g|gif|webp|heic|heif|bmp|tiff?)$/i
const PICTURE_EXT = /\.(png|jpe?g|gif|webp|heic|heif|bmp|tiff?|svg)$/i
const PRINTED_PATH = /(\/[^\s'"`()<>|;,]+\.(?:png|jpe?g|webp|gif|svg))(?![\w/])/gi
const EXT_OF: Record<string, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/gif': 'gif',
  'image/webp': 'webp',
  'application/pdf': 'pdf',
  'text/plain': 'txt',
}
const HALF_BLOCK = 0x2580
const THUMB_PIXELS = 320
const PREVIEW_PIXELS = 640
const LIST_ROWS = 8
const MAX_PICKED = 4
const ICON_PIXELS = 256
const ICON_TILE_COLUMNS = 14
const BACKDROPS: Backdrop[] = ['checker', 'light', 'dark']
const RATIOS: [string, number][] = [
  ['9:19.5', 9 / 19.5],
  ['9:16', 9 / 16],
  ['2:3', 2 / 3],
  ['3:4', 3 / 4],
  ['1:1', 1],
  ['4:3', 4 / 3],
  ['3:2', 3 / 2],
  ['16:10', 16 / 10],
  ['16:9', 16 / 9],
]

type Job = {
  key: string
  from: AttachmentFrom
  via?: string
  name?: string
  batch?: string
  batchSize?: number
} & ({ path: string } | { base64: string; mediaType: string })

type Pixels = { width: number; height: number; rgb: Uint8Array; alpha?: Uint8Array }

type Block = { type: string; [field: string]: unknown }

type Tile = { item: Attachment; label: string; key: string }

const queued = new Set<string>()
const pixelCache = new Map<string, Pixels>()
const readPathOf = new Map<string, string>()
let queue: Promise<void> = Promise.resolve()

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'attachments',
      description: 'Preview, compare and grid the images and assets of this session (add <file|folder>, grid, clear)',
    })
    const started = await next(e)
    void backfill($)

    return started
  })

  on('command.run', { command: 'attachments' }, async ($, e) => {
    const args = e.args.trim()
    const added = /^add\s+(.+)$/s.exec(args)
    if (added?.[1]) {
      const path = added[1].trim()
      const via = e.origin.kind === 'plugin' ? e.origin.name : undefined
      const folder = await listPictures($, path)
      if (folder) {
        const batch = `${baseName(path)} · ${folder.length}`
        for (const file of folder) {
          enqueue($, { key: `path:${file}`, from: 'you', via: via ?? 'folder', path: file, batch, batchSize: folder.length })
        }

        return { text: folder.length === 0 ? `No pictures in ${baseName(path)}.` : `Added ${folder.length} pictures from ${baseName(path)}.` }
      }
      enqueue($, { key: `path:${path}`, from: 'you', via, path })

      return { text: `Added ${baseName(path)} to attachments.` }
    }
    if (args === 'clear') {
      await update($, items, () => [])
      await update($, picked, () => [])
      await update($, selected, () => 0)
      queued.clear()

      return { text: 'Cleared the attachments.' }
    }
    if (args === 'grid' || args === 'compare' || args === 'single') await update($, view, () => args)
    const list = await read($, items)
    await $.ui.open({ id: PANE, title: TITLE, focus: true })

    return { text: list.length === 0 ? 'No attachments yet.' : `${list.length} attachment(s).` }
  })

  on('tool.call', async ($, e, next) => {
    if (e.tool === 'Read') readPathOf.set(e.tool_use_id, e.file_path)
    const ran = await next(e)
    if (e.tool === 'SendUserFile' && 'result' in ran && ran.result && !ran.isError) {
      const sent = ran.result as { attachments?: { path: string }[] }
      for (const file of sent.attachments ?? []) {
        enqueue($, { key: `path:${file.path}`, from: 'claude', via: 'SendUserFile', path: file.path })
      }
    }
    if (e.tool === 'Bash' && isGptkit(e.command) && 'result' in ran && ran.result && !ran.isError) {
      const output = ran.result as { stdout?: string; stderr?: string }
      captureGptkit($, e.command, `${output.stdout ?? ''}\n${output.stderr ?? ''}`, await $.clock.now())
    }

    return ran
  })

  on('session.append', async ($, e, next) => {
    const stored = await next(e)
    if (e.agentId !== undefined) return stored
    const from: AttachmentFrom | undefined =
      e.door === 'prompt' ? 'you' : e.door === 'tool-result' ? 'claude' : undefined
    if (from !== undefined) {
      const via = e.origin.kind === 'tool' ? e.origin.tool : undefined
      for (const job of jobsIn(e.message.content as Block[], from, via)) enqueue($, job)
    }

    return stored
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const list = await read($, items)
    const index = Math.min(Math.max(0, await read($, selected)), Math.max(0, list.length - 1))
    const chosenMode = await read($, mode)
    const chosenView = await read($, view)
    const pickedKeys = await read($, picked)
    const chosenBackdrop = await read($, backdrop)
    const { Box, Text, Button, Markdown } = $.ui.resolve(e)
    const item = list[index]

    if (!item) {
      return (
        <Box flexDirection="column">
          <Text dimColor>No attachments yet. Paste an image, run a gptkit command, or /attachments add a folder.</Text>
        </Box>
      )
    }

    const columns = Math.max(10, e.props.bodyColumns)
    const listRows = chosenView === 'grid' ? 0 : Math.min(list.length, LIST_ROWS) + 1
    const previewRows = Math.max(6, (e.props.scroll?.bodyRows ?? 24) - 4 - listRows)
    const usePixels = chosenMode === 'pixels' || (chosenMode === 'auto' && (await hasPixelGraphics($)))
    const select = (to: number) => update($, selected, () => (to + list.length) % list.length)
    const who = `${item.from === 'you' ? 'You' : 'Claude'}${item.via ? ` · ${item.via}` : ''}`
    const isPicked = pickedKeys.includes(item.key)

    const drawPicture = async (shown: Attachment, key: string, maxColumns: number, maxRows: number) => {
      if (e.surface !== 'terminal') return <Text dimColor>Pictures preview in the terminal.</Text>
      if (!shown.png || !shown.width || !shown.height) {
        return <Text dimColor>No preview for this file. Open it to view.</Text>
      }
      const { Image, Raster } = $.ui.resolve(e)
      const box = fitBox(shown.width, shown.height, maxColumns, maxRows)
      if (usePixels && !shown.hasAlpha) {
        return (
          <Image key={key} source={{ file: shown.png, format: 'png' }} columns={box.columns} rows={box.rows} alt={shown.name} />
        )
      }
      const source = usePixels ? (shown.preview ?? shown.thumb) : shown.thumb
      const pixels = source ? await loadPixels($, source) : undefined
      if (!pixels) return <Text dimColor>No preview for this file. Open it to view.</Text>
      const flat = shown.hasAlpha ? composite(pixels, chosenBackdrop) : pixels
      if (usePixels) {
        return (
          <Image
            key={key}
            source={{ rgba: toRgba(flat).toBase64(), width: flat.width, height: flat.height }}
            columns={box.columns}
            rows={box.rows}
            alt={shown.name}
          />
        )
      }

      return <Raster key={key} columns={box.columns} rows={box.rows} cells={halfBlockCells(flat, box.columns, box.rows)} />
    }

    const drawTiles = async (tiles: Tile[], perRow: number, tileRows: number) => {
      const tileColumns = Math.max(4, Math.floor((columns - (perRow - 1)) / perRow))
      const rows: Tile[][] = []
      for (let at = 0; at < tiles.length; at += perRow) rows.push(tiles.slice(at, at + perRow))
      const drawn = []
      for (const [rowAt, row] of rows.entries()) {
        const cells = []
        for (const tile of row) {
          cells.push(
            <Box key={`${tile.key}Box`} flexDirection="column" width={tileColumns}>
              {await drawPicture(tile.item, tile.key, tileColumns, tileRows)}
              <Text dimColor wrap="truncate-end">
                {tile.label}
              </Text>
            </Box>,
          )
        }
        drawn.push(
          <Box key={`row${rowAt}`} flexDirection="row" gap={1}>
            {cells}
          </Box>,
        )
      }

      return <Box flexDirection="column">{drawn}</Box>
    }

    let body
    if (chosenView === 'compare') {
      const chosen = pickedKeys.map(key => list.find(one => one.key === key)).filter(isPicture)
      if (chosen.length >= 2) {
        const tiles = chosen.map((one, at) => ({
          item: one,
          key: `tile${at}`,
          label: `${String.fromCharCode(65 + at)} · ${one.name}${detailsOf(one, true)}`,
        }))
        body = await drawTiles(tiles, tiles.length, previewRows - 1)
      } else {
        const before = list.slice(0, index).reverse().find(isPicture)
        body = before ? (
          await drawTiles(
            [
              { item: before, key: 'before', label: `before · ${before.name}` },
              { item, key: 'preview', label: `after · ${item.name}` },
            ],
            2,
            previewRows - 1,
          )
        ) : (
          <Box flexDirection="column">
            <Text dimColor>No earlier picture to compare with. Pick 2-4 with x, then compare.</Text>
            {await drawPicture(item, 'preview', columns, previewRows - 1)}
          </Box>
        )
      }
    } else if (chosenView === 'grid') {
      const group = groupOf(list, index)
      const isIcons = group.every(one => isIconSized(one))
      const perRow = isIcons
        ? Math.max(1, Math.floor((columns + 1) / (ICON_TILE_COLUMNS + 1)))
        : Math.max(1, Math.min(group.length, columns < 60 ? 2 : 4))
      const minTileRows = isIcons ? 5 : 10
      const gridRows = Math.max(1, Math.floor(previewRows / (minTileRows + 1)))
      const page = perRow * gridRows
      const at = Math.max(0, group.indexOf(item))
      const first = Math.floor(at / page) * page
      const shown = group.slice(first, first + page)
      const usedRows = Math.max(1, Math.ceil(shown.length / perRow))
      const tileRows = Math.max(3, Math.floor(previewRows / usedRows) - 1)
      const tiles = shown.map(one => ({
        item: one,
        key: one === item ? 'preview' : `tile${list.indexOf(one)}`,
        label: `${one === item ? '▸' : pickedKeys.includes(one.key) ? '★' : ' '}${list.indexOf(one) + 1} ${one.name}`,
      }))
      const pageNote = group.length > page ? ` · page ${Math.floor(first / page) + 1}/${Math.ceil(group.length / page)}` : ''
      body = (
        <Box flexDirection="column">
          <Text dimColor wrap="truncate-end">
            {item.batch ?? 'nearby'} · {group.length} {isIcons ? 'icons' : 'pictures'}
            {pageNote}
          </Text>
          {await drawTiles(tiles, perRow, tileRows)}
        </Box>
      )
    } else {
      body = await drawPicture(item, 'preview', columns, previewRows)
    }

    const firstListed = Math.max(0, Math.min(index - Math.floor(LIST_ROWS / 2), list.length - LIST_ROWS))
    const listText = list
      .slice(firstListed, firstListed + LIST_ROWS)
      .map((one, offset) => {
        const at = firstListed + offset
        const mark = at === index ? '▸' : pickedKeys.includes(one.key) ? '★' : ' '
        const by = one.from === 'you' ? 'You' : 'Claude'

        return `${mark} ${at + 1}. **${by}** [${escapeLabel(one.name)}](${fileUrl(one.openPath)})`
      })
      .join('\n')

    return (
      <Box flexDirection="column">
        <Text wrap="truncate-end">
          <Text bold>{who}</Text> {item.name}
          <Text dimColor>
            {' '}
            {index + 1}/{list.length}
            {detailsOf(item, false)}
          </Text>
        </Text>
        <Box flexDirection="row" gap={1}>
          <Button key="prev" hotkey="p" onPress={() => select(index - 1)}>
            ◀ prev
          </Button>
          <Button key="next" hotkey="n" onPress={() => select(index + 1)}>
            next ▶
          </Button>
          <Button key="open" hotkey="o" variant="primary" onPress={() => openFile($, item.openPath)}>
            open
          </Button>
          <Button key="reveal" hotkey="r" onPress={() => openFile($, item.openPath, true)}>
            reveal
          </Button>
          <Button key="pick" hotkey="x" onPress={() => update($, picked, keys => togglePick(keys, item.key))}>
            {isPicked ? '★ picked' : '☆ pick'}
          </Button>
        </Box>
        <Box flexDirection="row" gap={1}>
          <Button key="single" hotkey="s" onPress={() => update($, view, () => 'single')}>
            {chosenView === 'single' ? '[single]' : 'single'}
          </Button>
          <Button key="compare" hotkey="c" onPress={() => update($, view, was => (was === 'compare' ? 'single' : 'compare'))}>
            {chosenView === 'compare' ? '[compare]' : 'compare'}
          </Button>
          <Button key="grid" hotkey="g" onPress={() => update($, view, was => (was === 'grid' ? 'single' : 'grid'))}>
            {chosenView === 'grid' ? '[grid]' : 'grid'}
          </Button>
          <Button key="backdrop" hotkey="b" onPress={() => update($, backdrop, was => nextBackdrop(was))}>
            bg {chosenBackdrop}
          </Button>
          <Button key="mode" hotkey="m" onPress={() => update($, mode, () => (usePixels ? 'blocks' : 'pixels'))}>
            {usePixels ? 'blocks' : 'pixels'}
          </Button>
        </Box>
        {body}
        {chosenView === 'grid' ? null : (
          <Markdown
            key="list"
            text={listText}
            onLinkPress={link => {
              const at = list.findIndex(one => fileUrl(one.openPath) === link.href)
              if (at >= 0) void select(at)
            }}
          />
        )}
      </Box>
    )
  })
}

function enqueue($: EngineInterface, job: Job) {
  if (queued.has(job.key)) return
  queued.add(job.key)
  queue = queue.then(() => ingest($, job)).catch(error => {
    $.ui.log(`attachment-preview: ${String(error)}`, { to: 'debug' })
  })
}

async function ingest($: EngineInterface, job: Job) {
  const existing = await read($, items)
  if (existing.some(one => one.key === job.key)) return

  const dir = await cacheDir($)
  const id = `${existing.length + 1}-${fingerprint(job.key)}`
  await $.process.run(['/bin/mkdir', '-p', dir])

  let source: string
  let isImage: boolean
  if ('path' in job) {
    source = job.path.startsWith('/') ? job.path : `${await $.session.cwd()}/${job.path}`
    isImage = IMAGE_EXT.test(source)
  } else {
    const ext = EXT_OF[job.mediaType] ?? 'bin'
    source = `${dir}/${id}.${ext}`
    await $.fs.write(`${dir}/${id}.b64`, job.base64)
    await $.process.run(['/usr/bin/base64', '-d', '-i', `${dir}/${id}.b64`, '-o', source])
    await $.process.run(['/bin/rm', '-f', `${dir}/${id}.b64`])
    isImage = job.mediaType.startsWith('image/')
  }
  if (!(await $.fs.exists(source))) return

  let png: string | undefined
  if (isImage) {
    png = source.toLowerCase().endsWith('.png') ? source : `${dir}/${id}.png`
    if (png !== source) await $.process.run(['/usr/bin/sips', '-s', 'format', 'png', source, '--out', png])
  } else {
    // Quick Look renders SVGs, PDFs and documents to a PNG thumbnail
    await $.process.run(['/usr/bin/qlmanage', '-t', '-s', '1024', '-o', dir, source], { timeoutMs: 20_000 })
    png = `${dir}/${baseName(source)}.png`
  }
  if (!(await $.fs.exists(png))) png = undefined

  let thumb: string | undefined
  let preview: string | undefined
  let width: number | undefined
  let height: number | undefined
  let hasAlpha = false
  if (png) {
    thumb = `${dir}/${id}.bmp`
    await $.process.run(['/usr/bin/sips', '-Z', String(THUMB_PIXELS), '-s', 'format', 'bmp', png, '--out', thumb])
    if (!(await $.fs.exists(thumb))) thumb = undefined
    const probe = await $.process.run(['/usr/bin/sips', '-g', 'pixelWidth', '-g', 'pixelHeight', png])
    width = Number(/pixelWidth:\s*(\d+)/.exec(probe.stdout)?.[1]) || undefined
    height = Number(/pixelHeight:\s*(\d+)/.exec(probe.stdout)?.[1]) || undefined
    const pixels = thumb ? await loadPixels($, thumb) : undefined
    hasAlpha = !!pixels && isTransparent(pixels)
    if (hasAlpha) {
      preview = `${dir}/${id}-p.bmp`
      await $.process.run(['/usr/bin/sips', '-Z', String(PREVIEW_PIXELS), '-s', 'format', 'bmp', png, '--out', preview])
      if (!(await $.fs.exists(preview))) preview = undefined
    }
  }

  const stat = await $.fs.stat(source).catch(() => undefined)
  const name = job.name ?? ('path' in job ? baseName(job.path) : `${job.from === 'you' ? 'pasted' : 'image'}-${id}.${extOf(source)}`)
  const attachment: Attachment = {
    key: job.key,
    from: job.from,
    via: job.via,
    name,
    openPath: source,
    png,
    thumb,
    preview,
    width,
    height,
    bytes: stat?.size,
    hasAlpha,
    batch: job.batch,
    qa: await qaOf($, source),
    at: await $.clock.now(),
  }
  const critiqued = /_critique\.png$/i.test(source) ? source.replace(/_critique\.png$/i, '') : undefined
  let at = 0
  await update($, items, list => {
    const next = list.some(one => one.key === job.key) ? list : [...list, attachment]
    at = next.length - 1

    return next
  })
  if (critiqued) {
    // a critique overlay arrived after its mockup: carry its QA onto the mockup
    const qa = attachment.qa ?? (await qaOf($, `${critiqued}.png`))
    if (qa) await update($, items, list => list.map(one => (one.openPath === `${critiqued}.png` ? { ...one, qa } : one)))
  }
  const list = await read($, items)
  const batchItems = job.batch ? list.filter(one => one.batch === job.batch) : []
  const isLastOfBatch = !job.batch || batchItems.length >= (job.batchSize ?? 1)
  if (job.batch && (job.batchSize ?? 1) > 1) {
    // show a batch from its first picture, as a grid
    await update($, selected, () => list.findIndex(one => one.batch === job.batch))
    await update($, view, () => 'grid')
  } else {
    await update($, selected, () => at)
  }
  if (!isLastOfBatch) return
  void $.ui.open({ id: PANE, title: TITLE })
  $.ui.toast(
    job.batch && batchItems.length > 1
      ? `🖼 ${batchItems.length} pictures from ${job.batch.replace(/ · \d+$/, '')} · /attachments`
      : `📎 ${job.from === 'you' ? 'Your' : "Claude's"} attachment ready · /attachments`,
  )
}

async function backfill($: EngineInterface) {
  const messages = await $.session.messages({ as: 'api' })
  if (!Array.isArray(messages)) return
  const toolOf = new Map<string, { tool: string; path?: string; command?: string }>()
  for (const message of messages) {
    if (message.role !== 'assistant' || !Array.isArray(message.content)) continue
    for (const block of message.content as Block[]) {
      if (block.type !== 'tool_use') continue
      const input = (block.input ?? {}) as { file_path?: string; files?: string[]; command?: string }
      toolOf.set(String(block.id), { tool: String(block.name), path: input.file_path, command: input.command })
      if (block.name === 'SendUserFile') {
        for (const file of input.files ?? []) {
          enqueue($, { key: `path:${file}`, from: 'claude', via: 'SendUserFile', path: file })
        }
      }
    }
  }
  for (const message of messages) {
    if (message.role !== 'user' || !Array.isArray(message.content)) continue
    const blocks = message.content as Block[]
    const isToolResult = blocks.some(block => block.type === 'tool_result')
    for (const block of blocks) {
      if (block.type === 'tool_result') {
        const call = toolOf.get(String(block.tool_use_id))
        if (call?.path) readPathOf.set(String(block.tool_use_id), call.path)
        if (call?.tool === 'Bash' && call.command && isGptkit(call.command)) {
          captureGptkit($, call.command, textOf(block.content), String(block.tool_use_id))
        }
        for (const job of jobsIn([block], 'claude', call?.tool)) enqueue($, job)
      } else if (!isToolResult) {
        for (const job of jobsIn([block], 'you')) enqueue($, job)
      }
    }
  }
}

export function isGptkit(command: string): boolean {
  return /gptkit(\.py)?\b/.test(command)
}

/** The picture paths a gptkit command printed, in order, each once, review sheets included. */
export function printedPictures(output: string): string[] {
  const found: string[] = []
  for (const match of output.matchAll(PRINTED_PATH)) {
    const path = match[1]
    if (path && !found.includes(path)) found.push(path)
  }

  return found
}

export function gptkitLabel(command: string): string {
  const sub = /gptkit(?:\.py)?["']?\s+([a-z-]+)/.exec(command)?.[1]

  return sub ? `gptkit ${sub}` : 'gptkit'
}

function captureGptkit($: EngineInterface, command: string, output: string, runId: string | number) {
  const paths = printedPictures(output)
  if (paths.length === 0) return
  const label = gptkitLabel(command)
  const batch = `${label} · ${String(runId).slice(-6)}`
  for (const path of paths) {
    enqueue($, { key: `path:${path}`, from: 'claude', via: label, path, batch, batchSize: paths.length })
  }
}

async function listPictures($: EngineInterface, path: string): Promise<string[] | undefined> {
  const full = path.startsWith('/') ? path : `${await $.session.cwd()}/${path}`
  const stat = await $.fs.stat(full).catch(() => undefined)
  if (stat?.kind !== 'dir') return undefined
  const entries = await $.fs.list(full)

  return entries
    .filter(entry => entry.kind === 'file' && PICTURE_EXT.test(entry.name))
    .map(entry => `${full.replace(/\/$/, '')}/${entry.name}`)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
}

async function qaOf($: EngineInterface, picture: string): Promise<QaSummary | undefined> {
  const json = picture.replace(/(_critique)?\.[a-z]+$/i, '_critique.json')
  if (!(await $.fs.exists(json))) return undefined
  try {
    const log = JSON.parse(await $.fs.read(json)) as { rounds?: { issues?: { severity?: string }[]; weight?: number }[] }
    const first = log.rounds?.[0]
    if (!first?.issues) return undefined

    return { issues: first.issues.length, weight: first.weight ?? 0 }
  } catch {
    return undefined
  }
}

export function jobsIn(content: Block[], from: AttachmentFrom, via?: string): Job[] {
  const jobs: Job[] = []
  for (const block of content) {
    const media = mediaOf(block)
    if (media) jobs.push({ key: `data:${fingerprint(media.base64)}`, from, via, ...media })
    if (block.type !== 'tool_result' || !Array.isArray(block.content)) continue
    const readPath = readPathOf.get(String(block.tool_use_id))
    for (const inner of block.content as Block[]) {
      const nested = mediaOf(inner)
      if (!nested) continue
      if (readPath && IMAGE_EXT.test(readPath)) {
        jobs.push({ key: `path:${readPath}`, from, via, path: readPath })
      } else {
        jobs.push({ key: `data:${fingerprint(nested.base64)}`, from, via, ...nested })
      }
    }
  }

  return jobs
}

function mediaOf(block: Block): { base64: string; mediaType: string } | undefined {
  if (block.type !== 'image' && block.type !== 'document') return undefined
  const source = block.source as { type?: string; data?: unknown; media_type?: unknown } | undefined
  if (source?.type !== 'base64' || typeof source.data !== 'string') return undefined

  return { base64: source.data, mediaType: String(source.media_type ?? 'application/octet-stream') }
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''

  return (content as Block[]).map(block => (block.type === 'text' ? String(block.text ?? '') : '')).join('\n')
}

function isPicture(one: Attachment | undefined): one is Attachment {
  return !!one?.png && !!one.width && !!one.height
}

function isIconSized(one: Attachment): boolean {
  return (!!one.width && !!one.height && Math.max(one.width, one.height) <= ICON_PIXELS) || /\.svg$/i.test(one.openPath)
}

/** The pictures shown with `item` in the grid: its batch, else up to 12 neighbours. */
export function groupOf(list: Attachment[], index: number): Attachment[] {
  const item = list[index]
  if (!item) return []
  const pictures = list.filter(isPicture)
  if (item.batch) {
    const batch = pictures.filter(one => one.batch === item.batch)
    if (batch.length > 0) return batch
  }
  const at = Math.max(0, pictures.indexOf(item))
  const first = Math.max(0, Math.min(at - 6, pictures.length - 12))

  return pictures.slice(first, first + 12)
}

export function togglePick(keys: string[], key: string): string[] {
  if (keys.includes(key)) return keys.filter(one => one !== key)

  return [...keys, key].slice(-MAX_PICKED)
}

export function nextBackdrop(was: Backdrop): Backdrop {
  return BACKDROPS[(BACKDROPS.indexOf(was) + 1) % BACKDROPS.length] ?? 'checker'
}

export function ratioLabel(width: number, height: number): string {
  const ratio = width / height
  const known = RATIOS.find(([, value]) => Math.abs(ratio / value - 1) < 0.015)

  return known ? known[0] : ratio.toFixed(3)
}

export function sizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function detailsOf(one: Attachment, isShort: boolean): string {
  const parts: string[] = []
  if (one.width && one.height) parts.push(`${one.width}×${one.height}`, ratioLabel(one.width, one.height))
  if (!isShort && one.bytes) parts.push(sizeLabel(one.bytes))
  if (one.hasAlpha) parts.push('alpha')
  if (one.qa) parts.push(one.qa.issues === 0 ? 'QA clean' : `QA ${one.qa.issues} · w${one.qa.weight}`)

  return parts.length > 0 ? ` · ${parts.join(' · ')}` : ''
}

async function hasPixelGraphics($: EngineInterface): Promise<boolean> {
  const program = (await $.env.get('TERM_PROGRAM'))?.toLowerCase() ?? ''
  const term = (await $.env.get('TERM'))?.toLowerCase() ?? ''
  const kitty = await $.env.get('KITTY_WINDOW_ID')
  const ghostty = await $.env.get('GHOSTTY_RESOURCES_DIR')

  return program.includes('ghostty') || program.includes('kitty') || term.includes('kitty') || !!kitty || !!ghostty
}

async function cacheDir($: EngineInterface): Promise<string> {
  const home = await $.env.get('HOME')
  const session = await $.session.id()

  return `${home}/.claude/attachment-preview/${session}`
}

async function openFile($: EngineInterface, path: string, reveal = false) {
  await $.process.run(reveal ? ['/usr/bin/open', '-R', path] : ['/usr/bin/open', path])
}

async function loadPixels($: EngineInterface, bmpPath: string): Promise<Pixels | undefined> {
  const cached = pixelCache.get(bmpPath)
  if (cached) return cached
  const { base64 } = await $.fs.read(bmpPath, { as: 'bytes' })
  const pixels = parseBmp(Uint8Array.fromBase64(base64))
  if (pixels) pixelCache.set(bmpPath, pixels)

  return pixels
}

export function parseBmp(bytes: Uint8Array): Pixels | undefined {
  if (bytes.length < 54 || bytes[0] !== 0x42 || bytes[1] !== 0x4d) return undefined
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const offset = view.getUint32(10, true)
  const width = view.getInt32(18, true)
  const rawHeight = view.getInt32(22, true)
  const bpp = view.getUint16(28, true)
  if (width <= 0 || rawHeight === 0 || (bpp !== 24 && bpp !== 32)) return undefined
  const height = Math.abs(rawHeight)
  const isTopDown = rawHeight < 0
  const stride = Math.ceil((width * bpp) / 32) * 4
  const step = bpp / 8
  const rgb = new Uint8Array(width * height * 3)
  // sips writes 32-bit BMPs only for pictures with an alpha channel
  const alpha = bpp === 32 ? new Uint8Array(width * height) : undefined
  for (let y = 0; y < height; y++) {
    const row = offset + (isTopDown ? y : height - 1 - y) * stride
    for (let x = 0; x < width; x++) {
      const from = row + x * step
      const to = (y * width + x) * 3
      rgb[to] = bytes[from + 2] ?? 0
      rgb[to + 1] = bytes[from + 1] ?? 0
      rgb[to + 2] = bytes[from] ?? 0
      if (alpha) alpha[y * width + x] = bytes[from + 3] ?? 255
    }
  }

  return { width, height, rgb, alpha }
}

/** True when some pixels are see-through; an all-zero alpha channel is a writer that left it empty, not transparency. */
export function isTransparent(pixels: Pixels): boolean {
  if (!pixels.alpha) return false
  let isSeeThrough = false
  let isOpaque = false
  for (const value of pixels.alpha) {
    if (value < 250) isSeeThrough = true
    if (value > 5) isOpaque = true
    if (isSeeThrough && isOpaque) return true
  }

  return false
}

/** The picture laid over a checkerboard, light or dark backdrop, as opaque pixels. */
export function composite(pixels: Pixels, chosen: Backdrop): Pixels {
  if (!pixels.alpha) return pixels
  const { width, height } = pixels
  const square = Math.max(2, Math.round(Math.max(width, height) / 24))
  const rgb = new Uint8Array(width * height * 3)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const at = y * width + x
      const isDark = ((Math.floor(x / square) + Math.floor(y / square)) & 1) === 1
      const ground = chosen === 'light' ? 245 : chosen === 'dark' ? 28 : isDark ? 204 : 255
      const a = (pixels.alpha[at] ?? 255) / 255
      for (let c = 0; c < 3; c++) {
        rgb[at * 3 + c] = Math.round((pixels.rgb[at * 3 + c] ?? 0) * a + ground * (1 - a))
      }
    }
  }

  return { width, height, rgb }
}

function toRgba(pixels: Pixels): Uint8Array {
  const rgba = new Uint8Array(pixels.width * pixels.height * 4)
  for (let at = 0; at < pixels.width * pixels.height; at++) {
    rgba[at * 4] = pixels.rgb[at * 3] ?? 0
    rgba[at * 4 + 1] = pixels.rgb[at * 3 + 1] ?? 0
    rgba[at * 4 + 2] = pixels.rgb[at * 3 + 2] ?? 0
    rgba[at * 4 + 3] = 255
  }

  return rgba
}

export function fitBox(width: number, height: number, maxColumns: number, maxRows: number) {
  const scale = Math.min(maxColumns / width, (maxRows * 2) / height)
  const columns = Math.max(1, Math.min(255, maxColumns, Math.round(width * scale)))
  const rows = Math.max(1, Math.min(255, maxRows, Math.round((height * scale) / 2)))

  return { columns, rows }
}

export function halfBlockCells(pixels: Pixels, columns: number, rows: number): string {
  const words = new Uint32Array(columns * rows * 3)
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const top = averageColor(pixels, column / columns, (row * 2) / (rows * 2), 1 / columns, 1 / (rows * 2))
      const bottom = averageColor(pixels, column / columns, (row * 2 + 1) / (rows * 2), 1 / columns, 1 / (rows * 2))
      const at = (row * columns + column) * 3
      words[at] = HALF_BLOCK
      words[at + 1] = top
      words[at + 2] = bottom
    }
  }

  return new Uint8Array(words.buffer).toBase64()
}

function averageColor(pixels: Pixels, fx: number, fy: number, fw: number, fh: number): number {
  const x0 = Math.floor(fx * pixels.width)
  const y0 = Math.floor(fy * pixels.height)
  const x1 = Math.max(x0 + 1, Math.min(pixels.width, Math.floor((fx + fw) * pixels.width)))
  const y1 = Math.max(y0 + 1, Math.min(pixels.height, Math.floor((fy + fh) * pixels.height)))
  let r = 0
  let g = 0
  let b = 0
  let count = 0
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const at = (y * pixels.width + x) * 3
      r += pixels.rgb[at] ?? 0
      g += pixels.rgb[at + 1] ?? 0
      b += pixels.rgb[at + 2] ?? 0
      count++
    }
  }
  if (count === 0) return 0

  return (Math.round(r / count) << 16) | (Math.round(g / count) << 8) | Math.round(b / count)
}

function fingerprint(data: string): string {
  let hash = 0x811c9dc5
  const step = Math.max(1, Math.floor(data.length / 8192))
  for (let i = 0; i < data.length; i += step) {
    hash ^= data.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }

  return (hash >>> 0).toString(16).padStart(8, '0') + data.length.toString(36)
}

function fileUrl(path: string): string {
  return `file://${path.split('/').map(encodeURIComponent).join('/')}`
}

function escapeLabel(name: string): string {
  return name.replace(/([\\[\]*_`])/g, '\\$1')
}

function baseName(path: string): string {
  return path.replace(/\/$/, '').slice(path.replace(/\/$/, '').lastIndexOf('/') + 1)
}

function extOf(path: string): string {
  const dot = path.lastIndexOf('.')

  return dot < 0 ? 'bin' : path.slice(dot + 1)
}
