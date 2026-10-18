export type AttachmentFrom = 'you' | 'claude'

export type QaSummary = {
  issues: number
  weight: number
}

export type Attachment = {
  key: string
  from: AttachmentFrom
  via?: string
  name: string
  openPath: string
  png?: string
  thumb?: string
  /** A larger BMP (alpha kept) for compositing a transparent asset onto a backdrop. */
  preview?: string
  width?: number
  height?: number
  bytes?: number
  hasAlpha?: boolean
  /** Items captured together (one gptkit command, one folder) share a batch and show as a grid. */
  batch?: string
  /** Design QA from gptkit's `<stem>_critique.json` next to the picture. */
  qa?: QaSummary
  at: number
}

export type PreviewMode = 'auto' | 'pixels' | 'blocks'

export type ViewMode = 'single' | 'compare' | 'grid'

export type Backdrop = 'checker' | 'light' | 'dark'

declare module 'claude-code' {
  interface PluginState {
    'attachment-preview': {
      items: Attachment[]
      selected: number
      mode: PreviewMode
      view: ViewMode
      picked: string[]
      backdrop: Backdrop
    }
  }
}
