# attachment-preview

A Claude Code mod that keeps every image of the session in one terminal pane, built to review what
[gpt-ui-assets](../../README.md) generates without leaving the terminal.

## What it shows

- **Everything, automatically:** images you paste, images Claude reads or sends, and every picture a `gptkit`
  command prints (mockup variants, critique overlays, forge and SVG review sheets, icons, app-icon sheets).
- **Grid (`g`):** a gptkit run or a folder opens as one grid, its variants side by side. A batch of small pictures
  or SVGs becomes an icon contact sheet. Large batches page.
- **Compare (`c`):** pick 2 to 4 pictures with `x` (marked ★) and compare them side by side as A–D. With nothing
  picked, compare shows the picture before the current one next to it.
- **Transparent assets (`b`):** pictures with an alpha channel are drawn over a checkerboard, a light or a dark
  backdrop, so you can see their edges and how they sit on both themes.
- **At a glance:** size in pixels, aspect ratio (9:19.5, 2:3, 1:1 …), file size, `alpha`, and the design QA from
  gptkit's `<name>_critique.json` (`QA 2 · w4`, `QA clean`).
- **Pixels or blocks (`m`):** real pixels in Ghostty, Kitty and other terminals with the graphics protocol;
  colour half-blocks everywhere else.

Keys while the pane is focused: `p`/`n` previous/next, `o` open, `r` reveal in Finder, `x` pick, `s` single,
`c` compare, `g` grid, `b` backdrop, `m` pixels/blocks.

## Commands

| Command | Does |
|---|---|
| `/attachments` | Open the pane |
| `/attachments add <file>` | Add one file (other mods can call this too) |
| `/attachments add <folder>` | Add every picture in a folder as one batch, e.g. `build/home/svg` |
| `/attachments grid` · `compare` · `single` | Open in that view |
| `/attachments clear` | Empty the list |

## Install

```
/plugin marketplace add Shakibuzzaman3104/gpt-ui-assets
/plugin install attachment-preview@gpt-ui-assets
```

macOS only for now: it converts with `sips` and renders SVGs and documents with Quick Look (`qlmanage`).

## Develop

```
claude plugin validate mods/attachment-preview
claude plugin test mods/attachment-preview
```
