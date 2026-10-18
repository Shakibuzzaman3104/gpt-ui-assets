# Test fonts

Small Basic-Latin subsets used only by the offline tests (`tests/test_design_to_code.py`). They are not shipped to
users of the kit: `gptkit fonts` downloads full fonts from Google Fonts at run time.

| File | Font | Licence |
|---|---|---|
| `Inter-300.ttf`, `Inter-400.ttf`, `Inter-500.ttf`, `Inter-700.ttf` | Inter, subset | SIL OFL 1.1, [OFL-Inter.txt](OFL-Inter.txt) |
| `Roboto-400.ttf` | Roboto, subset | SIL OFL 1.1, [OFL-Roboto.txt](OFL-Roboto.txt) |
| `SerifTest-700.ttf` | A subset of Lora Bold, renamed because Lora's licence reserves the name "Lora" for unmodified versions | SIL OFL 1.1, [OFL-Lora.txt](OFL-Lora.txt) |

Each subset is a modified version of the original font and is distributed under the same licence.
