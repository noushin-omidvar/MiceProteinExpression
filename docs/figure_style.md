# Visual identity

One style for every output: figures, paper, report, slides and notebooks. It is implemented
once, in `src/mice_protein/plotting.py`, and reused everywhere.

## Principles

* Minimal, editorial, scientific: thin axes, subtle horizontal gridlines, generous whitespace.
* No decoration: no gradients, shadows, 3-D, unnecessary borders or extra colours.
* Each panel title states its takeaway; axes state units.
* Schematics are labelled **SCHEMATIC**; everything else is data.
* Never fabricate or smooth data; uncertainty is always shown (95% intervals unless stated).

## Colour roles

| Role | Hex | Use |
|---|---|---|
| Ink | `#161B1F` | Text, axes, means and intervals |
| Cobalt | `#2457D6` | Primary information; control mice |
| Vermilion | `#E33D3F` | Meaningful highlight; Ts65Dn mice; the effect in focus |
| Muted blue | `#8EA8EC` | Secondary or background data (never text) |
| Rule gray | `#D9DDE0` | Gridlines, borders and separators only |
| Background | `#FFFFFF` | All surfaces |

Diverging maps run cobalt → light gray → vermilion. The palette passes colour-vision-deficiency
checks; muted blue has low contrast on white (2.3:1), so it is reserved for data marks and is
always paired with a legend.

Identity is never colour alone: **treatment** is marker fill (open = saline, filled = memantine)
and **conditioning** is marker shape (circle = CS, triangle = SC).

## Typography

| Use | Typeface |
|---|---|
| Labels, body text | IBM Plex Sans |
| Figure and section headings | Fraunces (SemiBold) |
| Statistics, sample sizes, code | IBM Plex Mono |

Fonts ship in `assets/fonts` under the SIL Open Font License. IBM Plex is IBM's unmodified WOFF
release, decompressed to an untracked local cache (`.cache/fonts`) at runtime. Fraunces static
weights were instantiated from the variable font (OFL; no reserved font name).

## Sizes and export

* Widths follow journal columns: 89 mm (single) and 183 mm (double); text is 7 pt at print size.
* Every figure is exported as SVG (editable text), PDF (embedded fonts) and 300-dpi PNG, with a
  margin so nothing is cropped.

## Using it

```python
from mice_protein import plotting as P
P.apply_style()                 # once per process
fig, ax = plt.subplots(figsize=(P.SINGLE, 60 * P.MM))
P.title(ax, "Takeaway as a statement")
P.annotate_stat(ax, x, y, "p = 0.03")
P.save(fig, out_dir, "name")    # writes .svg, .pdf, .png
```
