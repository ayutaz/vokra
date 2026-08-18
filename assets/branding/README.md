# Vokra brand assets

The mark is a Roman inscriptional **V**, and it is not a letter drawn onto a
tile: it is the void left after cutting a V-section groove into a solid block,
which is how Roman inscriptions were actually made. Vokra takes its name from
*vox*, Latin for voice, so the letterform and the name come from the same
place.

The stroke contrast — heavy left downstroke, light right upstroke — is not
styling. A chisel widens the groove differently depending on the direction of
the cut, so the downstroke comes out heavier. The terminals are cut
horizontally and the strokes flare into them. Nothing in the outline was added
to look nice; every line follows from the tool.

That is also the argument for using it here:

| what the project claims | what the form does |
| --- | --- |
| depth over breadth, speech only | the form follows the tool and carries nothing that is not structural |
| zero dependencies | one block — no parts, no seams, nothing attached |
| never ingests a graph | the opposite of a node-and-edge mesh; a single unbreakable face |
| never falls back silently | every edge is a straight line: no gradients, no blur, nothing that fades out ambiguously |
| runs everywhere | one monochrome silhouette, identical at 24 px and 1280 px, with no simplified variant |

## Files

| file | use |
| --- | --- |
| `vokra-avatar.svg` / `-512.png` / `-1024.png` | **Default.** Org avatars (Hugging Face, GitHub), app icons |
| `vokra-avatar-mono.svg` / `-512.png` | Where the accent colour is unavailable or unwanted |
| `vokra-avatar-light.svg` / `-512.png` | Light-ground contexts, print |
| `vokra-mark.svg` | Mark alone, `currentColor`, no ground — inherits the surrounding text colour |
| `vokra-mark-minium.svg` | Mark alone, fixed accent |
| `vokra-mark-plain.svg` | Mark alone without terminal flare, for very small or low-fidelity rendering |
| `vokra-icon-32.png` / `-64.png` | Favicons |
| `vokra-social.png` | GitHub social preview (1280×640, GitHub's recommended size) |
| `vokra-lockup.png` | Horizontal mark + wordmark for README headers and slides |

Every vertex sits inside a circle of radius 200 in the 512 canvas, so the mark
survives a circular crop with its corners intact. The generator asserts this;
it is not eyeballed.

## Colour

| | hex | |
| --- | --- | --- |
| ink | `#15131A` | ground |
| paper | `#F6F4F0` | monochrome mark |
| minium | `#C8452B` | accent |

Minium (red lead) is the pigment Roman carvers packed into cut grooves so an
inscription could be read from a distance. The accent continues how the mark
is made rather than decorating it.

## Using it

Leave clear space of at least half the mark's height on every side. Do not
stretch, rotate, outline, add effects to, or re-colour the mark outside the
palette above. On a busy photograph, use the avatar (which carries its own
ground) rather than the bare mark.

## Regenerating

```sh
uv run --no-project --python 3.12 --with fonttools python \
  tools/branding/gen_brand.py
```

Every dimension is a constant at the top of that script, so weight,
proportion, and flare can all be re-tuned in one place. Developer-side only:
it needs `rsvg-convert` and `fonttools`, and nothing in the build, the runtime,
or CI depends on it (NFR-DS-02).

## Typeface

The wordmark is set in **Optima**. Hermann Zapf drew it in 1950 from Roman
gravestone lettering in the Basilica di Santa Croce, Florence — strokes that
widen towards their terminals without ending in a serif. That is the same
construction as the flare on this mark, so the pairing is a shared lineage
rather than a matter of taste.

**Optima appears in rasterised output only.** Committing its outlines as SVG
paths would redistribute the typeface, which its licence does not permit;
setting text into a PNG is ordinary use. Every SVG here is the mark alone and
contains no glyph data. A custom-drawn logotype, should one ever be wanted, is
separate work — until then, set the name in Optima or another humanist sans
and do not check font outlines into this repository.

## Licence

The source in this repository is Apache-2.0. That licence covers the code, not
the brand: Apache-2.0 §6 grants no trademark rights, and the Vokra name and
mark identify the project. Use them to refer to Vokra — in articles, ports,
integrations, and package listings — but not in a way that suggests a modified
build or a third-party product is the official one.
