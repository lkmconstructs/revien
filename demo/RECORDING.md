# Recording the Demo

Record an interactive terminal session of the Revien demo and convert it to a GIF for inclusion in documentation.

## Prerequisites

Install the recording and conversion tools:

```bash
pip install asciinema agg
pip install -e .
```

- **asciinema**: Records terminal sessions as `.cast` files
- **agg**: Converts `.cast` files to animated GIFs
- **revien**: Install the package in editable mode from the project root

## Record

Start recording with asciinema:

```bash
asciinema rec demo/revien_demo.cast --cols 100 --rows 30
```

Then run the demo:

```bash
python demo/openai_migration_demo.py
```

When the script finishes, exit the recording (Ctrl+D or type `exit`).

This creates `demo/revien_demo.cast`.

## Convert to GIF

Convert the cast file to an animated GIF:

```bash
agg demo/revien_demo.cast demo/revien_demo.gif --cols 100 --rows 30 --speed 1.5
```

Parameters:
- `--cols 100 --rows 30`: Match the recording dimensions
- `--speed 1.5`: Speed up playback by 1.5x for faster viewing

This creates `demo/revien_demo.gif` (~500KB–2MB depending on duration).

## Embed in README

Add the GIF to your project's README.md:

```markdown
![Revien Demo — Migrate from ChatGPT](./demo/revien_demo.gif)
```

The GIF will play automatically when the README is viewed on GitHub or other platforms that support inline animated images.

## Troubleshooting

- **Recording is too slow**: Increase `--speed` in the agg command (e.g., `--speed 2.0`).
- **GIF is too large**: Use higher `--speed` or reduce `--cols`/`--rows`.
- **agg not found**: Ensure it's installed: `pip install agg`.
- **asciinema not found**: Ensure it's installed: `pip install asciinema`.

## Tips

- Run the script once first to ensure it works
- Keep terminal width ~100 columns for readability
- The demo should complete in ~30–60 seconds of wall time
