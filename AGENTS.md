# AGENTS.md

Rules for agents working in this repository. Read this file before editing anything.

## Environment

`devenv.nix` is the only provisioning path. Do not add a flake, a `shell.nix`, or a second environment.

- Run every command under the devenv shell: `devenv shell -- <command>`.
- Do not install tools imperatively. There is no `pip install`, `npm -g`, `cargo install`, or `apt`.
- Do not commit unless the task asks for it. Keep each change inside the task's scope.

If devenv cannot create its runtime directory, prefix the command with `XDG_RUNTIME_DIR=/tmp/xdg-devenv XDG_CACHE_HOME=/tmp/cache-devenv`.

## Build and test

```sh
devenv shell -- bash scripts/build.sh          # configure and build into build/
devenv shell -- bash scripts/test.sh           # build, lit, ctest, pytest, prose, spec, emitted-project checks
```

`scripts/test.sh` runs every check that works without a card. It skips the emitted-project check with a message when `.deps` is absent, and never reports that skip as a pass.

Individual targets, after `scripts/build.sh`:

```sh
cmake --build build --target check-tensixflow    # lit/FileCheck IR suite
ctest --test-dir build --output-on-failure       # GoogleTest suites
python3 -m pytest python -q                      # golden, rounding, harness tests
```

## Emitted project

The emitted-project path builds against the pinned TT-Metalium release. Fetch once (see the next section), then build and run:

```sh
devenv shell -- bash scripts/build_generated.sh examples/matmul.mlir   # emit from an example, configure, compile, link
devenv shell -- bash scripts/run_generated.sh build-generated/matmul   # run under ttsim, no card needed
devenv shell -- bash scripts/run_generated.sh --device                 # run on a card
```

`scripts/test.sh` runs every emittable example after building. The examples are `examples/matmul.mlir`, `examples/matmul_small.mlir`, and `examples/rmsnorm_matmul_small.mlir`, the emittable fused example.

The last command needs a card. On a card host, probe first:

```sh
devenv shell -- python3 scripts/device_probe.py  # exit 2 with no card, 1 on mismatch
```

## Pinned upstreams

```sh
devenv shell -- bash scripts/fetch_tt_metal.sh       # tt-metal v0.82.0-rc2 artifacts + ttsim v1.15.1 into .deps/
devenv shell -- bash scripts/fetch_ttmlir_fixtures.sh # tt-mlir 0.9.0.dev20260221 wheel and fixtures into .deps/
```

Both caches live in `.deps/` and are gitignored. The lock files in `deps/` record the tags and sha256 digests. Change a pin in the lock file, then re-run the fetch script.

Generated device data has one source. Edit `spec/devices.json`, then regenerate:

```sh
devenv shell -- python3 tools/gen_device_spec.py
devenv shell -- python3 tools/gen_device_spec.py --check
```

Do not hand-edit `include/tensixflow/Generated/DeviceSpecs.h` or `python/tensixflow/devices.py`; the generator owns them.

## Prose

`tools/check-prose.py` is the mechanical gate over Markdown, Python, shell, Nix, YAML, C++, headers, TableGen, and MLIR files.

```sh
devenv shell -- python3 tools/check-prose.py
```

It rejects the em dash, a banned word list, and a banned phrase list. The short version: short declarative sentences, actor in the subject and action in the verb, active voice, concrete numbers, sentence-case headings, straight quotes, no emoji, and no closing summary paragraph. No bold scattered through a paragraph.

## Code style

- Braces after every `if`, even a single-line body.
- No magic numbers or strings. Name the constant, or use an enum.
- Prefer an enum over a boolean parameter, so a call site reads as what it means.
- Use early returns and `continue` to keep indentation shallow.
- Put a short comment above each logical block. Explain why the block is shaped that way, not what the line does.
- Avoid bare pronouns in comments. Name the thing.
- Write the test first. Watch it fail, then write the code, then watch it pass.

## Commit messages

1. Separate the subject from the body with one blank line.
2. Keep the subject near 50 characters, 72 at most.
3. Capitalize the subject and leave off the final period.
4. Use the imperative mood: "Add the plan device block", not "Added".
5. Wrap the body at 72 characters by hand.
6. Explain why in the body. The diff shows how.
7. Keep one logical change per commit.

## Claims

This machine has no Tenstorrent card. Anything that needs one is untested and must be labelled that way.

- Hardware values come from `spec/devices.json`, which cites a source URL and a retrieval date for each figure. Do not invent a number.
- The cost model reports projected cycles and projected traffic. Do not report them as measured performance.
- The emitted project compiles and links against the pinned release, and its matmul and fused programs run under ttsim. A simulator result is not a device result.
- Do not add signature-only tt-metal stub headers or call an API the pinned release does not export. The pin in `deps/` is the only surface.
- Documents under `docs/` state facts about the code as it is. Leave planning language, todo numbers, and decision codes out of them.
