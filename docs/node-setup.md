# Sage Edge Agent: Node Setup and Usage Guide

This is a start-to-finish guide for getting the agent running on an edge node, whether
that's an NVIDIA Jetson AGX Orin, a DGX Spark, or just your laptop for simulation.

If you're new, read the TL;DR first and come back for the details. If something breaks,
skip to the Troubleshooting section near the bottom.

## What this actually is

This repo is the runtime that runs on the node itself. It's an autonomous AI agent for
field science work like wildfire watch, animal and biodiversity tracking with PTZ
cameras, sky and weather observation, and agriculture. You give it a task in plain
English, and it drives the cameras and sensors, runs vision models, reasons about what
it sees, and reports back.

The agent lives in `ptz_node/`, and you talk to it with one command:

```bash
python -m ptz_node run "your task here"
```

Three pieces plug together to make it work:

- A reasoning LLM, which is the brain. This can be local (Ollama running on the node) or
  in the cloud (OpenRouter, Anthropic, or ANL's argo-proxy).
- Vision backends that look at camera frames: YOLO for objects, BioCLIP 2 for
  species and taxa, and Gemma for scene descriptions.
- A sensor gateway, which is the only thing that touches hardware. It talks to a real
  PTZ camera, or to a simulated one (a large stitched panorama image) so you can develop
  with no hardware at all.

One thing worth saying up front: you do not run Cursor or Claude Code on the node. You
run this program.

## TL;DR (the five-minute version)

Run these on the node, from the repo root, in order:

```bash
cd ~/sage-agent

# 1. Make the Python environment (creates .venv with Python 3.10, 3.11, or 3.12)
bash scripts/bootstrap_python311.sh
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-vision.txt        # YOLO/BioCLIP, optional but recommended

# 3. Pick a config and give it an LLM key (this example uses a cloud LLM via OpenRouter)
export PTZ_GRAPH_CONFIG=$PWD/config/dgx_spark.yaml
export OPENROUTER_API_KEY=sk-or-...            # your key from openrouter.ai/keys
export MSA_PTZ_BACKEND=sim                     # simulated camera, no hardware needed

# 4. Check that everything is wired up (these need no LLM)
python -m ptz_node doctor
python -m ptz_node gateway-smoke

# 5. Run the agent
python -m ptz_node run "List the devices, check detector status, and summarize node readiness."
```

If `doctor` comes back clean and step 5 prints a summary, you're good. The rest of this
guide explains what each piece does and how to change the LLM or the camera.

## Prerequisites

- A node you can SSH into (Jetson AGX Orin, DGX Spark, and so on), or your own Mac or
  Linux laptop for simulation.
- Python 3.10, 3.11, or 3.12. The bootstrap script finds a system Python in that range,
  and if the node only has something older it installs Python 3.11 with micromamba for
  you. Don't count on 3.13 or newer working; the code targets 3.10 and up.
- An LLM to reason with. Pick one of these:
  - Cloud: an OpenRouter API key (the easiest path), an Anthropic key, or argo-proxy if
    you're on the ANL network.
  - Local: Ollama running on the node with a model pulled. No API key needed, but it
    needs enough memory, which is covered in the caveats below.
- Optionally, a PTZ camera on the network. You don't need one. The `sim` backend uses a
  built-in panorama image.

## Step 1: Get the code onto the node

If you're developing on your laptop and pushing to a node, which is the common workflow,
use the sync script from your laptop. It copies the repo over SSH and skips junk like
`.venv`, `.local`, and caches by honoring `.gitignore`:

```bash
# from your laptop, in the repo root:
bash scripts/sync_to_spark.sh <user>@<node-host>:/home/<user>/sage-agent
# preview without copying:
DRY_RUN=1 bash scripts/sync_to_spark.sh <user>@<node-host>:/home/<user>/sage-agent
```

Or you can clone it directly on the node:

```bash
git clone https://github.com/<owner>/sage-agent.git ~/sage-agent
cd ~/sage-agent
```

One thing to know: the virtual environment does not travel with the code. `.venv/` is
deliberately never synced or committed, because it's specific to each machine. You always
create it fresh on each machine in Step 2. If you ever see `ModuleNotFoundError: No
module named 'langchain_core'`, it almost always means the venv is missing or just not
activated.

## Step 2: Create the Python environment

From the repo root on the node:

```bash
bash scripts/bootstrap_python311.sh
source .venv/bin/activate          # or: source scripts/activate_venv.sh
```

You should now see `(.venv)` at the start of your shell prompt.

Keep in mind that you have to re-activate the venv in every new SSH session, since it's
per-session. Each time you log in, run `cd ~/sage-agent && source .venv/bin/activate`. If
your prompt doesn't show `(.venv)`, you're on system Python and commands will fail.

## Step 3: Install dependencies

```bash
pip install -r requirements.txt            # core: agent loop, sensor gateway, sim PTZ
pip install -r requirements-vision.txt     # YOLO and BioCLIP 2 vision detectors
```

Here's what each file gives you:

| File | Needed for | Skip it if |
|------|-----------|------------|
| `requirements.txt` | The agent itself, sim PTZ, sensors, all LLM clients | Never skip this one |
| `requirements-vision.txt` | `ptz_detect` with YOLO and BioCLIP | You only want scene captions via Gemma/Ollama |
| `requirements-argo.txt` | The ANL `argo-proxy` LLM gateway | You use OpenRouter, Anthropic, or Ollama |

A couple of things to expect with vision. YOLO and BioCLIP need a GPU build of PyTorch
for aarch64 and CUDA on Jetson. If that isn't installed, those detectors simply show up
as unavailable, the agent still runs, and Gemma scene captions through Ollama still work.
Also, the first BioCLIP call downloads about 4.4 GB of model weights to
`~/.cache/huggingface`. That happens once and is cached afterward, so be patient and let
it finish.

## Step 4: Choose your LLM and configure it

The agent reads its settings from a YAML config, and you pick which one with the
`PTZ_GRAPH_CONFIG` environment variable. Secrets like API keys always come from
environment variables, never from the YAML files.

Here are the ready-made configs in `config/`:

| Config | Reasoning LLM | Camera | Use when |
|--------|--------------|--------|----------|
| `config/dgx_spark.yaml` | OpenRouter (cloud) | sim | A good default anywhere; just needs an OpenRouter key |
| `config/local.yaml` | argo-proxy (ANL) | sim | On the ANL network, or a laptop with an SSH tunnel |
| `config/default.yaml` | Ollama (local) | sim | Fully offline, or a local model on the node |

### Option A: Cloud LLM via OpenRouter (easiest)

```bash
export PTZ_GRAPH_CONFIG=$PWD/config/dgx_spark.yaml
export OPENROUTER_API_KEY=sk-or-...        # from https://openrouter.ai/keys
export MSA_PTZ_BACKEND=sim
```

If you want a different model, don't edit the tracked YAML on the node, because a re-sync
would overwrite it. Instead, drop a local override file at
`config/argo_proxy.local.yaml`. It's gitignored and gets merged last, on top of whatever
config you loaded:

```yaml
# config/argo_proxy.local.yaml, overrides anything above it
model:
  provider: openrouter
  model: openai/gpt-4o          # any tool-calling slug from openrouter.ai/models
```

### Option B: Local LLM via Ollama (no API key, runs on the node)

```bash
# install once: curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                  # leave running
ollama pull gemma4:31b          # or qwen2.5:7b; the model must support tool calling
export PTZ_GRAPH_CONFIG=$PWD/config/default.yaml
export MSA_PTZ_BACKEND=sim
```

### Option C: argo-proxy (ANL network)

```bash
bash scripts/setup_argo_proxy.sh -u YOUR_ANL_USER -m gpt-4o --jump node-V010
export PTZ_GRAPH_CONFIG=$PWD/config/local.yaml
python -m ptz_node argo test
```

One practical note on environment variables: the `export` lines disappear when you close
the shell. To keep them around, add them to `~/.bashrc` (or `~/.zshrc`) and then run
`source ~/.bashrc`. Just be careful, because if you put `PATH` changes in `~/.bashrc`
that knock out the venv, you'll need to re-activate it afterward. And keep secrets out of
any file that gets committed or synced.

## Step 5: Preflight checks (do this before your first run)

These need no LLM and confirm the node is healthy:

```bash
python -m ptz_node doctor          # environment, model, and camera checks
python -m ptz_node devices         # list every device the gateway sees
python -m ptz_node gateway-smoke   # exercise the camera and sensors end-to-end
```

`doctor` is the one to lean on. It tells you exactly what's missing and how to fix it.
Two things it commonly flags:

- If `ollama_model_pulled` fails, you chose a local model but haven't pulled it yet. Run
  the `ollama pull <model>` command it suggests.
- If you see vision or `gemma4` warnings, that's fine. Vision is optional and
  `gateway-smoke` still works. Only deal with it if you actually need those detectors.

Once `doctor` is clean and `gateway-smoke` returns `ok` for position, snapshot, and
sensors, you're ready to go.

## Step 6: Run the agent

```bash
python -m ptz_node run "Take a snapshot, run a tiled YOLO detection, and tell me what you see."
```

Here's what to expect while it runs:

- Live progress prints to your screen as it works. You'll see lines like
  `[  3s] → ptz_detect {"model":"yolo","tile":true}` followed by `[ 41s] ✓ ptz_detect`.
  So if it looks stuck, check whether a step is just slow, like a model loading or a
  cloud call, because the timestamps keep moving. Add `--quiet` to hide these lines.
- The final answer prints at the end, followed by a `# trace:` path.
- A full step-by-step trace gets saved to `.local/runs/<id>/summary.txt`. Read it if a
  run does something surprising.

A couple of flags worth knowing:

```bash
python -m ptz_node run "<task>" --limit 200    # allow more tool/LLM cycles for big jobs
python -m ptz_node run "<task>" --quiet         # suppress the live progress lines
```

Be aware that huge, open-ended prompts can run for a long time. If you ask it to scan the
entire panorama with every model on every subsection, the LLM ends up doing dozens of
cycles. For big sweeps like that, use a demo instead (the next section). A demo is a
scripted, bounded version that won't wander off or hit the cycle limit.

## Running canned workflows (demos)

Demos are pre-scripted Sage workflows that run as a single reliable call. They're great
for live demos, for smoke-testing the vision stack, or for "just do the whole thing"
requests. They print progress and save a JSON report to `.local/demos/`.

```bash
python -m ptz_node skill run demo --args '{"action":"list"}'                 # see them all
python -m ptz_node skill run demo --args '{"name":"edge_gateway_preflight"}' # no LLM, no vision
python -m ptz_node skill run demo --args '{"name":"panorama_scan"}'          # full sweep, every backend
python -m ptz_node skill run demo --args '{"name":"wildfire_smoke_patrol"}'
```

The available demos are `edge_gateway_preflight`, `ptz_multimodel_scientific_survey`,
`wildfire_smoke_patrol`, `aves_biodiversity_scan`, `land_cover_agriculture_scene`, and
`panorama_scan`. The agent can also call these itself. For example, if you ask it to scan
the whole panorama, it routes to `panorama_scan` automatically.

`panorama_scan` is bounded by a time budget and skips any backend that isn't installed,
so it can't hang. You can also limit which vision models it uses:

```bash
python -m ptz_node skill run demo --args '{"name":"panorama_scan","backends":["yolo","bioclip"]}'
```

## Connecting a real PTZ camera

If you have a camera on the ethernet network, the agent can find and configure it for
you. The `sensor_discovery` skill does this in three steps, and it never hangs because it
has retry caps and a time budget:

```bash
python -m ptz_node skill run sensor_discovery --args '{"action":"scan"}'
python -m ptz_node skill run sensor_discovery --args '{"action":"identify","ip":"192.168.1.108"}'
python -m ptz_node skill run sensor_discovery --args '{"action":"configure","ip":"192.168.1.108"}'
```

Or let the agent drive it for you:

```bash
python -m ptz_node run "Find the PTZ camera on the network and tell me how to set it up."
```

It hands back the exact environment variables to set. For a Reolink camera, that looks
like this:

```bash
export MSA_PTZ_BACKEND=reolink
export REOLINK_IP=192.168.1.108
export REOLINK_USER=admin
export REOLINK_PASSWORD=...        # the skill asks you for this; it never guesses
python -m ptz_node devices         # ptz_primary should now show backend=reolink
```

The camera password is yours to provide. The skill will never invent or store
credentials. It tells you which secrets it needs, like `REOLINK_PASSWORD`, and you set
them as environment variables. Never commit a camera password to a file.

## Running the test suite

The repo ships with Sage science test cases. They're the same scenarios as the demos, but
judged by the LLM, and they're a good way to check that the whole stack works:

```bash
python -m ptz_node test --list                              # list cases
python -m ptz_node test --id edge_gateway_preflight         # one case, no vision needed
python -m ptz_node test --all                               # everything
```

## Where things live

| Path | What it is |
|------|-----------|
| `ptz_node/` | The agent engine. Don't edit it unless you know why |
| `config/*.yaml` | Configs you select with `PTZ_GRAPH_CONFIG` |
| `config/argo_proxy.local.yaml` | Your personal overrides (gitignored) |
| `.local/runs/<id>/summary.txt` | Per-run trace. Read this after a weird run |
| `.local/demos/` | JSON reports from demo runs |
| `.local/debug/doctor.json` | The last `doctor` report |
| `.venv/` | Your Python environment (per-machine, not synced) |
| `.cursor/rules/` | Guidance for AI assistants editing this repo, not needed to run it |

Everything the agent writes at runtime goes under `.local/`, which is git-ignored, so
it's safe to delete if you want a clean slate.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `ModuleNotFoundError: No module named 'langchain_core'` | Venv is missing or not active. Run `bash scripts/bootstrap_python311.sh && source .venv/bin/activate`, then re-install requirements. |
| `command not found: argo-proxy` | Not installed, or the venv isn't active. Run `source .venv/bin/activate && pip install -r requirements-argo.txt`. |
| Prompt has no `(.venv)` prefix | You're on system Python. Run `source .venv/bin/activate`. |
| `doctor` says `ollama_model_pulled` failed | Run `ollama pull <model>`. The exact command is in the doctor output. |
| `ollama serve` says "address already in use" | Ollama is already running, which is fine. Do nothing. |
| A run looks like it's hanging | It's probably just slow, from a model download or load, or a big prompt. Watch the `[ Ns ]` progress timestamps, and in a second SSH session run `watch -n2 nvidia-smi` to confirm GPU activity. For big sweeps, use a demo instead. |
| The first vision call takes forever | BioCLIP is downloading about 4.4 GB to `~/.cache/huggingface`. It's a one-time thing, so let it finish. |
| `git push` returns 403 "Permission denied to <someone>" | Your machine cached the wrong GitHub account. On macOS, clear it with `printf 'protocol=https\nhost=github.com\n\n' \| git credential-osxkeychain erase`, then push again as the right user. |
| Vision detectors show as unavailable | GPU PyTorch isn't installed (`requirements-vision.txt`). This is optional, and Gemma captions via Ollama still work. |

When in doubt, run `python -m ptz_node doctor` and read `.local/runs/<id>/summary.txt`.

## Updating the code on a node

From your laptop, after making changes:

```bash
bash scripts/sync_to_spark.sh <user>@<node-host>:/home/<user>/sage-agent
```

Then on the node, but only if dependencies changed:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The sync skips `.venv/` and `.local/`, so your environment and run history on the node
stay intact. You don't need to re-bootstrap unless you deleted the venv.

## Quick command reference

```bash
python -m ptz_node doctor                 # health check, run this first
python -m ptz_node devices                # list cameras and sensors
python -m ptz_node gateway-smoke          # test the hardware path, no LLM
python -m ptz_node read sensor:system_stats   # one sensor reading
python -m ptz_node run "<task>"           # full agent loop
python -m ptz_node skill run demo --args '{"action":"list"}'   # canned workflows
python -m ptz_node skill run sensor_discovery --args '{"action":"scan"}'  # find cameras
python -m ptz_node test --all             # Sage test cases
```

If a node does something genuinely weird, grab the `.local/runs/<id>/summary.txt` file
and share it. That trace is the fastest way for someone to help you.
