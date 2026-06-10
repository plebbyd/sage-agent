# PTZ Camera App — Agentic Rules

## Identity
You are operating a simulated PTZ (Pan-Tilt-Zoom) camera overlooking a
360-degree panoramic scene. Your camera is backed by `stitched.png` — a high-
resolution panorama. You control what the camera sees by adjusting pan, tilt,
and field of view.

## Coordinate System
- **Pan**: 0–360 degrees. 0° is the left edge of the panorama. Wraps around
  (359° + 2° = 1°). Think of it as a compass heading.
- **Tilt**: 0° is the bottom of the scene, maximum (~153°) is the top. Clamped
  at edges — you cannot look below the ground or above the sky.
- **FOV**: Horizontal field of view in degrees. Smaller FOV = more zoom.
  Vertical FOV is derived automatically for a 16:9 aspect ratio.

## Available PTZ Tools

| Tool | What it does |
|------|-------------|
| `sim_ptz_move` | Go to an absolute pan/tilt position |
| `sim_ptz_pan` | Pan left (negative) or right (positive) by N degrees |
| `sim_ptz_tilt` | Tilt up (positive) or down (negative) by N degrees |
| `sim_ptz_position` | Report current pan, tilt, FOV |
| `sim_ptz_snapshot` | Save current viewport as JPEG |
| `sim_ptz_overview` | Save full panorama with viewport rectangle highlighted |
| `sim_ptz_composite` | Save panorama + viewport side by side |
| `sim_ptz_detect` | Run object detection on current viewport |
| `sim_ptz_caption` | Generate a text description of current viewport |
| `sim_ptz_mission` | **Agentic scan** — natural-language mission, panorama sweep or random views, detection, dedupe, counts |

## Agentic missions (`sim_ptz_mission`)

Use this when the task is to **search the whole scene** (or sample it), not just the
current viewport. Pass a free-text `mission` string; the backend picks YOLO classes
or BioCLIP with an optional taxon substring filter.

Examples:
- **Animals / wildlife** — `"scan for all animals"`, `"count the animals"` → COCO
  animal classes (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe).
- **Specific class** — `"find cows"`, `"cows"` → YOLO `cow`.
- **Random/explore** — `"random things"`, `"explore"` → random viewpoints, all classes.
- **Taxon / species hints** — `'Aves'` in quotes, or `"find deer"` (non-COCO) → BioCLIP
  with `target_taxon` as a substring filter on the predicted label (if BioCLIP is installed);
  otherwise missions fall back to YOLO when a single COCO class matches.

The JSON response includes `summary.counts_by_label`, `unique_instances_estimated`
(spatial dedupe), and `unique_detections` with approximate pan/tilt per hit.

CLI (same engine): `python3 -m tools.ptz_mission --mission "find cows"`

Prefer **`sim_ptz_mission`** over manual `sim_ptz_pan` loops when the user asks to
scan, count, or find objects **across the panorama**.

## Movement Strategy
- **Know where you are first.** Before moving, call `sim_ptz_position` if you
  don't already know the current heading. The position is also available in the
  PTZ state file (`scratchpads/sim_ptz_state.json`).
- **Move in purposeful increments.** Don't randomly jump around. If searching
  for something, use a systematic sweep pattern (see below).
- **Use absolute positions for known landmarks.** If you've previously noted
  that an object of interest is at pan=120°, go directly there.
- **Use relative moves for scanning.** Pan by FOV-width increments to ensure
  full coverage without overlap gaps.

## Scanning Patterns

### Full Horizon Sweep
To survey the entire scene at current tilt:
1. Start at pan=0°
2. Pan right by `fov_h` degrees each step (default 60°)
3. At each stop, run detection or captioning
4. Continue until you've covered 360°
5. Log findings in notes

### Grid Scan (Thorough)
For complete scene coverage:
1. Start at pan=0°, tilt=bottom (half of fov_v)
2. Sweep a full horizon row (as above)
3. Tilt up by `fov_v` degrees
4. Sweep the next row
5. Repeat until you reach the top
6. This produces a complete tiled map of the scene

### Targeted Investigation
When you've spotted something interesting:
1. Note the current position and detection
2. Zoom in: reduce FOV for a closer look (not yet implemented — note for future)
3. Caption the viewport for a detailed description
4. Take a snapshot for the record
5. Return to your previous position or continue scanning

## Detection & Captioning

### When to Use Each Model
- **YOLO** (`sim_ptz_detect model=yolo`): Fast, general-purpose object detection.
  Best for: people, vehicles, animals, common objects. Use `targets` arg to
  filter (e.g., `targets="person,car"`). Use `targets="*"` for everything.
- **BioCLIP** (`sim_ptz_detect model=bioclip`): Biological / ecological
  classification. Best for: identifying species of birds, plants, insects.
  Use `target_taxon` to specify a taxonomic substring filter on the predicted
  label. Returns classification confidence plus Grad-CAM localization.
  Use `sim_ptz_caption` for a top-k species text summary (same backend).

### Detection Workflow
1. **Survey first, detect second.** Start with a horizon sweep using YOLO to
   get a broad inventory of what's visible.
2. **Log all detections.** Record each detection's label, confidence, and the
   pan/tilt position where it was seen. Store in notes as structured data.
3. **Follow up on interesting detections.** If YOLO finds "bird", re-examine
   with BioCLIP for species identification or caption.
4. **Caption sparingly.** Captioning is slower than detection. Use it for
   notable viewpoints, not every position in a sweep.

### Reporting Detections
When logging detections in notes, use this format:
```
detection @ pan=120.0 tilt=45.0: 3x person (0.92), 1x car (0.87)
```
This makes detections searchable across cycles.

## State Management
- PTZ position persists in `scratchpads/sim_ptz_state.json` across cycles.
- After completing a scan or investigation, update notes with a summary of
  what was found and the last known camera position.
- If a task says "monitor area X", store the target pan/tilt in notes so you
  can return to it in future cycles.

## Snapshot Hygiene
- Snapshots are saved to the project root. They accumulate.
- Only take snapshots when there's a reason: a notable detection, a task
  requires visual evidence, or you're documenting an anomaly.
- Include the position in the filename or notes so snapshots can be correlated
  with detections.

## Multi-Cycle Awareness
The PTZ camera persists across agent cycles. Use this to your advantage:
- **Patrol routes**: define a sequence of positions in pending_actions and
  visit one per cycle.
- **Change detection**: compare current detections with notes from previous
  cycles to identify new or missing objects.
- **Scheduled sweeps**: use the cron system (`config/tasks.yaml`) to trigger
  periodic full-scene surveys.
