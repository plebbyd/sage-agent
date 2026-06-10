"""
tools/ptz_viewer.py — Interactive browser-based PTZ camera viewer with detection.

Starts a lightweight local web server that serves the panorama image
and provides an interactive viewport with pan/tilt/zoom controls,
keyboard shortcuts, and on-demand object detection / captioning.

Usage:
    python tools/ptz_viewer.py                    # default port 8088
    python tools/ptz_viewer.py --port 9000
    python tools/ptz_viewer.py --image other.png  # custom panorama

Opens automatically in your default browser.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import threading
import uuid
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """One handler thread per request so slow Reolink I/O cannot block /state or the HTML."""

    daemon_threads = True


_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_DEFAULT_PANORAMA = _PROJECT_ROOT / "stitched.png"
_STATE_FILE = _PROJECT_ROOT / "scratchpads" / "sim_ptz_state.json"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_state():
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    return {"pan": 180.0, "tilt": 76.3, "fov_h": 60.0}


def _state_for_api() -> dict:
    """Canonical pan / tilt / fov_h for GET /state and UI sync."""
    d = _load_state()
    try:
        from tools.sim_ptz_watch import get_watch_settings

        w = get_watch_settings()
    except Exception:
        w = {
            "enabled": False,
            "move_delay_seconds": 0.45,
            "inference_delay_seconds": 0.35,
        }
    backend = getattr(ViewerHandler, "backend", "sim")
    out = {
        "pan": float(d.get("pan", 180.0)),
        "tilt": float(d.get("tilt", 76.3)),
        "fov_h": float(d.get("fov_h", 60.0)),
        "watch_along": w["enabled"],
        "move_delay_seconds": w["move_delay_seconds"],
        "inference_delay_seconds": w["inference_delay_seconds"],
        "backend": backend,
        "pan_range": float(getattr(ViewerHandler, "pan_limit", 360.0)),
        "tilt_range": float(getattr(ViewerHandler, "tilt_range", 152.7)),
    }
    return out


def _save_state(pan, tilt, fov_h=None):
    """Persist PTZ state; merges with existing file so fov_h and watch_* are kept."""
    try:
        from tools.sim_ptz_watch import save_position_state

        save_position_state(pan, tilt, fov_h)
    except Exception:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        base: dict = {}
        if _STATE_FILE.exists():
            try:
                base = json.loads(_STATE_FILE.read_text())
            except Exception:
                pass
        base["pan"] = round(float(pan), 2)
        base["tilt"] = round(float(tilt), 2)
        if fov_h is not None:
            base["fov_h"] = round(float(fov_h), 1)
        elif "fov_h" not in base:
            base["fov_h"] = 60.0
        _STATE_FILE.write_text(json.dumps(base))


# Agentic mission jobs (background threads + progress for web UI)
_MISSION_JOBS: dict = {}
_MISSION_LOCK = threading.Lock()


def _mission_public(job: dict) -> dict:
    """JSON-serializable job view (no threading.Event)."""
    return {k: v for k, v in job.items() if k != "cancel_event"}


def _extract_viewport(pano_img, pan, tilt, fov_h, fov_v, tilt_range, ppd):
    """Crop the viewport region from the panorama, handling wrap-around."""
    from PIL import Image
    cx = pan * ppd
    cy = (tilt_range - tilt) * ppd
    hw = fov_h * ppd / 2
    hh = fov_v * ppd / 2
    top = max(0, int(cy - hh))
    bot = min(pano_img.height, int(cy + hh))
    left = int(cx - hw)
    right = int(cx + hw)
    vp_w = right - left
    vp_h = bot - top

    if left < 0:
        lp = pano_img.crop((left + pano_img.width, top, pano_img.width, bot))
        rp = pano_img.crop((0, top, right, bot))
        out = Image.new("RGB", (vp_w, vp_h))
        out.paste(lp, (0, 0))
        out.paste(rp, (lp.width, 0))
        return out
    if right > pano_img.width:
        lp = pano_img.crop((left, top, pano_img.width, bot))
        rp = pano_img.crop((0, top, right - pano_img.width, bot))
        out = Image.new("RGB", (vp_w, vp_h))
        out.paste(lp, (0, 0))
        out.paste(rp, (lp.width, 0))
        return out
    return pano_img.crop((left, top, right, bot))


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MSA Simulated PTZ Viewer</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0c0f14;
    --surface: #161b24;
    --border: #2a3040;
    --accent: #e63946;
    --accent-glow: rgba(230, 57, 70, 0.35);
    --green: #00ff41;
    --cyan: #00e5ff;
    --amber: #ffd600;
    --text: #e8e8e8;
    --muted: #8892a4;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Outfit', sans-serif;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    overflow: hidden;
    height: 100vh;
  }
  #header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    height: 44px;
  }
  #header h1 {
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
  }
  #header .subtitle {
    font-size: 12px;
    color: var(--muted);
    font-weight: 300;
  }
  #header .model-pills {
    margin-left: auto;
    display: flex;
    gap: 6px;
  }
  .pill {
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .pill.ok { border-color: var(--green); color: var(--green); }
  .pill.missing { opacity: 0.4; }
  #main {
    display: grid;
    grid-template-rows: 160px 1fr;
    height: calc(100vh - 44px);
  }
  #panorama-section {
    position: relative;
    background: #000;
    border-bottom: 1px solid var(--border);
    overflow: hidden;
    cursor: crosshair;
  }
  #panorama-canvas {
    display: block;
    width: 100%;
    height: 100%;
  }
  #crosshair {
    position: absolute;
    pointer-events: none;
    border: 2px solid var(--accent);
    box-shadow: 0 0 10px var(--accent-glow), inset 0 0 10px var(--accent-glow);
    transition: left 0.12s, top 0.12s, width 0.12s, height 0.12s;
  }
  #viewport-section {
    display: grid;
    grid-template-columns: 1fr 330px;
    gap: 0;
    overflow: hidden;
  }
  #camera-feed {
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
  }
  #feed-canvas {
    display: block;
  }
  #controls {
    background: var(--surface);
    border-left: 1px solid var(--border);
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow-y: auto;
  }
  .ctrl-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .ctrl-group label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    font-weight: 600;
  }
  .slider-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .slider-row .val {
    font-family: var(--mono);
    font-size: 14px;
    font-weight: 700;
    min-width: 60px;
    text-align: right;
  }
  input[type="range"] {
    -webkit-appearance: none;
    flex: 1;
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    box-shadow: 0 0 6px var(--accent-glow);
  }
  .btn-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 3px;
  }
  .btn-grid button, .detect-grid button {
    font-family: var(--mono);
    font-size: 11px;
    padding: 6px 4px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s;
  }
  .btn-grid button:hover, .detect-grid button:hover {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .detect-grid {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .detect-row {
    display: flex;
    gap: 4px;
  }
  .detect-row button {
    min-width: 90px;
    white-space: nowrap;
    font-family: var(--mono);
    font-size: 11px;
    padding: 6px 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s;
  }
  .detect-row input[type="text"] {
    flex: 1;
    font-family: var(--mono);
    font-size: 11px;
    padding: 5px 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    border-radius: 4px;
    outline: none;
    min-width: 0;
  }
  .detect-row input[type="text"]:focus {
    border-color: var(--accent);
  }
  .detect-row input[type="text"]::placeholder {
    color: #555;
  }
  .detect-row button.yolo { border-color: #00ff4155; color: var(--green); }
  .detect-row button.yolo:hover { background: #00ff4133; }
  .detect-row button.bioclip { border-color: #00e5ff55; color: var(--cyan); }
  .detect-row button.bioclip:hover { background: #00e5ff33; }
  .detect-row button.gemma4 { border-color: #ffab4055; color: #ffab40; }
  .detect-row button.gemma4:hover { background: #ffab4033; }
  button.caption-btn {
    width: 100%;
    font-family: var(--mono);
    font-size: 11px;
    padding: 6px 8px;
    border: 1px solid #ce93d855;
    background: var(--bg);
    color: #ce93d8;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s;
  }
  button.caption-btn:hover { background: #ce93d833; }
  button.clear-btn {
    width: 100%;
    font-family: var(--mono);
    font-size: 11px;
    padding: 5px 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--muted);
    cursor: pointer;
    border-radius: 4px;
  }
  button.clear-btn:hover { background: #333; }
  button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  #results-panel {
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.5;
    color: var(--muted);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
    max-height: 200px;
    overflow-y: auto;
    flex-shrink: 0;
  }
  .result-header {
    color: var(--text);
    font-weight: 700;
    margin-bottom: 4px;
  }
  .result-item {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    border-bottom: 1px solid #1a1f2a;
  }
  .result-label { color: var(--text); }
  .result-conf { color: var(--green); }
  .result-caption {
    color: #ce93d8;
    font-style: italic;
    word-wrap: break-word;
  }
  .result-error { color: var(--accent); }
  .result-empty { color: var(--muted); font-style: italic; }
  .hotkeys {
    font-size: 10px;
    color: var(--muted);
    line-height: 1.6;
  }
  .hotkeys kbd {
    font-family: var(--mono);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0px 4px;
    font-size: 9px;
  }
  .recording-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    animation: blink 1.5s infinite;
    margin-right: 6px;
    vertical-align: middle;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-left: 6px;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2px 0;
  }
  .mission-input {
    width: 100%;
    font-family: var(--mono);
    font-size: 11px;
    padding: 6px 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    border-radius: 4px;
    outline: none;
    box-sizing: border-box;
  }
  .mission-input:focus { border-color: #ce93d8; }
  .mission-actions {
    display: flex;
    gap: 6px;
    margin-top: 6px;
  }
  .mission-start {
    flex: 1;
    font-family: var(--mono);
    font-size: 11px;
    padding: 7px 8px;
    border: 1px solid #ce93d855;
    background: var(--bg);
    color: #ce93d8;
    cursor: pointer;
    border-radius: 4px;
  }
  .mission-start:hover:not(:disabled) { background: #ce93d833; }
  .mission-stop {
    font-family: var(--mono);
    font-size: 11px;
    padding: 7px 12px;
    border: 1px solid var(--accent);
    background: var(--bg);
    color: var(--accent);
    cursor: pointer;
    border-radius: 4px;
  }
  .mission-stop:hover:not(:disabled) { background: #e6394622; }
  .mission-progress-wrap { margin-top: 8px; }
  .mission-bar {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .mission-bar-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #ce93d8, #e63946);
    border-radius: 3px;
    transition: width 0.15s ease-out;
  }
  .mission-line {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--cyan);
    margin-top: 6px;
    line-height: 1.4;
  }
  .mission-log {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px;
    max-height: 100px;
    overflow-y: auto;
    margin-top: 6px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .mission-badge {
    font-family: var(--mono);
    font-size: 9px;
    color: #ce93d8;
    margin-left: 6px;
  }
  .sync-hint {
    font-size: 9px;
    color: var(--muted);
    margin-top: 4px;
  }
  .watch-along-group label.watch-along-label {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 11px;
    color: var(--text);
    cursor: pointer;
    line-height: 1.35;
  }
  .watch-along-group input[type="checkbox"] {
    margin-top: 2px;
    flex-shrink: 0;
  }
</style>
</head>
<body>

<div id="header">
  <h1><span class="recording-dot"></span>MSA PTZ</h1>
  <span class="subtitle">{{HEADER_SUBTITLE}}</span>
  <div class="model-pills" id="model-pills"></div>
</div>

<div id="main">
  <div id="panorama-section">
    <canvas id="panorama-canvas"></canvas>
    <div id="crosshair"></div>
  </div>
  <div id="viewport-section">
    <div id="camera-feed">
      <canvas id="feed-canvas"></canvas>
    </div>
    <div id="controls">

      <div class="ctrl-group">
        <label>Pan</label>
        <div class="slider-row">
          <input type="range" id="pan-slider" min="0" max="{{PAN_SLIDER_MAX}}" step="0.5" value="180">
          <span class="val" id="pan-val">180.0&deg;</span>
        </div>
      </div>
      <div class="ctrl-group">
        <label>Tilt</label>
        <div class="slider-row">
          <input type="range" id="tilt-slider" min="0" max="{{TILT_RANGE}}" step="0.5" value="76.3">
          <span class="val" id="tilt-val">76.3&deg;</span>
        </div>
      </div>
      <div class="ctrl-group">
        <label>FOV</label>
        <div class="slider-row">
          <input type="range" id="fov-slider" min="10" max="120" step="1" value="{{FOV_INIT}}">
          <span class="val" id="fov-val">{{FOV_INIT}}&deg;</span>
        </div>
      </div>
      <div class="sync-hint">Position syncs from disk every ~0.4s (MCP / Python / other tabs).</div>
      <div id="ptz-hardware-status" class="sync-hint" style="color:#e63946;min-height:14px;"></div>

      <div class="ctrl-group watch-along-group">
        <label class="watch-along-label" title="Adds pauses after each simulated move and each detection so you can follow agent/mission runs in this UI.">
          <input type="checkbox" id="watch-along"> Watch along (slow PTZ + inference)
        </label>
        <div class="sync-hint">Applies to agent tools, MCP, CLI missions, and Run scan — not to your slider drags.</div>
      </div>

      <div class="ctrl-group">
        <label>Move</label>
        <div class="btn-grid">
          <div></div>
          <button onclick="moveTilt(2)">&#9650; Up</button>
          <div></div>
          <button onclick="movePan(-2)">&#9664; L</button>
          <button onclick="goHome()">Home</button>
          <button onclick="movePan(2)">R &#9654;</button>
          <div></div>
          <button onclick="moveTilt(-2)">&#9660; Dn</button>
          <div></div>
        </div>
      </div>

      <div class="ctrl-group">
        <label>Presets</label>
        <div class="btn-grid">
          <button onclick="goTo(45, TILT_MID)">NE</button>
          <button onclick="goTo(135, TILT_MID)">SE</button>
          <button onclick="goTo(225, TILT_MID)">SW</button>
          <button onclick="goTo(315, TILT_MID)">NW</button>
          <button onclick="goTo(180, TILT_MAX * 0.8)">Sky</button>
          <button onclick="goTo(180, TILT_MAX * 0.2)">Ground</button>
        </div>
      </div>

      <hr class="divider">

      <div class="ctrl-group">
        <label>Detection</label>
        <div class="detect-grid">
          <div class="detect-row">
            <button class="yolo" id="btn-yolo" onclick="runDetection('yolo')">YOLO</button>
            <input type="text" id="yolo-targets" value="*" placeholder="target (* = all)">
          </div>
          <div class="detect-row">
            <button class="bioclip" id="btn-bioclip" onclick="runDetection('bioclip')">BioCLIP</button>
            <input type="text" id="bioclip-taxon" value="" placeholder="e.g. Mammalia or Animalia Chordata Mammalia" title="Matches any ranked prediction that fits this lineage (not only top-1). Commas or spaces OK.">
            <label class="watch-along-label" style="margin-top:6px;"><input type="checkbox" id="bioclip-debug"> BioCLIP debug (JSON + logs)</label>
          </div>
          <div class="detect-row">
            <button class="gemma4" id="btn-gemma4" onclick="runDetection('gemma4')">Gemma4</button>
            <input type="text" id="gemma4-target" value="" placeholder="detect hint (empty = all)">
          </div>
          <div class="detect-row" style="margin-top:4px;">
            <span style="font-size:10px;color:var(--muted);width:52px;flex-shrink:0;">soft tok</span>
            <input type="number" id="gemma4-soft-tokens" value="280" min="70" max="1120" step="70" style="max-width:88px;" title="Gemma4 visual token budget (Ollama; 70–1120)">
          </div>
          <button class="caption-btn" id="btn-caption" onclick="runCaption('bioclip')">Caption (BioCLIP)</button>
          <button class="caption-btn" id="btn-caption-gemma" onclick="runCaption('gemma4')" style="margin-top:4px;border-color:#ffab4055;color:#ffab40;">Caption (Gemma4)</button>
          <input type="text" id="gemma-caption-prompt" value="" placeholder="optional Gemma caption prompt" style="width:100%;margin-top:4px;font-size:11px;padding:5px 8px;border-radius:4px;border:1px solid var(--border);background:var(--bg);color:var(--text);">
          <button class="clear-btn" onclick="clearDetections()">Clear detections</button>
        </div>
      </div>

      <hr class="divider">

      <div class="ctrl-group">
        <label>Agentic mission <span id="mission-badge" class="mission-badge" style="display:none;"></span></label>
        <input type="text" id="mission-input" class="mission-input"
          placeholder="e.g. scan for animals, find cows, random things…">
        <div class="mission-actions">
          <button type="button" class="mission-start" id="btn-mission-start" onclick="startAgenticMission()">Run scan</button>
          <button type="button" class="mission-stop" id="btn-mission-stop" onclick="stopAgenticMission()" disabled>Stop</button>
        </div>
        <div id="mission-progress-wrap" class="mission-progress-wrap" style="display:none;">
          <div class="mission-bar"><div class="mission-bar-fill" id="mission-bar-fill"></div></div>
          <div class="mission-line" id="mission-status-line"></div>
          <pre class="mission-log" id="mission-log"></pre>
        </div>
      </div>

      <div class="ctrl-group" style="flex:1; min-height:60px;">
        <label>Results <span id="detect-status"></span></label>
        <div id="results-panel">Position the camera, then click a detection button.</div>
      </div>

      <div class="ctrl-group hotkeys">
        <label>Keys</label>
        <kbd>&larr;</kbd><kbd>&rarr;</kbd> pan
        <kbd>&uarr;</kbd><kbd>&darr;</kbd> tilt
        <kbd>+</kbd><kbd>-</kbd> zoom
        <kbd>H</kbd> home
        <kbd>S</kbd> save
      </div>
    </div>
  </div>
</div>

<script>
const IMG_SRC = '/panorama';
const IS_REOLINK = {{IS_REOLINK}};
const PAN_RANGE = {{PAN_RANGE}};
const TILT_RANGE = {{TILT_RANGE}};
const TILT_MID = TILT_RANGE / 2;
const TILT_MAX = TILT_RANGE;

let pan = {{PAN_INIT}};
let tilt = {{TILT_INIT}};
let fovH = {{FOV_INIT}};
let fovV = fovH * 9 / 16;

let statePollTimer = null;

let detections = [];
let detectionModel = '';
let detViewportW = 1;
let detViewportH = 1;
let detecting = false;
let captionText = '';

let missionRunning = false;
let missionJobId = null;
let missionPollTimer = null;

const COLORS = {yolo: '#00ff41', bioclip: '#00e5ff', gemma4: '#ffab40'};

const panoCanvas = document.getElementById('panorama-canvas');
const panoCtx = panoCanvas.getContext('2d');
const feedCanvas = document.getElementById('feed-canvas');
const feedCtx = feedCanvas.getContext('2d');
const crosshair = document.getElementById('crosshair');
const panSlider = document.getElementById('pan-slider');
const tiltSlider = document.getElementById('tilt-slider');
const fovSlider = document.getElementById('fov-slider');

let panoImg = new Image();
let liveFrameImg = new Image();
let imgW = 1, imgH = 1, ppd = 1;
/* Feed layout (must exist before Reolink path calls finishViewerInit -> render) */
let feedDx = 0, feedDy = 0, feedDw = 1, feedDh = 1;

function refreshLiveFrame() {
  if (!IS_REOLINK) return;
  const img = new Image();
  img.onload = () => {
    liveFrameImg = img;
    imgW = img.naturalWidth;
    imgH = img.naturalHeight;
    const st = document.getElementById('ptz-hardware-status');
    if (st) st.textContent = '';
    render();
  };
  img.onerror = () => {
    const st = document.getElementById('ptz-hardware-status');
    if (st) {
      st.textContent = 'Live frame failed (/live). Check REOLINK_* env, network, and server terminal logs.';
    }
  };
  img.src = '/live?t=' + Date.now();
}

function finishViewerInit() {
  resize();
  updateUI();
  loadModels();
  fetch('/state').then(r => r.json()).then(s => {
    const wa = document.getElementById('watch-along');
    if (wa && typeof s.watch_along === 'boolean') wa.checked = s.watch_along;
  }).catch(() => {});
  if (statePollTimer) clearInterval(statePollTimer);
  statePollTimer = setInterval(pollServerState, 400);
}

if (IS_REOLINK) {
  ppd = 1;
  finishViewerInit();
  refreshLiveFrame();
  setInterval(refreshLiveFrame, 280);
} else {
  panoImg.onload = () => {
    imgW = panoImg.naturalWidth;
    imgH = panoImg.naturalHeight;
    ppd = imgW / PAN_RANGE;
    finishViewerInit();
  };
  panoImg.src = IMG_SRC;
}

function resize() {
  const section = document.getElementById('panorama-section');
  panoCanvas.width = section.clientWidth;
  panoCanvas.height = section.clientHeight;

  const feedSection = document.getElementById('camera-feed');
  feedCanvas.width = feedSection.clientWidth;
  feedCanvas.height = feedSection.clientHeight;
}

/**
 * Mirror tools.ptz_viewer._extract_viewport (Python): integer crop bounds so the
 * canvas samples the same pixels as the server-side PIL crop used for detection.
 */
function viewportCropRect() {
  const vpW = fovH * ppd;
  const vpH = fovV * ppd;
  const cx = pan * ppd;
  const cy = (TILT_RANGE - tilt) * ppd;
  const hw = vpW / 2;
  const hh = vpH / 2;
  const top = Math.max(0, Math.trunc(cy - hh));
  const bot = Math.min(imgH, Math.trunc(cy + hh));
  const left = Math.trunc(cx - hw);
  const right = Math.trunc(cx + hw);
  const cropW = right - left;
  const cropH = bot - top;
  return { left, right, top, bot, cropW, cropH };
}

function render() {
  const cw = panoCanvas.width, ch = panoCanvas.height;
  if (IS_REOLINK) {
    panoCtx.fillStyle = '#1a1a22';
    panoCtx.fillRect(0, 0, cw, ch);
    panoCtx.fillStyle = '#8b8b9a';
    panoCtx.font = '14px "JetBrains Mono", monospace';
    panoCtx.fillText('Reolink hardware PTZ (no panorama strip)', 16, 28);
    panoCtx.fillText('Live stream is on the right.', 16, 48);
  } else {
    panoCtx.drawImage(panoImg, 0, 0, cw, ch);
  }

  if (!IS_REOLINK) {
    const scaleX = cw / imgW;
    const scaleY = ch / imgH;
    const vpW = fovH * ppd;
    const vpH = fovV * ppd;
    const cx = pan * ppd;
    const cy = (TILT_RANGE - tilt) * ppd;
    let left = (cx - vpW / 2) * scaleX;
    let top_ = (cy - vpH / 2) * scaleY;
    let w = vpW * scaleX;
    let h = vpH * scaleY;

    if (left < 0) {
      crosshair.style.display = 'none';
      panoCtx.strokeStyle = '#e63946'; panoCtx.lineWidth = 2;
      panoCtx.shadowColor = 'rgba(230,57,70,0.5)'; panoCtx.shadowBlur = 6;
      panoCtx.strokeRect(left + cw, top_, -left, h);
      panoCtx.strokeRect(0, top_, w + left, h);
      panoCtx.shadowBlur = 0;
    } else if (left + w > cw) {
      crosshair.style.display = 'none';
      panoCtx.strokeStyle = '#e63946'; panoCtx.lineWidth = 2;
      panoCtx.shadowColor = 'rgba(230,57,70,0.5)'; panoCtx.shadowBlur = 6;
      panoCtx.strokeRect(left, top_, cw - left, h);
      panoCtx.strokeRect(0, top_, (left + w) - cw, h);
      panoCtx.shadowBlur = 0;
    } else {
      crosshair.style.left = left + 'px';
      crosshair.style.top = top_ + 'px';
      crosshair.style.width = w + 'px';
      crosshair.style.height = h + 'px';
      crosshair.style.display = 'block';
    }
  } else {
    crosshair.style.display = 'none';
  }

  const fcw = feedCanvas.width, fch = feedCanvas.height;
  feedCtx.fillStyle = '#000';
  feedCtx.fillRect(0, 0, fcw, fch);

  const aspectVP = 16 / 9;
  const aspectFeed = fcw / fch;
  let dw, dh, dx, dy;
  if (aspectVP > aspectFeed) {
    dw = fcw; dh = fcw / aspectVP; dx = 0; dy = (fch - dh) / 2;
  } else {
    dh = fch; dw = fch * aspectVP; dx = (fcw - dw) / 2; dy = 0;
  }
  feedDx = dx; feedDy = dy; feedDw = dw; feedDh = dh;

  if (IS_REOLINK && liveFrameImg && liveFrameImg.naturalWidth > 0) {
    feedCtx.drawImage(liveFrameImg, dx, dy, dw, dh);
  } else if (!IS_REOLINK) {
    const cr = viewportCropRect();
    const { left: cLeft, right: cRight, top: cTop, cropW, cropH } = cr;

    if (cLeft < 0) {
      const leftW = -cLeft;
      const rightW = cropW - leftW;
      const ratio = leftW / cropW;
      feedCtx.drawImage(panoImg, imgW + cLeft, cTop, leftW, cropH, dx, dy, dw * ratio, dh);
      feedCtx.drawImage(panoImg, 0, cTop, rightW, cropH, dx + dw * ratio, dy, dw * (1 - ratio), dh);
    } else if (cRight > imgW) {
      const leftW = imgW - cLeft;
      const rightW = cropW - leftW;
      const ratio = leftW / cropW;
      feedCtx.drawImage(panoImg, cLeft, cTop, leftW, cropH, dx, dy, dw * ratio, dh);
      feedCtx.drawImage(panoImg, 0, cTop, rightW, cropH, dx + dw * ratio, dy, dw * (1 - ratio), dh);
    } else {
      feedCtx.drawImage(panoImg, cLeft, cTop, cropW, cropH, dx, dy, dw, dh);
    }
  } else {
    feedCtx.fillStyle = '#444';
    feedCtx.font = '13px "JetBrains Mono", monospace';
    feedCtx.fillText('Waiting for camera frame...', dx + 12, dy + 28);
  }

  /* Detection bounding boxes — coords are in server viewport pixels (image_size) */
  if (detections.length > 0) {
    const color = COLORS[detectionModel] || '#00ff41';
    const sxD = dw / detViewportW;
    const syD = dh / detViewportH;
    feedCtx.font = '13px "JetBrains Mono", monospace';

    detections.forEach(det => {
      const bx = dx + det.bbox[0] * sxD;
      const by = dy + det.bbox[1] * syD;
      const bw = (det.bbox[2] - det.bbox[0]) * sxD;
      const bh = (det.bbox[3] - det.bbox[1]) * syD;

      feedCtx.strokeStyle = color;
      feedCtx.lineWidth = 2;
      feedCtx.strokeRect(bx, by, bw, bh);

      const lbl = det.label + ' ' + (det.confidence * 100).toFixed(0) + '%';
      const tw = feedCtx.measureText(lbl).width + 8;
      feedCtx.fillStyle = 'rgba(0,0,0,0.75)';
      feedCtx.fillRect(bx, by - 18, tw, 18);
      feedCtx.fillStyle = color;
      feedCtx.fillText(lbl, bx + 4, by - 4);
    });
  }

  /* Caption overlay */
  if (captionText) {
    const pad = 10;
    feedCtx.font = '13px "JetBrains Mono", monospace';
    const lines = wrapText(captionText, dw - pad * 2);
    const lh = 17;
    const boxH = lines.length * lh + pad;
    feedCtx.fillStyle = 'rgba(0,0,0,0.7)';
    feedCtx.fillRect(dx, dy, dw, boxH);
    feedCtx.fillStyle = '#ce93d8';
    lines.forEach((line, i) => feedCtx.fillText(line, dx + pad, dy + pad + i * lh + 12));
  }

  /* HUD bar */
  feedCtx.fillStyle = 'rgba(0,0,0,0.55)';
  feedCtx.fillRect(dx, dy + dh - 24, dw, 24);
  feedCtx.font = '11px "JetBrains Mono", monospace';
  feedCtx.fillStyle = '#e63946';
  feedCtx.fillText('PAN ' + pan.toFixed(1) + '\u00b0  TILT ' + tilt.toFixed(1) + '\u00b0  FOV ' + fovH + '\u00b0', dx + 8, dy + dh - 7);

  /* Crosshairs on feed */
  feedCtx.strokeStyle = 'rgba(230,57,70,0.25)';
  feedCtx.lineWidth = 1;
  feedCtx.beginPath();
  feedCtx.moveTo(dx + dw / 2, dy); feedCtx.lineTo(dx + dw / 2, dy + dh);
  feedCtx.moveTo(dx, dy + dh / 2); feedCtx.lineTo(dx + dw, dy + dh / 2);
  feedCtx.stroke();
}

function wrapText(text, maxW) {
  const words = text.split(' ');
  const lines = []; let cur = '';
  for (const w of words) {
    const test = cur ? cur + ' ' + w : w;
    if (feedCtx.measureText(test).width > maxW && cur) {
      lines.push(cur); cur = w;
    } else { cur = test; }
  }
  if (cur) lines.push(cur);
  return lines;
}

function updateUI() {
  // Camera is about to move (or already moved): the previous detection
  // overlay is no longer aligned with the live frame, so wipe it before
  // we re-render. clearDetectionsOnMove is a no-op when nothing is shown.
  if (!missionRunning) clearDetectionsOnMove();
  document.getElementById('pan-val').innerHTML = pan.toFixed(1) + '&deg;';
  document.getElementById('tilt-val').innerHTML = tilt.toFixed(1) + '&deg;';
  document.getElementById('fov-val').innerHTML = fovH + '&deg;';
  panSlider.value = pan;
  tiltSlider.value = tilt;
  fovSlider.value = fovH;
  render();
  saveState();
}

var _saveStateTimer = null;
function saveState() {
  if (missionRunning) return;
  // Reolink: debounce slider/drag input to one POST so the motor isn't
  // chasing an updated absolute target 30 times per second.
  if (IS_REOLINK) {
    if (_saveStateTimer) clearTimeout(_saveStateTimer);
    _saveStateTimer = setTimeout(_saveStateNow, 180);
    return;
  }
  _saveStateNow();
}

function clearDetectionsOnMove() {
  if (detections.length === 0 && !captionText) return;
  detections = [];
  detectionModel = '';
  captionText = '';
  // Caller is expected to render() right after this.
}

function _saveStateNow() {
  _saveStateTimer = null;
  if (missionRunning) return;
  const wa = document.getElementById('watch-along');
  const st = document.getElementById('ptz-hardware-status');
  fetch('/state', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      pan, tilt, fov_h: fovH,
      watch_along: wa ? wa.checked : false,
    })
  })
    .then(async (r) => {
      let j = {};
      try {
        j = await r.json();
      } catch (_) {}
      return { ok: r.ok, j };
    })
    .then(({ ok, j }) => {
      if (!IS_REOLINK) return;
      if (!ok || (j && j.ok === false)) {
        const msg = (j && j.error) ? j.error : ('HTTP ' + (ok ? 'ok' : 'error'));
        if (st) st.textContent = 'PTZ move failed: ' + msg;
        console.error('PTZ /state', j);
      } else if (st && !j.error) {
        st.textContent = '';
      }
      if (IS_REOLINK && ok && j && j.ok !== false) refreshLiveFrame();
    })
    .catch((e) => {
      if (st && IS_REOLINK) st.textContent = 'PTZ sync failed: ' + e;
    });
}

function postWatchAlongOnly() {
  const wa = document.getElementById('watch-along');
  if (!wa) return;
  fetch('/state', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      pan, tilt, fov_h: fovH,
      watch_along: wa.checked,
    })
  }).catch(() => {});
}

function _sliderInteractionActive() {
  // Don't let the poll overwrite local slider state while:
  //   (a) a debounced saveState() is pending (the motor hasn't received the
  //       user's latest target yet), or
  //   (b) the user is actively holding a slider thumb down.
  if (_saveStateTimer) return true;
  if (_sliderActive) return true;
  // Grace period after release so we don't snap back to a stale position
  // before the camera has finished moving.
  if (_sliderReleasedAt && (performance.now() - _sliderReleasedAt) < 1500) return true;
  return false;
}

function pollServerState() {
  if (missionRunning) return;
  if (_sliderInteractionActive()) return;
  fetch('/state')
    .then(r => r.json())
    .then(s => {
      if (!s || typeof s.pan !== 'number') return;
      if (_sliderInteractionActive()) return;
      const ep = 0.08, et = 0.08, ef = 0.25;
      if (
        Math.abs(pan - s.pan) > ep ||
        Math.abs(tilt - s.tilt) > et ||
        Math.abs(fovH - s.fov_h) > ef
      ) {
        pan = s.pan;
        tilt = s.tilt;
        fovH = Math.round(s.fov_h);
        fovV = fovH * 9 / 16;
        panSlider.value = pan;
        tiltSlider.value = tilt;
        fovSlider.value = fovH;
        document.getElementById('pan-val').innerHTML = pan.toFixed(1) + '&deg;';
        document.getElementById('tilt-val').innerHTML = tilt.toFixed(1) + '&deg;';
        document.getElementById('fov-val').innerHTML = fovH + '&deg;';
        render();
      }
      const wa = document.getElementById('watch-along');
      if (wa && typeof s.watch_along === 'boolean' && wa.checked !== s.watch_along) {
        wa.checked = s.watch_along;
      }
    })
    .catch(() => {});
}

function jogRel(dp, dt) {
  if (missionRunning) return;
  // Any user-initiated move invalidates the existing detection overlay.
  clearDetectionsOnMove();
  render();
  const wa = document.getElementById('watch-along');
  const st = document.getElementById('ptz-hardware-status');
  fetch('/state', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      jog_pan: dp,
      jog_tilt: dt,
      fov_h: fovH,
      watch_along: wa ? wa.checked : false,
    })
  })
    .then(async (r) => {
      let j = {};
      try { j = await r.json(); } catch (_) {}
      return { ok: r.ok, j };
    })
    .then(({ ok, j }) => {
      if (!ok || (j && j.ok === false)) {
        const msg = (j && j.error) ? j.error : ('HTTP ' + (ok ? 'ok' : 'error'));
        if (st) st.textContent = 'PTZ jog failed: ' + msg;
        return;
      }
      if (st) st.textContent = '';
      if (typeof j.pan === 'number') pan = j.pan;
      if (typeof j.tilt === 'number') tilt = j.tilt;
      if (typeof j.fov_h === 'number') fovH = Math.round(j.fov_h);
      fovV = fovH * 9 / 16;
      document.getElementById('pan-val').innerHTML = pan.toFixed(1) + '&deg;';
      document.getElementById('tilt-val').innerHTML = tilt.toFixed(1) + '&deg;';
      document.getElementById('fov-val').innerHTML = fovH + '&deg;';
      panSlider.value = pan;
      tiltSlider.value = tilt;
      fovSlider.value = fovH;
      render();
      refreshLiveFrame();
    })
    .catch((e) => { if (st) st.textContent = 'PTZ jog failed: ' + e; });
}

function movePan(deg) {
  if (IS_REOLINK) {
    jogRel(deg, 0);
    return;
  }
  pan = ((pan + deg) % PAN_RANGE + PAN_RANGE) % PAN_RANGE;
  updateUI();
}
function moveTilt(deg) {
  if (IS_REOLINK) {
    jogRel(0, deg);
    return;
  }
  const halfV = fovV / 2;
  tilt = Math.max(halfV, Math.min(TILT_RANGE - halfV, tilt + deg));
  updateUI();
}
function goTo(p, t) {
  pan = p;
  tilt = t;
  if (IS_REOLINK) {
    pan = Math.max(0, Math.min(PAN_RANGE, pan));
    tilt = Math.max(0, Math.min(TILT_RANGE, tilt));
  } else {
    const halfV = fovV / 2;
    tilt = Math.max(halfV, Math.min(TILT_RANGE - halfV, tilt));
  }
  updateUI();
}
function goHome() { goTo(PAN_RANGE / 2, TILT_MID); }

var _sliderActive = false;
var _sliderReleasedAt = 0;
function _bindSliderGrab(el) {
  const press = () => { _sliderActive = true; };
  const release = () => {
    _sliderActive = false;
    _sliderReleasedAt = performance.now();
  };
  el.addEventListener('pointerdown', press);
  el.addEventListener('pointerup', release);
  el.addEventListener('pointercancel', release);
  el.addEventListener('mousedown', press);
  el.addEventListener('mouseup', release);
  el.addEventListener('touchstart', press, { passive: true });
  el.addEventListener('touchend', release);
  el.addEventListener('keydown', press);
  el.addEventListener('keyup', release);
  el.addEventListener('blur', release);
}
_bindSliderGrab(panSlider);
_bindSliderGrab(tiltSlider);
_bindSliderGrab(fovSlider);

panSlider.addEventListener('input', () => { pan = parseFloat(panSlider.value); updateUI(); });
tiltSlider.addEventListener('input', () => { tilt = parseFloat(tiltSlider.value); updateUI(); });
fovSlider.addEventListener('input', () => {
  fovH = parseInt(fovSlider.value);
  fovV = fovH * 9 / 16;
  updateUI();
});

const watchAlongEl = document.getElementById('watch-along');
if (watchAlongEl) {
  watchAlongEl.addEventListener('change', postWatchAlongOnly);
}

document.getElementById('panorama-section').addEventListener('click', e => {
  if (IS_REOLINK) return;
  const rect = panoCanvas.getBoundingClientRect();
  pan = ((e.clientX - rect.left) / panoCanvas.width) * PAN_RANGE;
  tilt = TILT_RANGE - ((e.clientY - rect.top) / panoCanvas.height) * TILT_RANGE;
  const halfV = fovV / 2;
  tilt = Math.max(halfV, Math.min(TILT_RANGE - halfV, tilt));
  updateUI();
});

let dragging = false;
document.getElementById('panorama-section').addEventListener('mousedown', () => { dragging = true; });
document.addEventListener('mousemove', e => {
  if (!dragging) return;
  const rect = panoCanvas.getBoundingClientRect();
  pan = Math.max(0, Math.min(PAN_RANGE, ((e.clientX - rect.left) / panoCanvas.width) * PAN_RANGE));
  tilt = TILT_RANGE - Math.max(0, Math.min(TILT_RANGE, ((e.clientY - rect.top) / panoCanvas.height) * TILT_RANGE));
  const halfV = fovV / 2;
  tilt = Math.max(halfV, Math.min(TILT_RANGE - halfV, tilt));
  updateUI();
});
document.addEventListener('mouseup', () => { dragging = false; });

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const step = e.shiftKey ? (IS_REOLINK ? 1 : 5) : (IS_REOLINK ? 3 : 15);
  switch (e.key) {
    case 'ArrowLeft':  movePan(-step); e.preventDefault(); break;
    case 'ArrowRight': movePan(step);  e.preventDefault(); break;
    case 'ArrowUp':    moveTilt(step); e.preventDefault(); break;
    case 'ArrowDown':  moveTilt(-step);e.preventDefault(); break;
    case '+': case '=': fovH = Math.max(10, fovH - 5); fovV = fovH * 9/16; updateUI(); break;
    case '-': case '_': fovH = Math.min(120, fovH + 5); fovV = fovH * 9/16; updateUI(); break;
    case 'h': case 'H': goHome(); break;
    case 's': case 'S': saveSnapshot(); break;
  }
});

function saveSnapshot() {
  feedCanvas.toBlob(blob => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'sim_ptz_snapshot_' + Date.now() + '.jpg';
    a.click();
  }, 'image/jpeg', 0.92);
}

/* ---- Model availability ---- */
function loadModels() {
  fetch('/models').then(r => r.json()).then(data => {
    const c = document.getElementById('model-pills');
    c.innerHTML = '';
    for (const [name, ok] of Object.entries(data)) {
      const span = document.createElement('span');
      span.className = 'pill ' + (ok ? 'ok' : 'missing');
      span.textContent = name;
      c.appendChild(span);
    }
    if (!data.yolo) document.getElementById('btn-yolo').disabled = true;
    if (!data.bioclip) document.getElementById('btn-bioclip').disabled = true;
    if (!data.bioclip) document.getElementById('btn-caption').disabled = true;
    if (!data.gemma4) {
      const bg = document.getElementById('btn-gemma4');
      const bc = document.getElementById('btn-caption-gemma');
      if (bg) bg.disabled = true;
      if (bc) bc.disabled = true;
    }
  }).catch(() => {});
}

/* ---- Detection API ---- */
function setDetecting(v) {
  detecting = v;
  document.querySelectorAll('.detect-grid button').forEach(b => {
    if (!b.disabled) b.style.pointerEvents = v ? 'none' : '';
  });
  document.getElementById('detect-status').innerHTML = v ? '<span class="spinner"></span>' : '';
}

async function runDetection(model) {
  if (detecting || missionRunning) return;
  setDetecting(true);
  clearDetections();

  try {
    const body = {
      model,
      pan, tilt, fov_h: fovH, fov_v: fovV,
      targets: document.getElementById('yolo-targets').value || '*',
      target_taxon: document.getElementById('bioclip-taxon').value || '',
      target: document.getElementById('gemma4-target').value || '',
      max_soft_tokens: parseInt(document.getElementById('gemma4-soft-tokens').value, 10) || 280,
      bioclip_debug: !!(document.getElementById('bioclip-debug') || {}).checked,
    };
    const resp = await fetch('/detect', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await resp.json();

    if (data.error) {
      showResults('<div class="result-error">' + data.error + '</div>');
      return;
    }

    detections = data.detections || [];
    detectionModel = model;
    detViewportW = (data.image_size || [1,1])[0];
    detViewportH = (data.image_size || [1,1])[1];

    let html = '<div class="result-header">' + model.toUpperCase() + ' &mdash; ' + detections.length + ' detection(s) (' + data.elapsed_ms + 'ms)</div>';
    if (detections.length === 0) {
      html += '<div class="result-empty">No objects detected</div>';
    } else {
      detections.forEach(d => {
        html += '<div class="result-item"><span class="result-label">' + d.label + '</span><span class="result-conf">' + (d.confidence * 100).toFixed(1) + '%</span></div>';
      });
    }
    if (data.bioclip_debug) {
      html += '<pre class="bioclip-debug-pre" style="margin-top:10px;padding:10px;font-size:10px;overflow:auto;max-height:240px;background:#0a0d12;border:1px solid var(--border);color:var(--muted);text-align:left;">'
        + JSON.stringify(data.bioclip_debug, null, 2).replace(/</g, '&lt;')
        + '</pre>';
    }
    showResults(html);
    render();
  } catch (e) {
    showResults('<div class="result-error">Error: ' + e.message + '</div>');
  } finally {
    setDetecting(false);
  }
}

async function runCaption(model) {
  if (detecting || missionRunning) return;
  model = model || 'bioclip';
  setDetecting(true);
  captionText = '';

  try {
    const body = {
      model,
      pan, tilt, fov_h: fovH, fov_v: fovV,
      prompt: (document.getElementById('gemma-caption-prompt') || {}).value || '',
      max_soft_tokens: parseInt(document.getElementById('gemma4-soft-tokens').value, 10) || 280,
    };
    const resp = await fetch('/caption', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await resp.json();

    if (data.error) {
      showResults('<div class="result-error">' + data.error + '</div>');
      return;
    }

    captionText = data.caption || '';
    showResults('<div class="result-header">Caption ' + model.toUpperCase() + ' (' + data.elapsed_ms + 'ms)</div><div class="result-caption">' + captionText + '</div>');
    render();
  } catch (e) {
    showResults('<div class="result-error">Error: ' + e.message + '</div>');
  } finally {
    setDetecting(false);
  }
}

function clearDetections() {
  detections = [];
  detectionModel = '';
  captionText = '';
  render();
}

function showResults(html) {
  document.getElementById('results-panel').innerHTML = html;
}

async function startAgenticMission() {
  if (missionRunning || detecting) return;
  const mission = document.getElementById('mission-input').value.trim();
  if (!mission) {
    showResults('<div class="result-error">Enter a mission first.</div>');
    return;
  }
  missionRunning = true;
  panSlider.disabled = true;
  tiltSlider.disabled = true;
  fovSlider.disabled = true;
  document.getElementById('btn-mission-start').disabled = true;
  document.getElementById('btn-mission-stop').disabled = false;
  document.getElementById('mission-progress-wrap').style.display = 'block';
  document.getElementById('mission-badge').style.display = 'inline';
  document.getElementById('mission-badge').textContent = 'SCANNING';
  document.getElementById('mission-bar-fill').style.width = '0%';
  document.getElementById('mission-log').textContent = '';
  document.getElementById('mission-status-line').textContent = 'Starting…';
  clearDetections();
  captionText = '';

  try {
    const resp = await fetch('/api/mission/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        mission,
        pan, tilt,
        random_views: 10,
        pan_step_ratio: 0.82,
        max_pan_stops: 48
      })
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    missionJobId = data.job_id;
    missionPollTimer = setInterval(pollMissionStatus, 150);
  } catch (e) {
    endMissionUI(false);
    showResults('<div class="result-error">Mission: ' + e.message + '</div>');
  }
}

async function stopAgenticMission() {
  if (!missionJobId) return;
  try {
    await fetch('/api/mission/cancel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ id: missionJobId })
    });
  } catch (e) {}
}

function endMissionUI(keepProgressVisible) {
  missionRunning = false;
  panSlider.disabled = false;
  tiltSlider.disabled = false;
  fovSlider.disabled = false;
  document.getElementById('btn-mission-start').disabled = false;
  document.getElementById('btn-mission-stop').disabled = true;
  document.getElementById('mission-badge').style.display = 'none';
  if (missionPollTimer) {
    clearInterval(missionPollTimer);
    missionPollTimer = null;
  }
  if (!keepProgressVisible) {
    document.getElementById('mission-progress-wrap').style.display = 'none';
  }
}

async function pollMissionStatus() {
  if (!missionJobId) return;
  try {
    const r = await fetch('/api/mission/status?id=' + encodeURIComponent(missionJobId));
    if (!r.ok) {
      endMissionUI(false);
      missionJobId = null;
      return;
    }
    const j = await r.json();
    const tot = j.total || 1;
    const cur = j.current || 0;
    const pct = Math.min(100, Math.round(100 * cur / tot));
    document.getElementById('mission-bar-fill').style.width = pct + '%';
    document.getElementById('mission-status-line').textContent =
      (j.message || '') + ' | raw sightings: ' + (j.cumulative_raw || 0);
    if (j.log && j.log.length) {
      document.getElementById('mission-log').textContent = j.log.slice(-28).join('\n');
    }
    if (typeof j.pan === 'number' && typeof j.tilt === 'number') {
      pan = j.pan;
      tilt = j.tilt;
      panSlider.value = pan;
      tiltSlider.value = tilt;
      document.getElementById('pan-val').innerHTML = pan.toFixed(1) + '&deg;';
      document.getElementById('tilt-val').innerHTML = tilt.toFixed(1) + '&deg;';
    }
    if (j.last_detections && j.last_detections.length) {
      detections = j.last_detections;
      const m = (j.model || 'yolo').toLowerCase();
      detectionModel = (m === 'bioclip') ? 'bioclip' : (m === 'gemma4') ? 'gemma4' : 'yolo';
      detViewportW = j.image_w || 1;
      detViewportH = j.image_h || 1;
    } else {
      detections = [];
    }
    render();

    if (j.status === 'done' || j.status === 'error' || j.status === 'cancelled') {
      missionJobId = null;
      endMissionUI(true);
      document.getElementById('mission-bar-fill').style.width = '100%';
      if (j.status === 'done' && j.result) {
        const s = j.result.summary || {};
        let html = '<div class="result-header">Mission complete — ' + (j.result.elapsed_ms || 0) + ' ms</div>';
        html += '<div class="result-item"><span class="result-label">Unique (est.)</span><span class="result-conf">' + (s.unique_instances_estimated ?? 0) + '</span></div>';
        html += '<div class="result-item"><span class="result-label">Raw total</span><span class="result-conf">' + (s.total_detections_raw ?? 0) + '</span></div>';
        if (s.counts_by_label) {
          for (const [k, v] of Object.entries(s.counts_by_label)) {
            html += '<div class="result-item"><span class="result-label">' + k + '</span><span class="result-conf">' + v + '</span></div>';
          }
        }
        showResults(html);
      } else if (j.status === 'cancelled') {
        showResults('<div class="result-header">Mission cancelled</div><div class="result-empty">Partial frames may be in the log above.</div>');
      } else {
        showResults('<div class="result-error">' + (j.error || 'Mission failed') + '</div>');
      }
      saveState();
    }
  } catch (e) {
    console.error(e);
  }
}

window.addEventListener('resize', () => { resize(); render(); });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Agentic mission (background thread)
# ---------------------------------------------------------------------------

def _spawn_mission(payload: dict) -> dict:
    """Start ``run_mission`` in a daemon thread; return ``{job_id}`` or ``{error}``."""
    mission = (payload.get("mission") or "").strip()
    if not mission:
        return {"error": "mission text is required"}

    job_id = uuid.uuid4().hex
    cancel_ev = threading.Event()
    job: dict = {
        "id": job_id,
        "status": "queued",
        "current": 0,
        "total": 0,
        "pan": float(payload.get("pan", _load_state().get("pan", 180.0))),
        "tilt": float(payload.get("tilt", _load_state().get("tilt", 76.3))),
        "model": "",
        "last_detections": [],
        "frame_error": None,
        "image_w": 1,
        "image_h": 1,
        "cumulative_raw": 0,
        "message": "Starting…",
        "log": [],
        "result": None,
        "error": None,
        "cancel_event": cancel_ev,
    }
    with _MISSION_LOCK:
        _MISSION_JOBS[job_id] = job

    def worker():
        try:
            from tools.ptz_mission import run_mission

            def on_progress(info: dict):
                with _MISSION_LOCK:
                    j = _MISSION_JOBS.get(job_id)
                    if not j:
                        return
                    j["status"] = "running"
                    j["current"] = info["index"] + 1
                    j["total"] = info["total"]
                    j["pan"] = info["pan"]
                    j["tilt"] = info["tilt"]
                    j["model"] = info["model"]
                    j["last_detections"] = info.get("detections") or []
                    j["frame_error"] = info.get("frame_error")
                    j["image_w"] = info["image_size"][0]
                    j["image_h"] = info["image_size"][1]
                    j["cumulative_raw"] = info.get("cumulative_raw", 0)
                    n = info["index"] + 1
                    tot = info["total"]
                    j["message"] = f"Stop {n}/{tot} — pan {info['pan']}° tilt {info['tilt']}°"
                    line = j["message"]
                    if info.get("frame_error"):
                        line += f" — ERR {info['frame_error'][:80]}"
                    elif info.get("scene_match") is not None:
                        line += (
                            " — scene MATCH"
                            if info["scene_match"]
                            else " — scene no match"
                        )
                    elif info.get("detections"):
                        line += f" — {len(info['detections'])} detection(s)"
                    else:
                        line += " — 0 detections"
                    j.setdefault("log", []).append(line)
                    if len(j["log"]) > 100:
                        j["log"] = j["log"][-100:]
                _save_state(info["pan"], info["tilt"])

            tlt = payload.get("tilt")
            tilt_arg = float(tlt) if tlt is not None else None
            result = run_mission(
                mission,
                model=payload.get("model"),
                random_views=int(payload.get("random_views", 10)),
                pan_step_ratio=float(payload.get("pan_step_ratio", 0.82)),
                tilt_step_ratio=float(payload.get("tilt_step_ratio", 0.82)),
                tilt=tilt_arg,
                max_pan_stops=int(payload.get("max_pan_stops", 48)),
                max_tilt_rows=int(payload.get("max_tilt_rows", 64)),
                max_total_stops=int(payload.get("max_total_stops", 512)),
                cancel_event=cancel_ev,
                on_progress=on_progress,
            )
            with _MISSION_LOCK:
                j = _MISSION_JOBS.get(job_id)
                if not j:
                    return
                j["result"] = result
                if result.get("cancelled"):
                    j["status"] = "cancelled"
                    j["message"] = "Cancelled"
                elif result.get("ok"):
                    j["status"] = "done"
                    j["message"] = "Complete"
                else:
                    j["status"] = "error"
                    j["error"] = result.get("error") or "mission failed"
        except Exception as exc:
            logger.exception("Agentic mission failed")
            with _MISSION_LOCK:
                j = _MISSION_JOBS.get(job_id)
                if j:
                    j["status"] = "error"
                    j["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ViewerHandler(SimpleHTTPRequestHandler):
    image_path = ""
    backend = "sim"
    pan_limit = 360.0
    tilt_range = 152.7
    init_pan = 180.0
    init_tilt = 76.3
    init_fov = 60.0
    pano_pil = None     # PIL.Image kept in memory for detection viewports
    ppd = 1.0

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/ptz/ping":
            from tools import ptz_facade as _pf

            self._respond_json(
                {
                    "viewer_backend": getattr(ViewerHandler, "backend", "?"),
                    "ptz_backend_resolved": _pf.ptz_backend_name(),
                    "reolink_ip": os.environ.get("REOLINK_IP", "")
                    or os.environ.get("REOLINK_HOST", ""),
                    "reolink_user": os.environ.get("REOLINK_USER", ""),
                    "has_reolink_password": bool(
                        os.environ.get("REOLINK_PASSWORD", "").strip()
                    ),
                    "reolink_op_timeout_s": _pf.reolink_op_timeout_s(),
                    "project_root": str(_PROJECT_ROOT),
                    "cwd": os.getcwd(),
                }
            )
            return

        if parsed.path == "/state":
            self._respond_json(_state_for_api())
            return

        if parsed.path == "/api/mission/status":
            qs = parse_qs(parsed.query)
            jid = (qs.get("id") or [None])[0]
            if not jid:
                self._respond_json({"error": "missing id"})
                return
            with _MISSION_LOCK:
                job = _MISSION_JOBS.get(jid)
                if not job:
                    self.send_response(404)
                    self.end_headers()
                    return
                self._respond_json(_mission_public(job))
            return

        if parsed.path == "/":
            html = _HTML_TEMPLATE
            is_rl = self.backend == "reolink"
            sub = (
                "Reolink hardware PTZ (live preview)"
                if is_rl
                else "Simulated camera — Grand Tetons panorama"
            )
            html = html.replace("{{HEADER_SUBTITLE}}", sub)
            html = html.replace("{{TILT_RANGE}}", f"{self.tilt_range:.1f}")
            html = html.replace("{{PAN_RANGE}}", f"{self.pan_limit:.1f}")
            html = html.replace("{{PAN_SLIDER_MAX}}", f"{self.pan_limit:.1f}")
            html = html.replace("{{IS_REOLINK}}", "true" if is_rl else "false")
            html = html.replace("{{PAN_INIT}}", f"{self.init_pan}")
            html = html.replace("{{TILT_INIT}}", f"{self.init_tilt}")
            html = html.replace("{{FOV_INIT}}", f"{self.init_fov:.1f}")
            self._respond(200, "text/html", html.encode())
            return

        if parsed.path == "/live":
            try:
                if getattr(ViewerHandler, "backend", "sim") == "reolink":
                    from tools.ptz_facade import get_reolink_preview_jpeg

                    deadline = time.monotonic() + 4.0
                    data = get_reolink_preview_jpeg()
                    while not data and time.monotonic() < deadline:
                        time.sleep(0.05)
                        data = get_reolink_preview_jpeg()
                    if not data:
                        self.send_response(503)
                        self.end_headers()
                        return
                    self._respond(200, "image/jpeg", data)
                    return
                logger.info("GET /live (snapshot, sim)")
                from tools.ptz_facade import get_ptz_camera

                cam = get_ptz_camera()
                pil = cam._crop_viewport()
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=88)
                data = buf.getvalue()
                logger.info("GET /live ok bytes=%s", len(data))
                self._respond(200, "image/jpeg", data)
            except Exception as exc:
                logger.exception("Live frame error")
                self.send_response(500)
                self.end_headers()
            return

        if parsed.path == "/panorama":
            img_path = Path(self.image_path)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(img_path.stat().st_size))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            with open(img_path, "rb") as f:
                self.wfile.write(f.read())
            return

        if parsed.path == "/models":
            from tools.detectors import available_models
            self._respond_json(available_models())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = self._read_body()

        if self.path == "/api/mission/start":
            try:
                data = json.loads(body) if body else {}
            except Exception:
                self._respond_json({"error": "Invalid JSON"})
                return
            out = _spawn_mission(data)
            self._respond_json(out)
            return

        if self.path == "/api/mission/cancel":
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
            jid = data.get("id")
            if jid:
                with _MISSION_LOCK:
                    j = _MISSION_JOBS.get(jid)
                    if j and j.get("cancel_event"):
                        j["cancel_event"].set()
            self._respond_json({"ok": True})
            return

        if self.path == "/state":
            try:
                data = json.loads(body) if body else {}
            except Exception:
                self._respond_json({"ok": False, "error": "Invalid JSON body"})
                return
            try:
                fh = data.get("fov_h")
                if getattr(ViewerHandler, "backend", "sim") == "reolink":
                    from tools.ptz_facade import get_ptz_camera

                    if "jog_pan" in data or "jog_tilt" in data:
                        jp = float(data["jog_pan"]) if "jog_pan" in data else 0.0
                        jt = float(data["jog_tilt"]) if "jog_tilt" in data else 0.0
                        logger.info("POST /state reolink jog pan=%s tilt=%s", jp, jt)
                        cam = get_ptz_camera()
                        cam.jog(jp, jt)
                        if fh is not None:
                            pos = cam.set_fov_h(float(fh))
                        else:
                            pos = cam.get_position()
                        try:
                            from tools.sim_ptz_watch import merge_watch_from_payload

                            merge_watch_from_payload(data)
                        except Exception:
                            pass
                        self._respond_json(
                            {
                                "ok": True,
                                "pan": pos["pan_deg"],
                                "tilt": pos["tilt_deg"],
                                "fov_h": pos["fov_h"],
                            }
                        )
                        return

                    logger.info(
                        "POST /state reolink pan=%s tilt=%s fov_h=%s",
                        data.get("pan"),
                        data.get("tilt"),
                        fh,
                    )
                    cam = get_ptz_camera()
                    cam.move_to(
                        float(data.get("pan", 0.0)),
                        float(data.get("tilt", 0.0)),
                    )
                    if fh is not None:
                        cam.set_fov_h(float(fh))
                    logger.info("POST /state reolink done")
                else:
                    _save_state(
                        data.get("pan", 180.0),
                        data.get("tilt", 76.3),
                        float(fh) if fh is not None else None,
                    )
                try:
                    from tools.sim_ptz_watch import merge_watch_from_payload

                    merge_watch_from_payload(data)
                except Exception:
                    pass
            except Exception as exc:
                logger.exception("POST /state failed")
                self._respond_json({"ok": False, "error": str(exc)})
                return
            self._respond_json({"ok": True})
            return

        if self.path == "/detect":
            self._handle_detect(body)
            return

        if self.path == "/caption":
            self._handle_caption(body)
            return

        self.send_response(404)
        self.end_headers()

    # -- helpers --

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _respond(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(data)

    def _respond_json(self, obj):
        self._respond(200, "application/json", json.dumps(obj).encode())

    def _get_viewport(self, data: dict):
        """Viewport image for detection: sim crop or Reolink snapshot after move."""
        if getattr(ViewerHandler, "backend", "sim") == "reolink":
            from tools.ptz_facade import get_ptz_camera

            cam = get_ptz_camera()
            p = float(data.get("pan", self.init_pan))
            t = float(data.get("tilt", self.init_tilt))
            fh = data.get("fov_h")
            cam.move_to(p, t)
            if fh is not None:
                cam.set_fov_h(float(fh))
            return cam._crop_viewport()
        p = float(data.get("pan", self.init_pan))
        t = float(data.get("tilt", self.init_tilt))
        fh = float(data.get("fov_h", 60))
        fv = float(data.get("fov_v", fh * 9 / 16))
        return _extract_viewport(self.pano_pil, p, t, fh, fv, self.tilt_range, self.ppd)

    def _handle_detect(self, raw_body):
        try:
            data = json.loads(raw_body)
        except Exception:
            self._respond_json({"error": "Invalid JSON"})
            return

        viewport = self._get_viewport(data)
        model = data.get("model", "yolo")

        try:
            from tools.detectors import detect as run_detect
            from tools.sim_ptz_watch import sleep_after_inference

            if data.get("bioclip_debug"):
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s",
                    force=True,
                )

            kwargs = {}
            if model == "yolo":
                kwargs["targets"] = data.get("targets", "*")
            elif model == "bioclip":
                kwargs["target_taxon"] = data.get("target_taxon", "")
                kwargs["rank"] = data.get("rank", "Class")
                kwargs["min_confidence"] = float(data.get("min_confidence", 0.1))
                if data.get("bioclip_debug"):
                    kwargs["bioclip_debug"] = True
            elif model == "gemma4":
                kwargs["target"] = data.get("target", "")
                mst = data.get("max_soft_tokens")
                if mst is not None:
                    kwargs["max_soft_tokens"] = int(mst)
            result = run_detect(viewport, model=model, **kwargs)
            sleep_after_inference()
            self._respond_json(result)
        except Exception as exc:
            logger.exception("Detection error")
            self._respond_json({"error": str(exc), "detections": [],
                                "image_size": list(viewport.size)})

    def _handle_caption(self, raw_body):
        try:
            data = json.loads(raw_body)
        except Exception:
            self._respond_json({"error": "Invalid JSON"})
            return

        viewport = self._get_viewport(data)

        try:
            from tools.detectors import caption as run_caption
            from tools.sim_ptz_watch import sleep_after_inference

            mdl = data.get("model", "bioclip")
            ckw = {}
            if mdl == "gemma4":
                if data.get("prompt"):
                    ckw["prompt"] = data["prompt"]
                if data.get("max_soft_tokens") is not None:
                    ckw["max_soft_tokens"] = int(data["max_soft_tokens"])
            result = run_caption(viewport, model=mdl, **ckw)
            sleep_after_inference()
            self._respond_json(result)
        except Exception as exc:
            logger.exception("Caption error")
            self._respond_json({"error": str(exc), "caption": ""})

    def log_message(self, fmt, *args):
        try:
            logger.info("%s " + fmt, self.address_string(), *args)
        except Exception:
            logger.info("%s log %s %s", self.address_string(), fmt, args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive PTZ Camera Viewer")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--image", default=str(_DEFAULT_PANORAMA),
                        help="Path to panorama image (simulated mode only)")
    parser.add_argument(
        "--reolink",
        action="store_true",
        help="Use Reolink hardware (REOLINK_IP, REOLINK_USER, REOLINK_PASSWORD; tools/calibration.json)",
    )
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser automatically")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [ptz-viewer] %(message)s",
    )

    state = _load_state()

    if args.reolink:
        os.environ["MSA_PTZ_BACKEND"] = "reolink"

        # Kick off background model warm-up (YOLO / BioCLIP / Gemma 4) and run
        # the synchronous PTZ self-test (creates calibration.json if missing or
        # mismatched) BEFORE the long-lived worker grabs the camera session.
        from tools.startup_checks import verify_ptz_health, warm_detection_models

        warm_detection_models()
        verify_ptz_health()

        cal_path = _PROJECT_ROOT / "tools" / "calibration.json"
        if not cal_path.exists():
            print(f"ERROR: Calibration not found after self-test: {cal_path}")
            sys.exit(1)
        cal = json.loads(cal_path.read_text())
        tilt_range = float(cal["tilt_degrees"])
        pan_limit = float(cal["pan_degrees"])
        ViewerHandler.backend = "reolink"
        ViewerHandler.pan_limit = pan_limit
        ViewerHandler.tilt_range = tilt_range
        ViewerHandler.ppd = 1.0
        ViewerHandler.pano_pil = None
        ViewerHandler.image_path = str(_DEFAULT_PANORAMA)
        state["ptz_backend"] = "reolink"
        ViewerHandler.init_pan = max(
            0.0, min(pan_limit, float(state.get("pan", pan_limit / 2)))
        )
        ViewerHandler.init_tilt = max(
            0.0, min(tilt_range, float(state.get("tilt", tilt_range / 2)))
        )
        ViewerHandler.init_fov = float(state.get("fov_h", 60.0))
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state))

        from tools.ptz_facade import warm_reolink_worker

        warm_reolink_worker()
    else:
        if "MSA_PTZ_BACKEND" not in os.environ:
            state["ptz_backend"] = "sim"
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(json.dumps(state))
        image_path = Path(args.image).resolve()
        if not image_path.exists():
            print(f"ERROR: Image not found: {image_path}")
            sys.exit(1)

        try:
            from PIL import Image
            pano = Image.open(image_path)
            w, h = pano.size
            ppd = w / 360.0
            tilt_range = round(h / ppd, 1)
        except ImportError:
            print("ERROR: Pillow required — pip install Pillow")
            sys.exit(1)

        ViewerHandler.backend = "sim"
        ViewerHandler.pan_limit = 360.0
        ViewerHandler.image_path = str(image_path)
        ViewerHandler.tilt_range = tilt_range
        ViewerHandler.ppd = ppd
        ViewerHandler.init_pan = float(state.get("pan", 180.0))
        ViewerHandler.init_tilt = float(state.get("tilt", tilt_range / 2))
        ViewerHandler.init_fov = float(state.get("fov_h", 60.0))
        ViewerHandler.pano_pil = pano

        from tools.startup_checks import warm_detection_models

        warm_detection_models()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"PTZ Viewer running at {url}")
    if args.reolink:
        print(f"Backend: Reolink (pan 0–{ViewerHandler.pan_limit:.0f}\u00b0, tilt 0–{ViewerHandler.tilt_range:.0f}\u00b0)")
        print(f"Diagnostics (env check, no PTZ): {url}/api/ptz/ping")
    else:
        print(
            f"Backend: simulated — {ViewerHandler.image_path} "
            f"({ViewerHandler.tilt_range:.1f}\u00b0 vertical range)"
        )

    try:
        from tools.detectors import available_models
        models = available_models()
        status = " | ".join(f"{k}: {'OK' if v else 'missing'}" for k, v in models.items())
        print(f"Detection models: {status}")
    except Exception:
        print("Detection models: not loaded (run from project root)")

    if os.environ.get("BIOCLIP_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
        print(
            "BIOCLIP_DEBUG: BioCLIP INFO logs to stderr; use the “BioCLIP debug” checkbox for JSON trace."
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            force=True,
        )

    print("Press Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
