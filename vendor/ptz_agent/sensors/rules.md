# Sensor Subsystem Rules

## Purpose
Sensors are your eyes, ears, and nerve endings. They provide ground-truth data
about the physical world and your host system. Every cycle should be informed
by sensor data — do not act blind.

## Reading Discipline
- Read sensors **at wake**, before making decisions. The agent loop does this
  automatically; the latest readings appear in `sensor_readings` in the prompt.
- If a task depends on a sensor value (e.g., "check temperature"), use
  `read_sensor` to get a fresh reading rather than relying on stale data in notes.
- Do NOT poll sensors in a tight loop. One read per sensor per cycle is the norm.
  If you need higher-frequency data, update the cron schedule instead.

## Interpreting Readings
- Treat every reading as a dict of named values. Check for `_error` keys first —
  a sensor that returns an error is degraded, not dead.
- When values are outside expected ranges, log the anomaly in `notes` with the
  sensor name, the value, and what you expected. Do NOT silently ignore it.
- Units are declared in the sensor's schema (`units` field). Never assume units.

## Anomaly Response
1. **Log it**: record the anomalous reading, timestamp, and sensor name in notes.
2. **Verify it**: if a second read is cheap, re-read on the next iteration to
   confirm the anomaly is persistent rather than a transient glitch.
3. **Act if possible**: e.g., if disk is >90% full, clean temp files. If a
   sensor is unreachable, note it for the operator.
4. **Do not panic**: a single bad reading is informational. Three consecutive
   anomalous readings on the same sensor is an incident worth escalating in notes.

## Sensor Lifecycle
- Sensors are auto-discovered from the `sensors/` directory on startup.
- New sensors can be added at any time by dropping a Python file that subclasses
  `BaseSensor`. The agent will pick it up on the next restart.
- Sensors have a `status()` method — use `sensor_status` to check health before
  blaming a missing reading on a bug.

## Interface Awareness
Sensors declare their interface type (system, serial, usb, i2c, spi, gpio,
network). This is metadata only — you don't need to manage the interface
directly. But if a sensor is on `serial` or `usb` and suddenly returns errors,
note the interface type since it helps the operator diagnose physical connection
issues.

## Privacy & Safety
- Do not log raw image or audio sensor data in the scratchpad. Log summaries,
  counts, or classifications instead.
- Never transmit sensor data to external URLs unless explicitly instructed by a
  task. The `http_get` tool is for fetching, not posting.
