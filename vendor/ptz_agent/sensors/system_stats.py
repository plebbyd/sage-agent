"""
sensors/system_stats.py — System statistics sensor plugin.

Reads CPU load, memory usage, disk usage, and uptime using only the
Python standard library.  Works fully on Linux; degrades gracefully
on macOS/other platforms (load + disk always available, memory and
uptime fall back to 'unavailable').
"""

import os
import sys

from msa.sensors import BaseSensor


class SystemStatsSensor(BaseSensor):
    name = "system_stats"
    description = "CPU load averages, memory, disk usage, and uptime."
    interface = "system"
    units = "mixed"

    def read(self) -> dict:
        reading = {}

        try:
            load1, load5, load15 = os.getloadavg()
            reading["cpu_load_1m"] = round(load1, 2)
            reading["cpu_load_5m"] = round(load5, 2)
            reading["cpu_load_15m"] = round(load15, 2)
        except OSError:
            reading["cpu_load"] = "unavailable"

        try:
            stat = os.statvfs("/")
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used = total - free
            reading["disk_total_gb"] = round(total / (1024 ** 3), 1)
            reading["disk_used_gb"] = round(used / (1024 ** 3), 1)
            reading["disk_free_gb"] = round(free / (1024 ** 3), 1)
            reading["disk_used_pct"] = round(100 * used / total, 1) if total else 0
        except OSError:
            reading["disk"] = "unavailable"

        try:
            with open("/proc/uptime") as f:
                uptime_sec = float(f.read().split()[0])
                reading["uptime_hours"] = round(uptime_sec / 3600, 1)
        except (FileNotFoundError, PermissionError):
            reading["uptime"] = "unavailable (non-Linux)"

        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(":")] = int(parts[1])
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", 0)
            used_kb = total_kb - avail_kb
            reading["mem_total_mb"] = round(total_kb / 1024)
            reading["mem_used_mb"] = round(used_kb / 1024)
            reading["mem_available_mb"] = round(avail_kb / 1024)
            reading["mem_used_pct"] = round(100 * used_kb / total_kb, 1) if total_kb else 0
        except (FileNotFoundError, PermissionError):
            reading["memory"] = "unavailable (non-Linux)"

        return reading

    def status(self) -> dict:
        base = super().status()
        base["platform"] = sys.platform
        return base
