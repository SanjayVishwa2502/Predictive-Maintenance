import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import paho.mqtt.client as mqtt

# --- CONFIGURATION ---
# Override via environment variables for different printers / brokers.
MQTT_BROKER = os.environ.get("PM_MQTT_BROKER", "192.168.201.95")
MQTT_PORT = int(os.environ.get("PM_MQTT_PORT", "1883"))
TOPIC_BASE = os.environ.get("PM_MQTT_TOPIC", "octoPrint/#")

# Optional: stable id for this printer (used for default output file naming)
MACHINE_ID = str(os.environ.get("PM_MACHINE_ID", "")).strip()

# Optional: directory for output CSVs (defaults to current working directory)
LOG_DIR = str(os.environ.get("PM_LOG_DIR", "")).strip()

# Logging modes:
# - clean: only write timestamp,bed_temp_c,nozzle_temp_c (recommended)
# - raw: only write raw MQTT topic/payload rows
# - both: write both files
LOG_MODE = str(os.environ.get("PM_LOG_MODE", "clean")).strip().lower()

_raw_default = f"{MACHINE_ID}_continuous_log.csv" if MACHINE_ID else "ender3_continuous_log.csv"
_clean_default = f"{MACHINE_ID}_temps_clean.csv" if MACHINE_ID else "ender3_temps_clean.csv"

RAW_CSV_FILE = os.environ.get("PM_LOG_FILE", _raw_default)
CLEAN_CSV_FILE = os.environ.get("PM_CLEAN_LOG_FILE", _clean_default)

# Minimum seconds between clean rows (prevents huge duplicate bursts)
MIN_CLEAN_WRITE_SECONDS = float(os.environ.get("PM_CLEAN_WRITE_MIN_SECONDS", "1"))


def _resolve_log_path(filename: str) -> str:
    if not LOG_DIR:
        return filename
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, filename)

def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes"}


def _iso_timestamp_now() -> str:
    # Match the existing cleaned dataset format in data_ingestion/processed
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class CsvWriters:
    raw_file: Optional[object] = None
    raw_writer: Optional[csv.writer] = None
    clean_file: Optional[object] = None
    clean_writer: Optional[csv.writer] = None


def _open_writers() -> CsvWriters:
    writers = CsvWriters()

    if LOG_MODE in {"raw", "both"}:
        raw_path = _resolve_log_path(RAW_CSV_FILE)
        raw_exists = os.path.isfile(raw_path)
        writers.raw_file = open(raw_path, "a", newline="", encoding="utf-8")
        writers.raw_writer = csv.writer(writers.raw_file)
        if not raw_exists:
            writers.raw_writer.writerow(["Timestamp", "Topic", "Payload"])
            writers.raw_file.flush()

    if LOG_MODE in {"clean", "both"}:
        clean_path = _resolve_log_path(CLEAN_CSV_FILE)
        clean_exists = os.path.isfile(clean_path)
        writers.clean_file = open(clean_path, "a", newline="", encoding="utf-8")
        writers.clean_writer = csv.writer(writers.clean_file)
        if not clean_exists:
            writers.clean_writer.writerow(["timestamp", "bed_temp_c", "nozzle_temp_c"])
            writers.clean_file.flush()

    return writers


WRITERS = _open_writers()


def _close_writers() -> None:
    for f in [WRITERS.raw_file, WRITERS.clean_file]:
        try:
            if f:
                f.close()
        except Exception:
            pass


def _write_raw(timestamp: str, topic: str, payload: str) -> None:
    if not WRITERS.raw_writer or not WRITERS.raw_file:
        return
    WRITERS.raw_writer.writerow([timestamp, topic, payload])
    WRITERS.raw_file.flush()


_latest_bed: Optional[float] = None
_latest_nozzle: Optional[float] = None
_last_clean_write_at: float = 0.0


def _maybe_write_clean(now_iso: str) -> None:
    global _last_clean_write_at
    if not WRITERS.clean_writer or not WRITERS.clean_file:
        return
    if _latest_bed is None or _latest_nozzle is None:
        return

    now = time.monotonic()
    if now - _last_clean_write_at < MIN_CLEAN_WRITE_SECONDS:
        return
    _last_clean_write_at = now

    WRITERS.clean_writer.writerow([
        now_iso,
        f"{_latest_bed:.3f}",
        f"{_latest_nozzle:.3f}",
    ])
    WRITERS.clean_file.flush()

# --- MQTT CALLBACKS ---

def on_connect(client, userdata, flags, rc, properties):
    print(f"Connected to MQTT Broker with result code {rc}")
    # Subscribe to everything coming from OctoPrint
    client.subscribe(TOPIC_BASE)

def on_message(client, userdata, msg):
    try:
        # Get current time
        timestamp_raw = time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_iso = _iso_timestamp_now()
        topic = msg.topic
        payload = msg.payload.decode("utf-8", errors="replace")

        # FILTERING for RAW log. (Clean log is temperature-only by design.)
        log_all = _truthy_env("PM_LOG_ALL", "0")
        should_log_raw = log_all

        # Temperature updates -> update clean state
        if "temperature" in topic:
            try:
                data = json.loads(payload)
                actual = data.get("actual")
                if isinstance(actual, (int, float)):
                    global _latest_bed, _latest_nozzle
                    if topic.endswith("/bed"):
                        _latest_bed = float(actual)
                    elif topic.endswith("/tool0"):
                        _latest_nozzle = float(actual)
                    _maybe_write_clean(timestamp_iso)
            except Exception:
                # Ignore malformed payloads
                pass

            if not should_log_raw:
                should_log_raw = True

        # Position updates (optional raw)
        elif "event/PositionUpdate" in topic:
            if not should_log_raw:
                should_log_raw = True

        # Connected/disconnected status (optional raw)
        elif "mqtt" in topic:
            if not should_log_raw:
                should_log_raw = True
            print(f"!!! SYSTEM STATUS CHANGE: {payload} !!!")

        if should_log_raw:
            print(f"[{timestamp_raw}] {topic}: {payload}")
            _write_raw(timestamp_raw, topic, payload)

    except Exception as e:
        print(f"Error processing message: {e}")

# --- MAIN LOOP ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

print(f"Connecting to Broker... {MQTT_BROKER}:{MQTT_PORT} (topic: {TOPIC_BASE})")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
    print(f"Make sure MQTT broker at {MQTT_BROKER}:{MQTT_PORT} is accessible")
    _close_writers()
    exit(1)

# Run forever
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopping Logger...")
    _close_writers()