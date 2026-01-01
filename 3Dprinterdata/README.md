
# OctoPrint MQTT Temperature Logger

This folder contains a minimal MQTT → CSV logger for OctoPrint.

By default it writes a **clean temperature dataset** (not raw topic/payload).

## Output formats

### Clean temperature CSV (default)

File (default): `ender3_temps_clean.csv`

Columns:
- `timestamp` (ISO-like: `YYYY-MM-DDTHH:MM:SS`)
- `bed_temp_c`
- `nozzle_temp_c`

This matches the “clean dataset” shape used by the project ingestion/training pipeline.

### Raw MQTT CSV (optional)

File (default): `ender3_continuous_log.csv`

Columns:
- `Timestamp`
- `Topic`
- `Payload`

## Requirements

- Python 3.x
- `paho-mqtt`

Install:

```bash
pip install -r requirements.txt
```

## Configuration (environment variables)

- `PM_MQTT_BROKER` (default: `192.168.55.94`)
- `PM_MQTT_PORT` (default: `1883`)
- `PM_MQTT_TOPIC` (default: `octoPrint/#`)

- `PM_LOG_MODE` (default: `clean`) values: `clean`, `raw`, `both`
- `PM_CLEAN_LOG_FILE` (default: `ender3_temps_clean.csv`)
- `PM_LOG_FILE` (default: `ender3_continuous_log.csv`) (raw mode)
- `PM_CLEAN_WRITE_MIN_SECONDS` (default: `1`) minimum seconds between clean rows

- `PM_LOG_ALL` (default: `0`) if `1`, raw mode logs all topics (can get huge)

## Run

```bash
python datalogger.py
```

## Do I need to run it continuously?

Yes — if you want a continuous time-series log, this script should run the whole time the printer is operating.

If you want it to survive reboots/crashes, run it as a service:
- Raspberry Pi/Linux: `systemd` service
- Windows: Task Scheduler (run at startup) or a persistent terminal session

## Configuration (environment variables)

- `PM_MQTT_BROKER` (default: `192.168.55.94`)
- `PM_MQTT_PORT` (default: `1883`)
- `PM_MQTT_TOPIC` (default: `octoPrint/#`)
- `PM_LOG_FILE` (default: `ender3_continuous_log.csv`)
- `PM_LOG_ALL` set to `1` to log all topics (warning: huge file)

Example:

```bash
set PM_MQTT_BROKER=192.168.1.50
set PM_LOG_FILE=printer01_log.csv
python datalogger.py
```

The system provides actionable recommendations:
- Thermal anomalies → Check thermistor/heater
- High PID error → Tune PID or check thermal paste
- Motion anomalies → Check belts, lubrication
- Low health score → Comprehensive maintenance check

## ⚙️ Customization

Edit parameters in each module:
- Sampling interval (default: 1min)
- Window sizes for rolling statistics
- Anomaly detection thresholds
- Model hyperparameters
- Feature selection

## 📊 Example Results

```
ANALYSIS SUMMARY
======================================================================

📊 Data:
  Total samples: 756
  Features: 45
  Time range: 2025-12-23 15:21:59 to 2025-12-23 16:13:30

❤️  Health:
  Current score: 0.852
  Average score: 0.847
  Status: ✅ GOOD

🔍 Anomalies:
  Total detected: 45 (5.9%)
  Critical: 3 (0.4%)

⚡ Failure Risk:
  Current: 0.23
  Average: 0.28
  Level: ✅ LOW
```

## 🤝 Contributing

Feel free to extend this system with:
- Additional sensors
- More sophisticated models
- Real-time monitoring
- Alert systems
- Integration with other tools

## 📄 License

MIT License - Use freely for personal or commercial projects

## 🙏 Acknowledgments

Built for the 3D printing community to help maintain printers proactively using machine learning.
