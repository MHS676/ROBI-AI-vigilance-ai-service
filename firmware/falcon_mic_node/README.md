# Falcon AI Microphone Node — ESP32-S3 Firmware

## Hardware

| Component     | Part                        |
|---------------|-----------------------------|
| Microcontroller | ESP32-S3 DevKitC-1        |
| Microphone    | INMP441 I2S MEMS            |
| Power         | 5V USB or LiPo + TP4056    |

### Wiring

```
INMP441  →  ESP32-S3
───────────────────
VDD      →  3V3
GND      →  GND
SD       →  GPIO 8   (I2S Data In)
WS       →  GPIO 45  (Word Select / LR clock)
SCK      →  GPIO 46  (Bit Clock)
L/R      →  GND      (select LEFT channel)
```

---

## Build & Flash

### Option A — PlatformIO (recommended)

```bash
cd ai-service/firmware/falcon_mic_node
pio run --target upload
pio device monitor
```

### Option B — Arduino IDE 2.x

1. **Board Manager** → search `esp32` → install **Espressif Systems esp32** ≥ 3.0.0
2. **Library Manager** → install:
   - `WiFiManager` by tzapu ≥ 2.0.17
   - `PubSubClient` by Nick O'Leary ≥ 2.8.0
   - `ArduinoJson` by Benoit Blanchon ≥ 7.0.0
3. Open `falcon_mic_node.ino`, select **ESP32S3 Dev Module**, upload.

---

## First Boot — WiFi Provisioning

On first boot (no WiFi credentials stored), the device launches an Access Point:

```
SSID     : FalconMic-Setup-XXYY   (last 2 bytes of MAC)
Password : falcon1234
Portal   : http://192.168.4.1
```

Connect, enter the branch WiFi credentials, save.  
The device reboots and connects to the network.

---

## Device Provisioning Handshake

```
ESP32                            Falcon Backend (NestJS)
──────                           ────────────────────────
  │── [SUB] falcon/provision/AA_BB_CC_DD_EE_FF ──────────▶│
  │                                                        │
  │── [PUB] falcon/discovery/pending ───────────────────▶ │
  │   { macAddress, firmwareVer, deviceType,              │
  │     ipAddress, hostname, timestamp }                  │
  │                                                        │
  │                              Super Admin approves      │
  │                              in dashboard              │
  │                                                        │
  │◀── [PUB] falcon/provision/AA_BB_CC_DD_EE_FF ─────────│
  │   { centerId, centerCode, tableId,                    │
  │     wifiSsid, wifiPassword, serverUrl, mqttUrl }      │
  │                                                        │
  │── [PUB] falcon/provision/ack/AA_BB_CC_DD_EE_FF ─────▶ │
  │   { macAddress, status: "OK" }                        │
  │                                                        │
  │ Config saved to NVS — audio task starts               │
  │                                                        │
  │── [PUB] falcon/center/{centerId}/audio-level ────────▶│  every 250ms
  │── [PUB] falcon/center/{centerId}/device-status ──────▶│  every 10s
```

---

## Runtime MQTT Commands

Publish to `falcon/cmd/AA_BB_CC_DD_EE_FF` (replace with actual MAC):

| Command JSON                             | Effect                     |
|------------------------------------------|----------------------------|
| `{"cmd":"MUTE"}`                         | Stop publishing audio      |
| `{"cmd":"UNMUTE"}`                       | Resume audio               |
| `{"cmd":"SET_SAMPLE_RATE","value":8000}` | Change I2S sample rate     |
| `{"cmd":"REBOOT"}`                       | Graceful restart           |
| `{"cmd":"GET_STATUS"}`                   | Force immediate heartbeat  |

Valid sample rates: **8000, 16000, 22050, 44100** Hz

---

## MQTT Topic Reference

| Direction | Topic                                          | Description              |
|-----------|------------------------------------------------|--------------------------|
| PUB       | `falcon/discovery/pending`                     | Birth / discovery message|
| SUB       | `falcon/provision/{MAC_underscores}`           | Receive config           |
| PUB       | `falcon/provision/ack/{MAC_underscores}`       | Config accepted ACK      |
| PUB       | `falcon/center/{centerId}/audio-level`         | RMS dB level (250ms)     |
| PUB       | `falcon/center/{centerId}/device-status`       | Heartbeat (10s)          |
| SUB       | `falcon/cmd/{MAC_underscores}`                 | Runtime commands         |

MAC format in topics: colons replaced with underscores (`AA_BB_CC_DD_EE_FF`)

---

## Audio Level Payload

```json
{
  "centerId":         "clxyz123",
  "microphoneId":     "AA:BB:CC:DD:EE:FF",
  "tableId":          "cltable456",
  "dbLevel":          72.5,
  "threshold":        70.0,
  "event":            "HIGH_AUDIO_LEVEL",
  "timestamp":        1712800000
}
```

Events: `SILENT` | `NORMAL` | `HIGH_AUDIO_LEVEL` | `SCREAM`

## Heartbeat Payload

```json
{
  "centerId":    "clxyz123",
  "deviceId":    "AA:BB:CC:DD:EE:FF",
  "macAddress":  "AA:BB:CC:DD:EE:FF",
  "deviceType":  "MICROPHONE",
  "status":      "ONLINE",
  "ipAddress":   "192.168.1.201",
  "uptimeMs":    3600000,
  "sampleRate":  16000,
  "muted":       false,
  "timestamp":   1712800000
}
```
