// =============================================================================
//  FALCON SECURITY LIMITED — ESP32-S3 AI Microphone Node
//  Hardware : ESP32-S3 (any variant) + INMP441 I2S MEMS Microphone
//  Firmware : v2.2.0
//  Build    : Arduino IDE 2.x / PlatformIO
//
//  ┌─────────────────────────────────────────────────────────────────────────┐
//  │  Required Libraries (install via Arduino Library Manager)               │
//  │  ─────────────────────────────────────────────────────────────────────  │
//  │  • WiFiManager    by tzapu          (>= 2.0.17)                         │
//  │  • PubSubClient   by Nick O'Leary   (>= 2.8.0)                          │
//  │  • ArduinoJson    by Benoit Blanchon (>= 7.0.0)                         │
//  │  • Preferences    (built-in ESP-IDF / arduino-esp32)                    │
//  │  • driver/i2s.h   (built-in ESP-IDF)                                    │
//  └─────────────────────────────────────────────────────────────────────────┘
//
//  ╔═══════════════════════════════════════════════════════════════════════╗
//  ║  INMP441 → ESP32-S3 Wiring                                           ║
//  ║  ─────────────────────────────────────────────────────────────────── ║
//  ║  INMP441 Pin │ ESP32-S3 GPIO                                         ║
//  ║  ────────────┼──────────────                                         ║
//  ║  VDD         │ 3V3                                                   ║
//  ║  GND         │ GND                                                   ║
//  ║  SD (data)   │ GPIO 8   (I2S_DATA_IN)                                ║
//  ║  WS (L/R)    │ GPIO 45  (I2S_WS)                                     ║
//  ║  SCK (bclk)  │ GPIO 46  (I2S_BCLK)                                   ║
//  ║  L/R select  │ GND      (LEFT channel — matches MicrophoneChannel)   ║
//  ╚═══════════════════════════════════════════════════════════════════════╝
//
//  ═══════════════════  DEVICE LIFECYCLE  ════════════════════════════════════
//
//  [POWER ON]
//      │
//      ▼
//  [WiFiManager] ── credentials in NVS? ──YES──► Connect to branch WiFi
//      │                                              │
//      NO                                             ▼
//      ▼                                    [MQTT Connect to broker]
//  [AP Mode] "FalconMic-XXXX"                         │
//  Admin opens 192.168.4.1,                           ▼
//  enters SSID + password                  [STATE: UNPROVISIONED]
//      │                                   Publish birth to:
//      ▼                                   falcon/discovery/pending
//  [Saved → Reboot]                        Subscribe to:
//                                          falcon/provision/AA_BB_CC_DD_EE_01
//                                               │
//                                          Super Admin approves in dashboard
//                                               │
//                                          Backend publishes config to:
//                                          falcon/provision/AA_BB_CC_DD_EE_01
//                                          { centerId, tableId, wifiSsid,
//                                            wifiPassword, serverUrl, mqttUrl }
//                                               │
//                                               ▼
//                                    [SAVE CONFIG TO NVS]
//                                    Publish ACK to:
//                                    falcon/provision/ack/AA_BB_CC_DD_EE_01
//                                               │
//                                               ▼
//                                    [STATE: PROVISIONED]
//                                    Start I2S audio sampling task
//                                    Publish audio to:
//                                    falcon/center/{centerId}/audio-level
//                                    Publish heartbeat every 10s to:
//                                    falcon/center/{centerId}/device-status
//
// =============================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiManager.h>         // tzapu/WiFiManager
#include <PubSubClient.h>        // Nick O'Leary/PubSubClient
#include <ArduinoJson.h>         // Benoit Blanchon/ArduinoJson
#include <Preferences.h>         // NVS (Non-Volatile Storage)
#include <driver/i2s.h>          // ESP-IDF I2S driver

// =============================================================================
//  COMPILE-TIME CONFIGURATION
// =============================================================================

// ── Firmware identity ─────────────────────────────────────────────────────────
#define FIRMWARE_VERSION        "v2.2.0"
#define DEVICE_TYPE             "AI_MICROPHONE"
#define DEVICE_MODEL            "Falcon-MicNode-S3"

// ── WiFiManager AP (shown when no WiFi credentials are stored) ────────────────
#define WIFIMANAGER_AP_NAME     "FalconMic-Setup"   // Suffix: last 4 hex of MAC
#define WIFIMANAGER_AP_PASS     "falcon1234"         // AP password (min 8 chars)
#define WIFI_CONNECT_TIMEOUT_S  180                  // Wait 3 min in AP mode

// ── Default MQTT broker (overridden after provisioning) ───────────────────────
#define DEFAULT_MQTT_HOST       "broker.local"       // mDNS or IP
#define DEFAULT_MQTT_PORT       1883
#define MQTT_RECONNECT_DELAY_MS 5000
#define MQTT_KEEPALIVE_S        60
#define MQTT_QOS1               1

// ── I2S / INMP441 GPIO Pins ───────────────────────────────────────────────────
#define I2S_PORT                I2S_NUM_0
#define I2S_BCLK_PIN            46           // Bit clock
#define I2S_WS_PIN              45           // Word select (L/R)
#define I2S_DATA_PIN            8            // Serial data in
#define I2S_CHANNEL_FORMAT      I2S_CHANNEL_FMT_ONLY_LEFT  // INMP441 L/R → GND

// ── Audio sampling defaults ────────────────────────────────────────────────────
#define AUDIO_SAMPLE_RATE_HZ    16000        // Default 16 kHz
#define AUDIO_BITS_PER_SAMPLE   I2S_BITS_PER_SAMPLE_32BIT
#define I2S_DMA_BUF_COUNT       4
#define I2S_DMA_BUF_LEN         1024         // Samples per DMA buffer
#define AUDIO_FRAME_SAMPLES     512          // Samples per RMS window
#define AUDIO_PUBLISH_INTERVAL_MS 250        // Publish RMS every 250 ms
#define HIGH_AUDIO_THRESHOLD_DB   70.0f      // dB SPL threshold for alert

// ── Heartbeat ─────────────────────────────────────────────────────────────────
#define HEARTBEAT_INTERVAL_MS   10000        // 10 seconds

// ── NVS namespace key names ───────────────────────────────────────────────────
#define NVS_NAMESPACE           "falcon"
#define NVS_KEY_CENTER_ID       "centerId"
#define NVS_KEY_TABLE_ID        "tableId"
#define NVS_KEY_MQTT_HOST       "mqttHost"
#define NVS_KEY_MQTT_PORT       "mqttPort"
#define NVS_KEY_PROVISIONED     "provisioned"
#define NVS_KEY_SAMPLE_RATE     "sampleRate"
#define NVS_KEY_MUTED           "muted"

// =============================================================================
//  MQTT TOPIC BUILDERS  (mirror backend's mqtt.constants.ts exactly)
// =============================================================================

// MAC address is stored as "AA_BB_CC_DD_EE_FF" (colons → underscores)
// for use in MQTT topics (colons are illegal in some brokers / topic levels).

static char g_macDash[18];   // "AA:BB:CC:DD:EE:FF"
static char g_macTopic[18];  // "AA_BB_CC_DD_EE_FF"

// Populated after provisioning
static char g_centerId[64]  = "";
static char g_tableId[64]   = "";
static char g_mqttHost[128] = DEFAULT_MQTT_HOST;
static int  g_mqttPort      = DEFAULT_MQTT_PORT;

// Topics built at runtime after centerId is known
static char TOPIC_AUDIO_LEVEL[128];     // falcon/center/{centerId}/audio-level
static char TOPIC_DEVICE_STATUS[128];   // falcon/center/{centerId}/device-status
static char TOPIC_PROVISION_IN[128];    // falcon/provision/{macTopic}
static char TOPIC_PROVISION_ACK[128];   // falcon/provision/ack/{macTopic}
static const char* TOPIC_BIRTH     = "falcon/discovery/pending";
static const char* TOPIC_CMD_BASE  = "falcon/cmd/";   // falcon/cmd/{macTopic}
static char TOPIC_CMD[128];             // falcon/cmd/{macTopic}

inline void buildTopics() {
    snprintf(TOPIC_AUDIO_LEVEL,   sizeof(TOPIC_AUDIO_LEVEL),
             "falcon/center/%s/audio-level",   g_centerId);
    snprintf(TOPIC_DEVICE_STATUS, sizeof(TOPIC_DEVICE_STATUS),
             "falcon/center/%s/device-status", g_centerId);
    snprintf(TOPIC_CMD,           sizeof(TOPIC_CMD),
             "falcon/cmd/%s",                  g_macTopic);
}

// =============================================================================
//  GLOBAL STATE
// =============================================================================

WiFiClient    g_wifiClient;
PubSubClient  g_mqtt(g_wifiClient);
Preferences   g_prefs;

// ── Device state machine ──────────────────────────────────────────────────────
enum class DeviceState : uint8_t {
    WIFI_CONNECTING,
    UNPROVISIONED,      // Connected to MQTT, birth published, awaiting Super Admin
    PROVISIONED         // Config received, audio sampling active
};
static volatile DeviceState g_state = DeviceState::WIFI_CONNECTING;

// ── Audio / I2S ───────────────────────────────────────────────────────────────
static volatile bool  g_muted         = false;
static volatile int   g_sampleRate    = AUDIO_SAMPLE_RATE_HZ;
static volatile bool  g_i2sRunning    = false;
static volatile bool  g_restartI2S    = false;    // Set to true to restart I2S with new rate

// ── Timing ────────────────────────────────────────────────────────────────────
static unsigned long g_lastHeartbeat   = 0;
static unsigned long g_lastBirthMsg    = 0;
static unsigned long g_birthRetryMs    = 30000;   // Re-publish birth every 30s until provisioned

// ── FreeRTOS task handles ─────────────────────────────────────────────────────
static TaskHandle_t  g_i2sTaskHandle   = nullptr;
static TaskHandle_t  g_mqttTaskHandle  = nullptr;

// =============================================================================
//  NVS: PERSIST & LOAD PROVISIONING CONFIG
// =============================================================================

/**
 * Load provisioning config from NVS.
 * Returns true if device was previously provisioned.
 */
bool loadProvisionConfig() {
    g_prefs.begin(NVS_NAMESPACE, true);  // read-only

    bool provisioned = g_prefs.getBool(NVS_KEY_PROVISIONED, false);
    if (provisioned) {
        g_prefs.getString(NVS_KEY_CENTER_ID, g_centerId,    sizeof(g_centerId));
        g_prefs.getString(NVS_KEY_TABLE_ID,  g_tableId,     sizeof(g_tableId));
        g_prefs.getString(NVS_KEY_MQTT_HOST, g_mqttHost,    sizeof(g_mqttHost));
        g_mqttPort   = g_prefs.getInt(NVS_KEY_MQTT_PORT, DEFAULT_MQTT_PORT);
        g_muted      = g_prefs.getBool(NVS_KEY_MUTED, false);
        g_sampleRate = g_prefs.getInt(NVS_KEY_SAMPLE_RATE, AUDIO_SAMPLE_RATE_HZ);
    }

    g_prefs.end();
    return provisioned;
}

/**
 * Persist provisioning config received from the backend to NVS.
 */
void saveProvisionConfig(
    const char* centerId,
    const char* tableId,
    const char* mqttHost,
    int         mqttPort
) {
    g_prefs.begin(NVS_NAMESPACE, false);  // read-write
    g_prefs.putBool(NVS_KEY_PROVISIONED, true);
    g_prefs.putString(NVS_KEY_CENTER_ID, centerId);
    g_prefs.putString(NVS_KEY_TABLE_ID,  tableId);
    g_prefs.putString(NVS_KEY_MQTT_HOST, mqttHost);
    g_prefs.putInt(NVS_KEY_MQTT_PORT,    mqttPort);
    g_prefs.end();

    strncpy(g_centerId, centerId, sizeof(g_centerId) - 1);
    strncpy(g_tableId,  tableId,  sizeof(g_tableId)  - 1);
    strncpy(g_mqttHost, mqttHost, sizeof(g_mqttHost) - 1);
    g_mqttPort = mqttPort;
}

void saveAudioSettings() {
    g_prefs.begin(NVS_NAMESPACE, false);
    g_prefs.putBool(NVS_KEY_MUTED,      g_muted);
    g_prefs.putInt(NVS_KEY_SAMPLE_RATE, g_sampleRate);
    g_prefs.end();
}

// =============================================================================
//  I2S DRIVER
// =============================================================================

/**
 * Install the I2S driver with the current g_sampleRate.
 * Call i2s_driver_uninstall() before calling this again to change sample rate.
 */
void i2s_install(int sampleRate) {
    const i2s_config_t i2s_config = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = (uint32_t)sampleRate,
        .bits_per_sample      = AUDIO_BITS_PER_SAMPLE,
        .channel_format       = I2S_CHANNEL_FORMAT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = I2S_DMA_BUF_COUNT,
        .dma_buf_len          = I2S_DMA_BUF_LEN,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0,
    };

    const i2s_pin_config_t pin_config = {
        .bck_io_num   = I2S_BCLK_PIN,
        .ws_io_num    = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_DATA_PIN,
    };

    ESP_ERROR_CHECK(i2s_driver_install(I2S_PORT, &i2s_config, 0, nullptr));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_PORT, &pin_config));
    ESP_ERROR_CHECK(i2s_zero_dma_buffer(I2S_PORT));
    g_i2sRunning = true;
    Serial.printf("[I2S] Installed @ %d Hz\n", sampleRate);
}

void i2s_uninstall() {
    if (g_i2sRunning) {
        i2s_driver_uninstall(I2S_PORT);
        g_i2sRunning = false;
        Serial.println("[I2S] Uninstalled");
    }
}

/**
 * Calculate dB SPL from a buffer of 32-bit I2S samples.
 *
 * INMP441 outputs 24-bit data in MSB of a 32-bit word — shift right 8 bits.
 * RMS of the samples → converted to dB: dB = 20 * log10(RMS / full_scale)
 * Full-scale is 2^23 = 8,388,608 for the INMP441's 24-bit range.
 */
float computeRmsDb(const int32_t* samples, size_t count) {
    if (count == 0) return -90.0f;
    double sumSq = 0.0;
    for (size_t i = 0; i < count; i++) {
        // Shift 32-bit I2S word right 8 to get 24-bit signed value
        int32_t s = samples[i] >> 8;
        sumSq += (double)s * (double)s;
    }
    double rms = sqrt(sumSq / (double)count);
    if (rms < 1.0) return -90.0f;   // silence floor
    return 20.0f * log10f((float)rms / 8388608.0f) + 94.0f;  // 94 dB ref = 1 Pa
}

// =============================================================================
//  I2S SAMPLING TASK  (runs on Core 1)
// =============================================================================

/**
 * FreeRTOS task: continuously read I2S DMA buffers, compute RMS dB level,
 * and publish an AudioLevelPayload to MQTT every AUDIO_PUBLISH_INTERVAL_MS.
 *
 * Only active when g_state == PROVISIONED and !g_muted.
 * When g_restartI2S is set (sample rate change), the task re-installs I2S.
 */
void i2sTask(void* pvParameters) {
    static int32_t  samples[AUDIO_FRAME_SAMPLES];
    static char     pubBuf[256];
    static float    accumDb   = 0.0f;
    static int      accumCnt  = 0;
    unsigned long   lastPublish = 0;

    i2s_install(g_sampleRate);

    while (true) {
        // ── Handle sample-rate change request ────────────────────────────────
        if (g_restartI2S) {
            g_restartI2S = false;
            i2s_uninstall();
            vTaskDelay(pdMS_TO_TICKS(100));
            i2s_install(g_sampleRate);
        }

        // ── Skip read if muted ────────────────────────────────────────────────
        if (g_muted) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // ── Read one DMA frame ────────────────────────────────────────────────
        size_t bytesRead = 0;
        esp_err_t rc = i2s_read(
            I2S_PORT,
            samples,
            sizeof(samples),
            &bytesRead,
            portMAX_DELAY
        );

        if (rc != ESP_OK || bytesRead == 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        size_t samplesRead = bytesRead / sizeof(int32_t);
        float db = computeRmsDb(samples, samplesRead);

        accumDb  += db;
        accumCnt += 1;

        // ── Publish at publish interval ───────────────────────────────────────
        unsigned long now = millis();
        if ((now - lastPublish) >= AUDIO_PUBLISH_INTERVAL_MS && accumCnt > 0) {
            lastPublish = now;

            float avgDb = accumDb / (float)accumCnt;
            accumDb  = 0.0f;
            accumCnt = 0;

            if (g_state != DeviceState::PROVISIONED || !g_mqtt.connected()) continue;

            // Classify audio event
            const char* audioEvent;
            if      (avgDb >= HIGH_AUDIO_THRESHOLD_DB + 15.0f) audioEvent = "SCREAM";
            else if (avgDb >= HIGH_AUDIO_THRESHOLD_DB)          audioEvent = "HIGH_AUDIO_LEVEL";
            else if (avgDb >= 40.0f)                             audioEvent = "NORMAL";
            else                                                  audioEvent = "SILENT";

            // Build AudioLevelPayload JSON (matches mqtt-payload.interface.ts)
            StaticJsonDocument<256> doc;
            doc["centerId"]         = g_centerId;
            doc["microphoneId"]     = g_macDash;       // MAC used as device ID pre-DB-link
            if (strlen(g_tableId) > 0) doc["tableId"] = g_tableId;
            doc["dbLevel"]          = serialized(String(avgDb, 1));
            doc["threshold"]        = HIGH_AUDIO_THRESHOLD_DB;
            doc["event"]            = audioEvent;
            doc["timestamp"]        = (unsigned long)(millis() / 1000UL);

            serializeJson(doc, pubBuf, sizeof(pubBuf));
            g_mqtt.publish(TOPIC_AUDIO_LEVEL, pubBuf, false);
        }
    }
}

// =============================================================================
//  MQTT PAYLOAD BUILDERS
// =============================================================================

/**
 * Publish a Discovery / Birth message to falcon/discovery/pending.
 * Matches the DeviceBirthPayload interface in discovery.controller.ts exactly.
 * QoS 1, retained = true — broker holds it until backend connects.
 */
void publishBirthMessage() {
    StaticJsonDocument<256> doc;
    doc["macAddress"]  = g_macDash;
    doc["firmwareVer"] = FIRMWARE_VERSION;
    doc["deviceType"]  = DEVICE_TYPE;
    doc["model"]       = DEVICE_MODEL;
    doc["ipAddress"]   = WiFi.localIP().toString();
    doc["hostname"]    = WiFi.getHostname();
    doc["timestamp"]   = (unsigned long)(millis() / 1000UL);

    char buf[256];
    serializeJson(doc, buf, sizeof(buf));

    // QoS 1, retained: broker re-delivers to any newly subscribed backend instance
    bool ok = g_mqtt.publish(TOPIC_BIRTH, (uint8_t*)buf, strlen(buf), true);
    Serial.printf("[MQTT] Birth message %s → %s\n", ok ? "OK" : "FAIL", buf);
}

/**
 * Publish device heartbeat / status to falcon/center/{centerId}/device-status.
 * Matches the DeviceStatusPayload interface in mqtt-payload.interface.ts.
 */
void publishHeartbeat(const char* statusEvent = "ONLINE") {
    StaticJsonDocument<256> doc;
    doc["centerId"]    = (strlen(g_centerId) > 0) ? g_centerId : "unprovisioned";
    doc["timestamp"]   = (unsigned long)(millis() / 1000UL);
    doc["deviceId"]    = g_macDash;
    doc["macAddress"]  = g_macDash;
    doc["deviceType"]  = "MICROPHONE";
    doc["status"]      = statusEvent;
    doc["ipAddress"]   = WiFi.localIP().toString();
    doc["firmwareVer"] = FIRMWARE_VERSION;
    // Custom fields (not in base interface, ignored by backend if unexpected)
    doc["uptimeMs"]    = millis();
    doc["sampleRate"]  = g_sampleRate;
    doc["muted"]       = g_muted;

    char buf[256];
    serializeJson(doc, buf, sizeof(buf));

    const char* topic =
        (g_state == DeviceState::PROVISIONED)
            ? TOPIC_DEVICE_STATUS
            : "falcon/status";   // Fallback before provisioning

    g_mqtt.publish(topic, buf, false);
    Serial.printf("[MQTT] Heartbeat → %s\n", topic);
}

/**
 * Publish provisioning ACK to falcon/provision/ack/{macTopic}.
 * Matches ProvisionAckPayload in discovery.controller.ts.
 */
void publishProvisionAck(bool success, const char* message = nullptr) {
    StaticJsonDocument<128> doc;
    doc["macAddress"] = g_macDash;
    doc["status"]     = success ? "OK" : "ERROR";
    if (message) doc["message"] = message;

    char buf[128];
    serializeJson(doc, buf, sizeof(buf));
    g_mqtt.publish(TOPIC_PROVISION_ACK, buf, false);
    Serial.printf("[MQTT] Provision ACK → %s : %s\n", TOPIC_PROVISION_ACK, buf);
}

// =============================================================================
//  MQTT COMMAND HANDLER
//  Receives commands on:
//   • falcon/provision/{macTopic}   ← provisioning config from backend
//   • falcon/cmd/{macTopic}         ← runtime commands from Super Admin
// =============================================================================

/**
 * Handle a provisioning config payload from the backend.
 *
 * Expected JSON (from ProvisioningService.assign()):
 * {
 *   "centerId":     "clxyz123",
 *   "centerCode":   "FAL-LGS-001",
 *   "tableId":      "cltable456",        // optional
 *   "wifiSsid":     "FalconBranch-LGS001",
 *   "wifiPassword": "SecureWifi2024!",
 *   "serverUrl":    "http://...",
 *   "mqttUrl":      "mqtt://192.168.1.50:1883",
 *   "provisionedAt":"2026-04-11T..."
 * }
 */
void handleProvisionConfig(const uint8_t* payload, unsigned int len) {
    StaticJsonDocument<512> doc;
    DeserializationError err = deserializeJson(doc, payload, len);
    if (err) {
        Serial.printf("[PROV] JSON parse error: %s\n", err.c_str());
        publishProvisionAck(false, "JSON parse error");
        return;
    }

    const char* centerId = doc["centerId"] | "";
    const char* tableId  = doc["tableId"]  | "";
    const char* mqttUrl  = doc["mqttUrl"]  | "";

    if (strlen(centerId) == 0) {
        publishProvisionAck(false, "Missing centerId");
        return;
    }

    // Parse mqttUrl: "mqtt://hostname:port"
    char newMqttHost[128] = DEFAULT_MQTT_HOST;
    int  newMqttPort      = DEFAULT_MQTT_PORT;
    if (strlen(mqttUrl) > 7) {
        const char* hostStart = mqttUrl;
        if (strncmp(mqttUrl, "mqtt://", 7) == 0) hostStart = mqttUrl + 7;
        char tmp[128];
        strncpy(tmp, hostStart, sizeof(tmp) - 1);
        char* colon = strchr(tmp, ':');
        if (colon) {
            *colon = '\0';
            newMqttPort = atoi(colon + 1);
        }
        strncpy(newMqttHost, tmp, sizeof(newMqttHost) - 1);
    }

    // WiFi credential update: if new SSID provided, trigger WiFiManager reset
    const char* wifiSsid = doc["wifiSsid"] | "";
    const char* wifiPass = doc["wifiPassword"] | "";
    if (strlen(wifiSsid) > 0) {
        Serial.printf("[PROV] New WiFi credentials received for SSID: %s\n", wifiSsid);
        WiFiManager wm;
        wm.resetSettings();                          // Clear old credentials
        WiFi.begin(wifiSsid, wifiPass);              // Try connecting immediately
        unsigned long t0 = millis();
        while (WiFi.status() != WL_CONNECTED && (millis() - t0) < 15000) {
            delay(500);
            Serial.print(".");
        }
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("\n[PROV] WiFi connect failed — keeping old credentials");
            // Don't abort — still save center/MQTT config
        } else {
            Serial.printf("\n[PROV] WiFi connected: %s\n", WiFi.localIP().toString().c_str());
        }
    }

    // Persist config
    saveProvisionConfig(centerId, tableId, newMqttHost, newMqttPort);
    buildTopics();

    Serial.printf("[PROV] Provisioned! center=%s table=%s mqtt=%s:%d\n",
                  centerId, tableId, newMqttHost, newMqttPort);

    // Send ACK before changing state (broker still connected)
    publishProvisionAck(true, "Device provisioned successfully");

    // Transition to PROVISIONED — I2S task will be started from loop()
    g_state = DeviceState::PROVISIONED;
}

/**
 * Handle runtime commands from the Super Admin dashboard.
 *
 * Expected JSON on falcon/cmd/{macTopic}:
 * { "cmd": "MUTE" }
 * { "cmd": "UNMUTE" }
 * { "cmd": "SET_SAMPLE_RATE", "value": 8000 }
 * { "cmd": "REBOOT" }
 * { "cmd": "GET_STATUS" }
 */
void handleCommand(const uint8_t* payload, unsigned int len) {
    StaticJsonDocument<128> doc;
    if (deserializeJson(doc, payload, len) != DeserializationError::Ok) return;

    const char* cmd = doc["cmd"] | "";
    Serial.printf("[CMD] Received command: %s\n", cmd);

    if (strcmp(cmd, "MUTE") == 0) {
        g_muted = true;
        saveAudioSettings();
        Serial.println("[CMD] Microphone MUTED");

    } else if (strcmp(cmd, "UNMUTE") == 0) {
        g_muted = false;
        saveAudioSettings();
        Serial.println("[CMD] Microphone UNMUTED");

    } else if (strcmp(cmd, "SET_SAMPLE_RATE") == 0) {
        int newRate = doc["value"] | 16000;
        // Validate: only allow standard rates
        if (newRate == 8000 || newRate == 16000 || newRate == 22050 || newRate == 44100) {
            g_sampleRate  = newRate;
            g_restartI2S  = true;    // Signal I2S task to restart with new rate
            saveAudioSettings();
            Serial.printf("[CMD] Sample rate changing to %d Hz\n", newRate);
        } else {
            Serial.printf("[CMD] Invalid sample rate: %d — rejected\n", newRate);
        }

    } else if (strcmp(cmd, "REBOOT") == 0) {
        Serial.println("[CMD] Reboot command received — rebooting in 1s");
        publishHeartbeat("REBOOT");
        delay(1000);
        ESP.restart();

    } else if (strcmp(cmd, "GET_STATUS") == 0) {
        publishHeartbeat("ONLINE");

    } else {
        Serial.printf("[CMD] Unknown command: %s\n", cmd);
    }
}

// =============================================================================
//  MQTT CALLBACK  (runs in main task / loop())
// =============================================================================

void mqttCallback(char* topic, uint8_t* payload, unsigned int length) {
    Serial.printf("[MQTT] Message on [%s] len=%u\n", topic, length);

    if (strcmp(topic, TOPIC_PROVISION_IN) == 0) {
        // ── Provisioning config from Super Admin ──────────────────────────────
        handleProvisionConfig(payload, length);

    } else if (strcmp(topic, TOPIC_CMD) == 0) {
        // ── Runtime command from Super Admin ──────────────────────────────────
        handleCommand(payload, length);
    }
}

// =============================================================================
//  MQTT CONNECTION & SUBSCRIPTIONS
// =============================================================================

/**
 * Build a unique MQTT client ID: "falcon-mic-AABBCCDDEEFF"
 */
void buildClientId(char* out, size_t outLen) {
    snprintf(out, outLen, "falcon-mic-%02X%02X%02X%02X%02X%02X",
             (uint8_t)g_macDash[0],   // Quick hack — use WiFi MAC bytes
             WiFi.macAddress()[0], WiFi.macAddress()[1], WiFi.macAddress()[2],
             WiFi.macAddress()[3], WiFi.macAddress()[4]);
    // Actually use numeric MAC bytes from WiFi
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(out, outLen, "falcon-mic-%02X%02X%02X%02X%02X%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

/**
 * Connect / reconnect to MQTT broker and subscribe to device-specific topics.
 * Publishes an offline LWT (Last Will & Testament) so the backend detects drops.
 */
bool mqttConnect() {
    char clientId[32];
    buildClientId(clientId, sizeof(clientId));

    // ── Last Will & Testament ─────────────────────────────────────────────────
    // Backend receives this if the TCP connection drops unexpectedly
    StaticJsonDocument<128> lwt;
    lwt["centerId"]   = (strlen(g_centerId) > 0) ? g_centerId : "unprovisioned";
    lwt["deviceId"]   = g_macDash;
    lwt["macAddress"] = g_macDash;
    lwt["deviceType"] = "MICROPHONE";
    lwt["status"]     = "OFFLINE";
    lwt["timestamp"]  = (unsigned long)(millis() / 1000UL);
    char lwtBuf[128];
    serializeJson(lwt, lwtBuf, sizeof(lwtBuf));

    const char* lwtTopic =
        (g_state == DeviceState::PROVISIONED)
            ? TOPIC_DEVICE_STATUS
            : "falcon/status";

    g_mqtt.setServer(g_mqttHost, g_mqttPort);
    g_mqtt.setCallback(mqttCallback);
    g_mqtt.setKeepAlive(MQTT_KEEPALIVE_S);
    g_mqtt.setBufferSize(1024);

    Serial.printf("[MQTT] Connecting to %s:%d as %s …\n",
                  g_mqttHost, g_mqttPort, clientId);

    bool ok = g_mqtt.connect(
        clientId,
        nullptr, nullptr,       // no username/password (secured by network)
        lwtTopic,
        MQTT_QOS1,
        true,                   // LWT retained
        lwtBuf
    );

    if (!ok) {
        Serial.printf("[MQTT] Connect failed, rc=%d\n", g_mqtt.state());
        return false;
    }

    Serial.println("[MQTT] Connected ✓");

    // ── Subscribe to device-specific topics ───────────────────────────────────
    g_mqtt.subscribe(TOPIC_PROVISION_IN, MQTT_QOS1);   // Provisioning config
    g_mqtt.subscribe(TOPIC_CMD,          MQTT_QOS1);   // Runtime commands

    Serial.printf("[MQTT] Subscribed to: %s\n", TOPIC_PROVISION_IN);
    Serial.printf("[MQTT] Subscribed to: %s\n", TOPIC_CMD);

    return true;
}

// =============================================================================
//  WiFi — with WiFiManager Fallback AP
// =============================================================================

/**
 * Derive the full MAC address strings from the WiFi hardware.
 * g_macDash  = "AA:BB:CC:DD:EE:FF"
 * g_macTopic = "AA_BB_CC_DD_EE_FF"  (colons → underscores for MQTT topics)
 */
void initMacStrings() {
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(g_macDash, sizeof(g_macDash), "%02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    snprintf(g_macTopic, sizeof(g_macTopic), "%02X_%02X_%02X_%02X_%02X_%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    Serial.printf("[MAC] Dash:  %s\n", g_macDash);
    Serial.printf("[MAC] Topic: %s\n", g_macTopic);
}

/**
 * Build topic strings that require the MAC address.
 * Called once after initMacStrings().
 */
void initTopicsFromMac() {
    snprintf(TOPIC_PROVISION_IN,  sizeof(TOPIC_PROVISION_IN),
             "falcon/provision/%s", g_macTopic);
    snprintf(TOPIC_PROVISION_ACK, sizeof(TOPIC_PROVISION_ACK),
             "falcon/provision/ack/%s", g_macTopic);
    snprintf(TOPIC_CMD,           sizeof(TOPIC_CMD),
             "falcon/cmd/%s", g_macTopic);
}

/**
 * Connect to WiFi.
 * - If credentials exist in flash → connect directly.
 * - If not → launch WiFiManager AP "FalconMic-XXXX" for manual setup.
 *
 * WiFiManager portal also exposes custom fields if needed.
 */
void connectWifi() {
    // Use last 4 hex chars of MAC for unique AP name
    char apName[32];
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(apName, sizeof(apName), "%s-%02X%02X",
             WIFIMANAGER_AP_NAME, mac[4], mac[5]);

    WiFiManager wm;
    wm.setConfigPortalTimeout(WIFI_CONNECT_TIMEOUT_S);
    wm.setConnectTimeout(30);
    wm.setTitle("Falcon Security — Mic Node Setup");
    wm.setMinimumSignalQuality(10);

    // If portal times out with no config, reboot and try again
    wm.setAPCallback([](WiFiManager* mgr) {
        Serial.printf("[WiFi] AP started: %s — open 192.168.4.1 to configure\n",
                      mgr->getConfigPortalSSID().c_str());
    });

    Serial.printf("[WiFi] Starting WiFiManager AP: %s\n", apName);

    // autoConnect: connect if creds exist, otherwise launch portal
    if (!wm.autoConnect(apName, WIFIMANAGER_AP_PASS)) {
        Serial.println("[WiFi] Config portal timed out — rebooting");
        ESP.restart();
    }

    Serial.printf("[WiFi] Connected! IP: %s  RSSI: %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());

    // Set hostname for mDNS / DHCP identification
    char hostname[32];
    snprintf(hostname, sizeof(hostname), "falcon-mic-%02X%02X", mac[4], mac[5]);
    WiFi.setHostname(hostname);
}

// =============================================================================
//  ARDUINO SETUP
// =============================================================================

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n\n══════════════════════════════════════");
    Serial.println("  FALCON SECURITY — AI Mic Node");
    Serial.printf ("  Firmware: %s\n", FIRMWARE_VERSION);
    Serial.println("══════════════════════════════════════");

    // ── 1. Derive MAC strings ────────────────────────────────────────────────
    initMacStrings();
    initTopicsFromMac();   // Build provision + cmd topics from MAC

    // ── 2. Load persisted config from NVS ───────────────────────────────────
    bool alreadyProvisioned = loadProvisionConfig();
    if (alreadyProvisioned) {
        buildTopics();     // Build audio + status topics from loaded centerId
        Serial.printf("[NVS] Provisioned config loaded — center=%s\n", g_centerId);
    }

    // ── 3. WiFi (with fallback AP) ───────────────────────────────────────────
    connectWifi();
    g_state = DeviceState::UNPROVISIONED;

    // ── 4. Connect to MQTT broker ────────────────────────────────────────────
    g_mqtt.setServer(g_mqttHost, g_mqttPort);
    g_mqtt.setCallback(mqttCallback);
    g_mqtt.setBufferSize(1024);

    while (!mqttConnect()) {
        Serial.println("[MQTT] Retrying in 5s…");
        delay(MQTT_RECONNECT_DELAY_MS);
    }

    // ── 5. If already provisioned, skip discovery handshake ──────────────────
    if (alreadyProvisioned) {
        g_state = DeviceState::PROVISIONED;
        publishHeartbeat("ONLINE");
        Serial.println("[PROV] Device is provisioned — starting audio task");
    } else {
        // ── 6. Start UNPROVISIONED: publish birth / discovery message ─────────
        Serial.println("[PROV] Device not provisioned — publishing birth message");
        publishBirthMessage();
        g_lastBirthMsg = millis();
    }

    // ── 7. Start I2S sampling task on Core 1 (only if provisioned) ───────────
    if (g_state == DeviceState::PROVISIONED) {
        xTaskCreatePinnedToCore(
            i2sTask,          // Task function
            "I2S_AudioTask",  // Name
            8192,             // Stack size (bytes)
            nullptr,          // Parameters
            5,                // Priority (5 = high)
            &g_i2sTaskHandle, // Handle
            1                 // Core 1 (Core 0 used by WiFi/MQTT)
        );
    }

    Serial.println("[SETUP] Setup complete ✓");
}

// =============================================================================
//  ARDUINO LOOP  (runs on Core 0)
// =============================================================================

void loop() {
    // ── MQTT keepalive ────────────────────────────────────────────────────────
    if (!g_mqtt.connected()) {
        Serial.println("[MQTT] Disconnected — reconnecting…");
        unsigned long t0 = millis();
        while (!mqttConnect()) {
            if (millis() - t0 > 60000) {
                Serial.println("[MQTT] Cannot reconnect after 60s — rebooting");
                ESP.restart();
            }
            delay(MQTT_RECONNECT_DELAY_MS);
        }
        // Re-publish birth / status after reconnect
        if (g_state == DeviceState::PROVISIONED) {
            publishHeartbeat("ONLINE");
        } else {
            publishBirthMessage();
        }
    }
    g_mqtt.loop();   // Process incoming MQTT messages

    unsigned long now = millis();

    // ── Heartbeat every 10 seconds ────────────────────────────────────────────
    if ((now - g_lastHeartbeat) >= HEARTBEAT_INTERVAL_MS) {
        g_lastHeartbeat = now;
        publishHeartbeat("ONLINE");
    }

    // ── Re-publish birth every 30s until provisioned ──────────────────────────
    if (g_state == DeviceState::UNPROVISIONED) {
        if ((now - g_lastBirthMsg) >= g_birthRetryMs) {
            g_lastBirthMsg = now;
            Serial.println("[PROV] Re-publishing birth message (not yet provisioned)");
            publishBirthMessage();
        }
    }

    // ── Start I2S task after provisioning (transition from UNPROVISIONED) ─────
    if (g_state == DeviceState::PROVISIONED && g_i2sTaskHandle == nullptr) {
        Serial.println("[PROV] Provisioning complete — starting I2S audio task");
        xTaskCreatePinnedToCore(
            i2sTask,
            "I2S_AudioTask",
            8192,
            nullptr,
            5,
            &g_i2sTaskHandle,
            1
        );
    }

    // ── WiFi watchdog: reconnect if dropped ───────────────────────────────────
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Connection lost — reconnecting…");
        WiFi.reconnect();
        unsigned long t0 = millis();
        while (WiFi.status() != WL_CONNECTED && (millis() - t0) < 30000) {
            delay(500);
            Serial.print(".");
        }
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("\n[WiFi] Reconnect failed — rebooting");
            ESP.restart();
        }
        Serial.printf("\n[WiFi] Reconnected: %s\n", WiFi.localIP().toString().c_str());
    }

    delay(10);  // Yield to RTOS scheduler
}

// =============================================================================
//  END OF FIRMWARE
// =============================================================================
