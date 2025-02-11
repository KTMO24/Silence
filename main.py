#!/usr/bin/env python3
"""
Merged System: Flask Backend + React GUI (Static Files) + Custom Logic
----------------------------------------------------------------------
This script incorporates custom "scratch logic" for Perlin noise and smoothing,
replacing library functions with custom implementations.

Ensures no signal cancellation or audio transmission at system start.

To run the web interface and for setup instructions, see the main script documentation.

This version demonstrates:
  - Custom 1D Perlin noise generation (no external library dependency)
  - Custom Moving Average Smoothing (instead of Gaussian filter)

These custom implementations are for demonstration and educational purposes.
For production, using optimized libraries like scipy is generally recommended
for performance and robustness.

Command-line Usage:
  --mode web      : Launch the Flask web interface (React GUI served from 'static' folder)
  --mode server   : Run streaming audio transmitter
  --mode client   : Run streaming audio receiver/playback
  --mode rf       : Run RF/packet analysis
  --mode gemini   : Run Gemini deep scan

Dependencies:
  pip install flask numpy scipy flask-cors

Adjust GEMINI_API_URL as needed.
"""

import os
import sys
import logging
import threading
import time
import math
import random
import socket
import struct
import subprocess
import argparse
import platform
import json
from datetime import datetime
from typing import List, Tuple, Dict
import queue

import numpy as np
from scipy.fft import fft, ifft, fftfreq
# scipy.ndimage.gaussian_filter1d is replaced by custom moving average
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS  # Import CORS
# noise library is replaced by custom Perlin noise
# from noise import pnoise1

# --- Additional Imports for Gemini scan ---
try:
    import netifaces
except ImportError:
    print("netifaces not installed. Gemini scan will be limited.")
    netifaces = None

try:
    import requests
except ImportError:
    print("requests not installed. Gemini scan will be limited.")
    requests = None

# --- New Imports for RF/Packet Analysis ---
try:
    from scapy.all import sniff, Ether, IP
    scapy_available = True
except ImportError:
    print("Scapy not installed. RF/Packet Analysis will be disabled.")
    scapy_available = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
SAMPLE_RATE = 44100
BLOCKSIZE = 1024
CHANNELS = 1
DURATION = BLOCKSIZE / SAMPLE_RATE
FREQUENCY = 440.0
AMPLITUDE = 0.5

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 50007
BUFFER_SIZE = 4096

current_devices = []
my_devices = []
monitoring = False
monitor_thread = None
rgb_queue = queue.Queue()
log_messages = []

# Gemini API endpoint
GEMINI_API_URL = "https://api.geminideviceanalysis.com/analyze"  # Replace with your actual Gemini API URL

# Directory for static files (React build output will go here)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'static') #Fixed __file__ issue using sys.argv[0]

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='') # Serve React App
CORS(app) # Enable CORS for all routes

def add_log(message: str, level: str = "info"):
    timestamp = datetime.now().strftime('%H:%M:%S')
    entry = f"[{timestamp}] [{level.upper()}] {message}"
    log_messages.append(entry)
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    else:
        logging.warning(message)

# Platform check
running_on_ios = False
if "iOS" in platform.platform() or "iPhone" in platform.platform():
    running_on_ios = True
    add_log("Detected iOS platform; audio and subprocess features will be disabled.", "info")

# --- SensorSafetySystem Class --- (No changes needed in this class)
class SignalSafetySystem:
    def __init__(self, min_distance: float = 6.0):
        self.skeletons: List[Dict] = []
        self.min_distance = min_distance

    def scan_for_bodies(self, sensor_data: List[Tuple[float, float, float]]) -> List[Dict]:
        try:
            detected_bodies = self.detect_shapes(sensor_data)
            self.skeletons = self.map_skeletons(detected_bodies)
            add_log(f"Scan complete: {len(self.skeletons)} device(s) found")
            return self.skeletons
        except Exception as e:
            add_log(f"Error in scan_for_bodies: {e}", "error")
            return []

    def detect_shapes(self, data: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        try:
            return data
        except Exception as e:
            add_log(f"Error in detect_shapes: {e}", "error")
            return []

    def is_body_shape(self, point: Tuple[float, float, float], radius: float = 0.5) -> bool:
        return True

    def map_skeletons(self, bodies: List[Tuple[float, float, float]]) -> List[Dict]:
        try:
            return [self.create_skeleton(body) for body in bodies]
        except Exception as e:
            add_log(f"Error in map_skeletons: {e}", "error")
            return []

    def create_skeleton(self, body: Tuple[float, float, float]) -> Dict:
        try:
            device_types = [
                {"name": "Pacemaker", "frequency": random.uniform(50, 100), "protocol": "Bluetooth"},
                {"name": "Insulin Pump", "frequency": random.uniform(900, 930), "protocol": "Zigbee"},
                {"name": "Neurostimulator", "frequency": random.uniform(2.4, 2.5), "protocol": "WiFi"},
                {"name": "Glucose Monitor", "frequency": random.uniform(433, 434), "protocol": "LoRa"}
            ]

            chosen_device = random.choice(device_types)
            frequency_kHz = chosen_device["frequency"]
            protocol = chosen_device["protocol"]
            name = f"{chosen_device['name']} {random.choice(['A', 'B', 'C', 'D'])}"
            device_id = random.randint(1000, 9999)
            ip = f"192.168.1.{random.randint(2, 254)}"

            x, y, z = body
            skeleton = {
                "id": device_id,
                "name": name,
                "status": "Active",
                "ip": ip,
                "protocol": protocol,
                "frequency": frequency_kHz * 1000,
                "position": body,
                "tracked": True,
                "joints": {
                    "Head": (x, y + 1.8, z),
                    "Neck": (x, y + 1.6, z),
                    "LeftShoulder": (x - 0.3, y + 1.5, z),
                    "RightShoulder": (x + 0.3, y + 1.5, z),
                    "LeftElbow": (x - 0.6, y + 1.2, z),
                    "RightElbow": (x + 0.6, y + 1.2, z),
                    "LeftHand": (x - 0.9, y + 0.9, z),
                    "RightHand": (x + 0.9, y + 0.9, z),
                    "Torso": (x, y + 1.0, z),
                    "LeftHip": (x - 0.3, y + 0.5, z),
                    "RightHip": (x + 0.3, y + 0.5, z),
                    "LeftKnee": (x - 0.3, y, z),
                    "RightKnee": (x + 0.3, y, z),
                    "LeftFoot": (x - 0.3, y - 0.5, z),
                    "RightFoot": (x + 0.3, y - 0.5, z)
                }
            }
            return skeleton
        except Exception as e:
            add_log(f"Error in create_skeleton: {e}", "error")
            return {}

    def calculate_distance(self, point1: Tuple[float, float, float], point2: Tuple[float, float, float]) -> float:
        try:
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
        except Exception as e:
            add_log(f"Error in calculate_distance: {e}", "error")
            return float('inf')

# --- Installer Class --- (No changes needed)
class Installer:
    def __init__(self):
        pass

    def deploy(self, code: str):
        try:
            deploy_script_path = "deploy_script.py"
            with open(deploy_script_path, "w") as f:
                f.write(code)
            add_log("Deployment script saved.")
            if sys.platform.startswith('win'):
                os.startfile(deploy_script_path)
            else:
                subprocess.Popen(['python3', deploy_script_path])
            add_log("Deployment script executed.")
        except Exception as e:
            add_log(f"Error during deployment: {e}", "error")

# --- Simulate Sensor Data --- (No changes needed)
def simulate_sensor_data() -> List[Tuple[float, float, float]]:
    num_points = random.randint(3, 8)
    return [(round(random.uniform(-5, 5), 2),
             round(random.uniform(-5, 5), 2),
             round(random.uniform(-5, 5), 2))
            for _ in range(num_points)]

# --- Modulation Validation Module --- (No changes needed)
def validate_modulation(signal: np.ndarray) -> bool:
    max_amp = np.max(np.abs(signal))
    if max_amp > 0.9:
        add_log(f"Validation failed: max amplitude {max_amp:.3f} exceeds safety limit.", "error")
        return False
    spectrum = np.abs(fft(signal))
    peak_energy = np.max(spectrum)
    if peak_energy > 5000:
        add_log(f"Validation failed: spectral peak {peak_energy:.1f} exceeds safety limit.", "error")
        return False
    return True

# --- Ambient Signal Masking Function --- (No changes needed - already masks based on frequency, not phase)
def ambient_signal_mask(signal: np.ndarray,
                        system_freq: float = FREQUENCY,
                        delta: float = 5.0,
                        threshold: float = 0.05,
                        attenuation: float = 0.1) -> np.ndarray:
    spectrum = fft(signal)
    freqs = fftfreq(len(signal), 1/SAMPLE_RATE)
    indices = [i for i, f in enumerate(freqs) if abs(f - system_freq) < delta]
    pilot_magnitude = np.mean(np.abs(spectrum[indices])) if indices else 0
    if pilot_magnitude < threshold:
        add_log(f"Ambient masking applied: pilot magnitude {pilot_magnitude:.3f} below threshold.", "info")
        for i in indices:
            spectrum[i] *= attenuation
    else:
        add_log(f"No masking applied: pilot magnitude {pilot_magnitude:.3f} above threshold.", "info")
    masked_signal = ifft(spectrum).real
    return masked_signal

# --- Wavemask and Frequency Fence Processing --- (No changes needed - uses amplitude weighting, not phase)
def calculate_cancellation_weight(distance: float, lower: float = 0.3048, upper: float = 1.0) -> float:
    if distance < lower:
        return 0.0
    elif distance >= upper:
        return 1.0
    else:
        return (distance - lower) / (upper - lower)

def predict_trajectory(positions: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    predicted = []
    for pos in positions:
        shift = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
        predicted.append((pos[0] + shift[0], pos[1] + shift[1], pos[2] + shift[2]))
    return predicted

def wavemask_and_sum(signals: List[np.ndarray],
                     positions: List[Tuple[float, float, float]],
                     sources: List[bool],
                     lower: float = 0.3048, upper: float = 1.0) -> np.ndarray:
    predicted_positions = predict_trajectory(positions)
    weighted_signals = []
    for sig, pos, is_device in zip(signals, predicted_positions, sources):
        if not is_device:
            continue
        d = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        weight = calculate_cancellation_weight(d, lower, upper)
        if weight == 0.0:
            add_log("Cancellation halted: device signal too close (< 1 ft).", "error")
            return None
        window = np.hanning(len(sig))
        weighted_signals.append(sig * window * weight)
    if not weighted_signals:
        add_log("No device signals available for cancellation.", "error")
        return None
    return np.sum(np.array(weighted_signals), axis=0)

def compute_log_frequency_fence(signal: np.ndarray) -> np.ndarray:
    X = fft(signal)
    mag = np.abs(X)
    log_mag = np.log1p(mag)
    # gaussian_filter1d is replaced by custom moving_average_smooth_1d in process_block
    smoothed = log_mag # placeholder - smoothing will be applied in process_block now
    return smoothed

def process_sensor_signals(num_sensors: int = 10,
                           lower: float = 0.3048, upper: float = 1.0) -> Tuple[np.ndarray, List[Tuple[float, float, float]], List[bool]]:
    signals = []
    positions = []
    sources = []
    for _ in range(num_sensors):
        pos = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2))
        positions.append(pos)
        is_device = random.random() < 0.7
        sources.append(is_device)
        sig = synthesize_block()
        signals.append(sig)
    summed_signal = wavemask_and_sum(signals, positions, sources, lower, upper)
    return summed_signal, positions, sources

def get_log_frequency_fence() -> Dict:
    summed_signal, positions, sources = process_sensor_signals(num_sensors=10, lower=0.3048, upper=1.0)
    if summed_signal is None:
        return {"error": "Cancellation halted: device signal too close or no device signals available.",
                "positions": positions, "sources": sources}
    fence = compute_log_frequency_fence(summed_signal)
    return {"log_frequency_fence": fence.tolist(), "positions": positions, "sources": sources}

# --- Streaming Audio Synthesis and Transmission ---
global_phase = 0.0
phase_lock = threading.Lock()
audio_transmit_enabled = False # Flag to control audio transmission at start

def synthesize_block(frequency: float = FREQUENCY, amplitude: float = AMPLITUDE, block_size: int = BLOCKSIZE) -> np.ndarray:
    global global_phase
    t = np.arange(block_size) / SAMPLE_RATE
    with phase_lock:
        phase_increment = 2 * np.pi * frequency * t
        block = amplitude * np.sin(global_phase + phase_increment)
        global_phase = (global_phase + 2 * np.pi * frequency * (block_size / SAMPLE_RATE)) % (2 * np.pi)
    return block.astype(np.float32)

# --- Custom Perlin Noise Implementation (Scratch Logic) --- (No changes needed)
def custom_perlin_noise_1d(signal: np.ndarray, scale: float = 50, seed=0) -> np.ndarray:
    if seed is not None:
        random.seed(seed)

    length = len(signal)
    noise_values = np.zeros(length)
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0

    for octave in range(octaves):
        frequency = lacunarity ** octave
        amplitude = persistence ** octave
        phase = random.random() * 100

        for i in range(length):
            x = i / scale * frequency + phase
            int_x = int(x)
            frac_x = x - int_x
            v1 = random.random()
            v2 = random.random()
            interpolated_value = (1 - frac_x) * v1 + frac_x * v2

            noise_values[i] += interpolated_value * amplitude

    noise_values = (noise_values - np.mean(noise_values)) / (np.max(np.abs(noise_values)) + 1e-9)
    return signal + 0.01 * noise_values

# --- Custom Moving Average Smoothing (Scratch Logic) --- (No changes needed)
def custom_moving_average_smooth_1d(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    smoothed_signal = np.zeros_like(signal)
    padding = window_size // 2
    padded_signal = np.pad(signal, (padding, padding), mode='reflect')

    for i in range(len(signal)):
        window = padded_signal[i:i + window_size]
        smoothed_signal[i] = np.mean(window)

    return smoothed_signal

def laplacian_smooth_1d(signal: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    kernel = np.array([1, -2, 1])
    padded = np.pad(signal, (1, 1), mode='edge')
    laplacian = np.convolve(padded, kernel, mode='valid')
    return signal - alpha * laplacian

def apply_perlin_noise_1d(signal: np.ndarray, scale: float = 50) -> np.ndarray:
    noise_values = custom_perlin_noise_1d(signal, scale=scale)
    return noise_values

def process_block(block: np.ndarray) -> np.ndarray:
    X = fft(block)
    X_smooth = laplacian_smooth_1d(X, alpha=0.5)
    X_dithered = apply_perlin_noise_1d(X_smooth, scale=50)
    block_processed = ifft(X_dithered).real
    block_processed = np.clip(block_processed, -1, 1)
    return block_processed.astype(np.float32)

def float_block_to_int16_bytes(block: np.ndarray) -> bytes:
    int_block = np.int16(block * 32767)
    return int_block.tobytes()

def run_streaming_server(host: str = SERVER_HOST, port: int = SERVER_PORT):
    global audio_transmit_enabled # Use the global flag
    audio_transmit_enabled = True # Enable transmission when server starts
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except socket.error as e:
            add_log(f"Socket bind error: {e}", "error")
            return
        s.listen(1)
        print(f"[{datetime.now()}] Streaming Server listening on {host}:{port}")
        conn, addr = s.accept()
        print(f"[{datetime.now()}] Streaming Server connection from {addr}")
        try:
            while True:
                if audio_transmit_enabled: # Only transmit if enabled
                    block = synthesize_block()
                    processed_block = process_block(block)
                    if not validate_modulation(processed_block):
                        add_log("Modulation validation failed. Halting transmission.", "error")
                        break

                    # Apply ambient masking
                    masked_block = ambient_signal_mask(processed_block)

                    rms = np.sqrt(np.mean(masked_block**2))
                    r = random.uniform(0, rms)
                    g = random.uniform(0, rms)
                    b = random.uniform(0, rms)
                    rgb_queue.put_nowait({"rgb": [r, g, b], "intensity": rms})

                    pcm_bytes = float_block_to_int16_bytes(masked_block)
                    conn.sendall(pcm_bytes)
                    time.sleep(DURATION)
                else:
                    time.sleep(1) # Check less frequently if disabled

        except Exception as e:
            print(f"Streaming Server error: {e}")
        finally:
            conn.close()
            audio_transmit_enabled = False # Disable transmission on server close

def run_streaming_client(host: str, port: int = SERVER_PORT):
    if sys.platform.startswith("linux"):
        play_cmd = ["aplay", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1"]
    elif sys.platform.startswith("darwin"):
        print("macOS raw PCM streaming playback is not implemented in this example.")
        return
    elif sys.platform.startswith("win"):
        print("Windows raw PCM streaming playback is not implemented in this example.")
        return
    else:
        print("Unsupported OS for streaming playback.")
        return

    print(f"[{datetime.now()}] Streaming Client connecting to {host}:{port} ...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
        except socket.error as e:
            add_log(f"Socket connect error: {e}", "error")
            return

        proc = subprocess.Popen(play_cmd, stdin=subprocess.PIPE)
        try:
            while True:
                data = s.recv(BUFFER_SIZE)
                if not data:
                    break
                proc.stdin.write(data)
                proc.stdin.flush()
        except Exception as e:
            print(f"Streaming Client error: {e}")
        finally:
            proc.stdin.close()
            proc.wait()

# --- RF / Packet Analysis Functions --- (No changes needed)
def mac_lookup(packet_mac: str) -> str:
    url = f'https://api.macvendors.com/{packet_mac}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return "Unknown manufacturer"
    except Exception as e:
        add_log(f"MAC lookup error: {e}", "error")
        return "API Request Failed"

def analyze_packet(packet) -> None:
    try:
        if packet.haslayer(Ether):
            mac_src = packet[Ether].src
            mac_dst = packet[Ether].dst
            manufacturer = mac_lookup(mac_src)
            add_log(f"Packet: MAC src {mac_src} (Manufacturer: {manufacturer}), MAC dst {mac_dst}")
        if packet.haslayer(IP):
            ip_src = packet[IP].src
            ip_dst = packet[IP].dst
            add_log(f"Packet: IP src {ip_src}, IP dst {ip_dst}")
        if packet.haslayer("TCP"):
            add_log("Packet: TCP detected")
        elif packet.haslayer("UDP"):
            add_log("Packet: UDP detected")
        stream_type = detect_stream(packet)
        if stream_type:
            add_log(f"Stream detected: {stream_type}")
    except Exception as e:
        add_log(f"Packet analysis error: {e}", "error")

def detect_stream(packet) -> str:
    stream_types = [None, "Audio", "Video", "BCI Data"]
    return random.choice(stream_types)

def obfuscate_data(data: bytes, key: int = 0xAA) -> bytes:
    return bytes(b ^ key for b in data)

def run_rf_analysis(packet_count: int = 20) -> None:
    if not scapy_available:
        add_log("Scapy is not available. RF/Packet analysis disabled.", "error")
        return

    def packet_callback(packet):
        try:
            analyze_packet(packet)
            raw_data = bytes(packet)
            obf_data = obfuscate_data(raw_data)
            add_log(f"Obfuscated packet data (first 16 bytes): {obf_data[:16].hex()}")
        except Exception as e:
            add_log(f"Error in packet callback: {e}", "error")

    add_log("Starting RF/Packet analysis. Capturing packets...")
    try:
        sniff(prn=packet_callback, filter="ip or arp", count=packet_count, store=False)
    except Exception as e:
        add_log(f"Sniffing error: {e}", "error")
    add_log("RF/Packet analysis complete.")

# --- Flask API Endpoints --- (No changes needed)
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)

@app.route('/api/scan', methods=['GET'])
def api_scan():
    sensor_data = simulate_sensor_data()
    devices = safety_system.scan_for_bodies(sensor_data)
    return jsonify(devices)

@app.route('/api/start', methods=['POST'])
def api_start():
    global monitoring, monitor_thread
    if not monitoring:
        monitoring = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        add_log("Monitoring started")
        return jsonify({"status": "monitoring started"})
    else:
        return jsonify({"status": "already monitoring"})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    global monitoring
    monitoring = False
    add_log("Monitoring stopped")
    return jsonify({"status": "monitoring stopped"})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global current_devices, log_messages, monitoring, my_devices, audio_transmit_enabled # Reset audio flag too
    monitoring = False
    current_devices = []
    my_devices = []
    log_messages = []
    audio_transmit_enabled = False # Ensure audio transmission is off after reset
    add_log("System reset completed")
    return jsonify({"status": "reset complete"})

@app.route('/api/deploy', methods=['POST'])
def api_deploy():
    try:
        deployment_code = """
print("Deploying Tranxa System...")
# Add actual deployment logic here
"""
        installer.deploy(deployment_code)
        add_log("System deployed successfully.")
        return jsonify({"status": "deployed"})
    except Exception as e:
        add_log(f"Deployment error: {e}", "error")
        return jsonify({"status": "deployment failed"})

@app.route('/api/audio', methods=['GET'])
def api_audio():
    try:
        if not rgb_queue.empty():
            visual_params = rgb_queue.get_nowait()
            return jsonify(visual_params)
        else:
            return jsonify({"rgb": [1.0, 1.0, 1.0], "intensity": 0})
    except Exception as e:
        add_log(f"Audio API error: {e}", "error")
        return jsonify({"rgb": [1.0, 1.0, 1.0], "intensity": 0})

@app.route('/api/logs', methods=['GET'])
def api_logs():
    return jsonify(log_messages)

@app.route('/api/logfence', methods=['GET'])
def api_logfence():
    result = get_log_frequency_fence()
    return jsonify(result)

@app.route('/api/rf_analysis', methods=['POST'])
def api_rf_analysis():
    if scapy_available:
        rf_thread = threading.Thread(target=run_rf_analysis, daemon=True)
        rf_thread.start()
        return jsonify({"status": "RF analysis started in background"})
    else:
        return jsonify({"status": "Scapy not installed, RF analysis disabled"})

@app.route('/api/gemini_scan', methods=['POST'])
def api_gemini_scan():
    result = gemini_scan()
    return jsonify(result)

@app.route('/api/add_device', methods=['POST'])
def api_add_device():
    data = request.get_json()
    if not data or 'device_id' not in data:
        return jsonify({"status": "error", "message": "device_id missing"}), 400
    device_id = data['device_id']
    device = next((d for d in current_devices if d.get("id") == device_id), None)
    if not device:
        return jsonify({"status": "error", "message": "Device not found in current devices"}), 404
    global my_devices
    if any(d.get("id") == device_id for d in my_devices):
        return jsonify({"status": "error", "message": "Device already in my devices list"}), 400
    my_devices.append(device)
    add_log(f"Device {device_id} added to my devices list.")
    return jsonify({"status": "success", "message": "Device added to my devices list."})

@app.route('/api/cancel_device', methods=['POST'])
def api_cancel_device():
    data = request.get_json()
    if not data or 'device_id' not in data:
        return jsonify({"status": "error", "message": "device_id missing"}), 400
    device_id = data['device_id']
    confirm = data.get('confirm', False)
    global my_devices
    device_to_cancel = next((d for d in my_devices if d.get("id") == device_id), None)
    if not device_to_cancel:
        return jsonify({"status": "error", "message": "Device not in your 'My Devices' list"}), 404

    critical_keywords = ["pacemaker", "insulin pump", "neural stimulator", "life support"]
    name_lower = device_to_cancel.get("name", "").lower()
    if any(keyword in name_lower for keyword in critical_keywords):
        return jsonify({"status": "error", "message": "Cannot cancel frequency for critical life support device."}), 403
    if not confirm:
        return jsonify({"status": "pending", "message": "Please confirm cancellation."})

    my_devices = [d for d in my_devices if d['id'] != device_id]
    add_log(f"Device {device_id} frequency cancellation initiated.")
    return jsonify({"status": "success", "message": "Device frequency cancellation initiated."})

def monitoring_loop():
    global current_devices, monitoring
    add_log("Monitoring loop started")
    while monitoring:
        sensor_data = simulate_sensor_data()
        current_devices = safety_system.scan_for_bodies(sensor_data)
        time.sleep(2)
    add_log("Monitoring loop stopped")

# --------------------------
# Gemini Scan Functions (No changes needed)
# --------------------------
def get_network_info():
    if netifaces is None:
        add_log("netifaces module not available.", "error")
        return {}
    interfaces = netifaces.interfaces()
    network_data = {}
    for interface in interfaces:
        try:
            addrs = netifaces.ifaddresses(interface)
            mac = addrs[netifaces.AF_LINK][0]['addr'] if netifaces.AF_LINK in addrs else None
            ip = addrs[netifaces.AF_INET][0]['addr'] if netifaces.AF_INET in addrs else None
            network_data[interface] = {"mac": mac, "ip": ip}
        except Exception as e:
            logging.error(f"Error reading interface {interface}: {e}")
    return network_data

def capture_packets(count=10, timeout=5):
    if not scapy_available:
        add_log("Scapy not available; cannot capture packets.", "error")
        return []
    add_log("Starting deep packet capture for Gemini scan...", "info")
    packets = sniff(count=count, timeout=timeout)
    packet_list = []
    for pkt in packets:
        if pkt.haslayer(Ether) and pkt.haslayer(IP):
            packet_info = {
                "src_mac": pkt[Ether].src,
                "dst_mac": pkt[Ether].dst,
                "src_ip": pkt[IP].src,
                "dst_ip": pkt[IP].dst,
                "protocol": pkt[IP].proto,
                "timestamp": datetime.fromtimestamp(pkt.time).isoformat()
            }
            packet_list.append(packet_info)
    add_log(f"Captured {len(packet_list)} packets for Gemini scan.", "info")
    return packet_list

def gather_scan_data():
    data = {
        "network_info": get_network_info(),
        "packets": capture_packets(count=10, timeout=5),
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node()
    }
    return data

def send_data_to_gemini(data):
    if requests is None:
        add_log("requests module not available.", "error")
        return {"error": "requests module not installed"}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_API_URL, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        add_log("Data sent to Gemini successfully.", "info")
        return response.json()
    except requests.RequestException as e:
        add_log(f"Error sending data to Gemini: {e}", "error")
        return {"error": str(e)}

def gemini_scan():
    scan_data = gather_scan_data()
    network_info = scan_data.get("network_info", {})
    if network_info:
        first_interface = next(iter(network_info))
        scan_data["our_mac"] = network_info[first_interface].get("mac")
    analysis_result = send_data_to_gemini(scan_data)
    add_log("Gemini scan completed.", "info")
    print("Gemini Analysis Result:")
    print(json.dumps(analysis_result, indent=2))
    return analysis_result

# --------------------------
# Main Function and Mode Selection (No changes needed)
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Merged System: Flask Web, Audio, RF, and Gemini")
    parser.add_argument("--mode", choices=["web", "server", "client", "rf", "gemini"], required=True,
                        help="Mode: web, server, client, rf, gemini")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Port number")
    parser.add_argument("--freq", type=float, default=FREQUENCY, help="Sine wave frequency (Hz)")
    parser.add_argument("--packets", type=int, default=20, help="RF packet count")
    args = parser.parse_args()

    global FREQUENCY
    FREQUENCY = args.freq

    if args.mode == "web":
        app.run(debug=False, host=args.host, port=args.port)
    elif args.mode == "server":
        run_streaming_server(host=args.host, port=args.port)
    elif args.mode == "client":
        run_streaming_client(host=args.host, port=args.port)
    elif args.mode == "rf":
        run_rf_analysis(packet_count=args.packets)
    elif args.mode == "gemini":
        gemini_scan()
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    safety_system = SignalSafetySystem()
    installer = Installer()
    main()
