"""
Rock-solid Server & Multi-Tunnel Runner for Permanent Online Access
"""
import sys
import os
import re
import time
import subprocess
import threading
import requests

# Fix Windows console encoding
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def run_uvicorn():
    """Runs FastAPI uvicorn server in a persistent loop"""
    python_exe = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe")
    while True:
        try:
            print("[UVICORN] Starting FastAPI server on port 8000...")
            # Run without --reload so it never restarts on file changes
            cmd = [python_exe, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
            subprocess.run(cmd)
        except Exception as e:
            print(f"[UVICORN ERROR] {e}")
        time.sleep(2)

def run_localtunnel():
    """Runs Localtunnel with FIXED permanent subdomain: vnstock-ai"""
    time.sleep(3)
    while True:
        try:
            print("[LOCALTUNNEL] Starting permanent tunnel on https://vnstock-ai.loca.lt ...")
            proc = subprocess.Popen(
                ["npx.cmd", "localtunnel", "--port", "8000", "--subdomain", "vnstock-ai"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )
            for line in iter(proc.stdout.readline, ""):
                if "your url is" in line.lower():
                    print("\n" + "#"*65)
                    print("⭐ LINK CO DINH 100% (KHONG BAO GIO DOI):")
                    print("👉 https://vnstock-ai.loca.lt")
                    print("#"*65 + "\n")
            proc.wait()
        except Exception as e:
            print(f"[LOCALTUNNEL ERROR] {e}")
        time.sleep(3)

def run_cloudflared():
    """Runs Cloudflare tunnel as backup fast tunnel"""
    time.sleep(3)
    cf_path = os.path.abspath("cloudflared.exe")
    if not os.path.exists(cf_path):
        return

    url_pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
    while True:
        try:
            print("[CLOUDFLARE] Starting Cloudflare backup tunnel...")
            proc = subprocess.Popen(
                [cf_path, "tunnel", "--url", "http://localhost:8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )
            for line in iter(proc.stdout.readline, ""):
                match = url_pattern.search(line)
                if match:
                    cf_url = match.group(0)
                    with open("tunnel_url.txt", "w", encoding="utf-8") as f:
                        f.write(cf_url)
                    print(f"[CLOUDFLARE LIVE] {cf_url}")
                    try:
                        requests.post("http://localhost:8000/api/set-tunnel-url", json={"url": cf_url}, timeout=3)
                    except Exception:
                        pass
            proc.wait()
        except Exception as e:
            print(f"[CLOUDFLARE ERROR] {e}")
        time.sleep(3)

if __name__ == "__main__":
    t_uvicorn = threading.Thread(target=run_uvicorn, daemon=True)
    t_uvicorn.start()

    t_lt = threading.Thread(target=run_localtunnel, daemon=True)
    t_lt.start()

    run_cloudflared()
