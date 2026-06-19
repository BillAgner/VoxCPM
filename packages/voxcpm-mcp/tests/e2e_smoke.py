"""MCP end-to-end smoke test: drive voxcpm-mcp over stdio with JSON-RPC.

Sends `initialize`, `notifications/initialized`, then `tools/call` for
`voxcpm_health` and `voxcpm_synthesize`. Verifies a WAV file is produced.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def main() -> int:
    out_dir = Path(r"C:\Data\Hermes\audio_cache\voxcpm_mcp_e2e")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "HF_HUB_DISABLE_SYMLINKS": "1",
        "VOXCPM_MODEL_ID": "openbmb/VoxCPM-0.5B",
        "VOXCPM_OUTPUT_DIR": str(out_dir),
        "VOXCPM_DENOISER_ID": "none",
    }
    proc_env = {**subprocess.os.environ, **env}

    proc = subprocess.Popen(
        [r"C:\Data\Hermes\~\VoxCPM\.venv\Scripts\voxcpm-mcp.exe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=proc_env,
        bufsize=0,
    )

    def send(obj: dict) -> None:
        line = (json.dumps(obj) + "\n").encode()
        proc.stdin.write(line)
        proc.stdin.flush()

    lines: list[bytes] = []
    responses: dict[int, bytes] = {}

    def reader() -> None:
        for raw in iter(proc.stdout.readline, b""):
            try:
                obj = json.loads(raw.decode("utf-8", errors="replace"))
                if "id" in obj and isinstance(obj.get("id"), int):
                    responses[obj["id"]] = raw
                else:
                    lines.append(raw)
            except Exception:
                lines.append(raw)

    threading.Thread(target=reader, daemon=True).start()

    def call(req_id: int, method: str, params: dict | None = None, timeout: float = 240.0) -> bytes:
        """Send a JSON-RPC request and block until the matching response arrives."""
        msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        send(msg)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if req_id in responses:
                return responses.pop(req_id)
            time.sleep(0.5)
        raise TimeoutError(f"no response for id={req_id} ({method}) within {timeout}s")

    # 1) initialize (must complete before other requests)
    init_resp = call(1, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "e2e", "version": "0.1"},
    })
    print(f"init ok: serverInfo={json.loads(init_resp)['result'].get('serverInfo')}", flush=True)

    # 2) initialized notification (no response expected)
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    # 3) health check
    health_resp = call(2, "tools/call", {"name": "voxcpm_health", "arguments": {}}, timeout=30)
    print("\n--- voxcpm_health ---", flush=True)
    print(json.loads(health_resp)["result"]["content"][0]["text"][:600], flush=True)

    # 4) real synthesis — this is the test
    print("\n--- voxcpm_synthesize (lazy-loads model, ~95s) ---", flush=True)
    synth_resp = call(
        3,
        "tools/call",
        {
            "name": "voxcpm_synthesize",
            "arguments": {
                "text": "VoxCPM integration test successful. The MCP server is working.",
                "cfg_value": 2.0,
                "inference_timesteps": 10,
            },
        },
        timeout=240,
    )

    synth_block = json.loads(synth_resp)["result"]["content"][0]["text"]
    print("\n--- synthesize result ---", flush=True)
    print(synth_block[:800], flush=True)

    rc = 0
    try:
        payload = json.loads(synth_block)
        wav = payload.get("output_path")
        if wav and Path(wav).exists():
            size = Path(wav).stat().st_size
            print(f"\nVERIFIED: {wav} ({size} bytes)", flush=True)
        else:
            print(f"\n!! synthesize output_path missing or absent: {wav}", flush=True)
            rc = 1
    except Exception as exc:
        print(f"\n!! could not parse synthesize response: {exc}", flush=True)
        rc = 1

    # 5) list_outputs to confirm cache management works
    list_resp = call(4, "tools/call", {"name": "voxcpm_list_outputs", "arguments": {}}, timeout=10)
    print("\n--- list_outputs ---", flush=True)
    print(json.loads(list_resp)["result"]["content"][0]["text"][:400], flush=True)

    proc.stdin.close()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    err = proc.stderr.read().decode("utf-8", errors="replace")
    print("\n=== last 6 lines of stderr ===", flush=True)
    for line in err.splitlines()[-6:]:
        print(line, flush=True)

    return rc


if __name__ == "__main__":
    sys.exit(main())