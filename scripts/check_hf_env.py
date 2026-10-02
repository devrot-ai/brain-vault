"""Pre-flight check for publishing to Hugging Face Spaces.

Prints exactly one line per layer and never prints the token itself:
  1. python version
  2. huggingface_hub import + version
  3. HF_TOKEN presence (length + prefix shape only)
  4. token validity via whoami() -> prints the HF username

Exit 0 = ready to publish; 2 = fix the printed layer first.
Run from the repo root:  .venv/Scripts/python.exe scripts/check_hf_env.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    print(f"1. python {sys.version.split()[0]}")

    try:
        import huggingface_hub as h
    except ImportError:
        print("2. FAIL: huggingface_hub is not installed in this venv")
        print("   fix:  .venv/Scripts/python.exe -m pip install -r scripts/requirements-hf.txt")
        return 2
    print(f"2. huggingface_hub {h.__version__}")

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not tok.strip():
        print("3. FAIL: HF_TOKEN is not set in this shell")
        print("   fix (Git Bash):  export HF_TOKEN=hf_...    (paste YOUR Write token)")
        print("   fix (cmd.exe):   set HF_TOKEN=hf_...")
        print("   or add --token to the sync command to type it at a hidden prompt")
        return 2
    print(f"3. token present: {len(tok)} chars, starts with 'hf_': {tok.startswith('hf_')}")

    from huggingface_hub import HfApi
    try:
        me = HfApi(token=tok.strip()).whoami()
    except Exception as exc:  # noqa: BLE001 - report the real reason
        print(f"4. FAIL: huggingface.co rejected the token ({type(exc).__name__})")
        print("   create a WRITE token at https://huggingface.co/settings/tokens")
        print(f"   detail: {str(exc)[:300]}")
        return 2
    name = me.get("name", "?")
    print(f"4. token valid — logged in as: {name}")
    print()
    print(f"READY. publish with:")
    print(f"  .venv/Scripts/python.exe scripts/sync_hf_space.py --space {name}/brainvault-api")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
