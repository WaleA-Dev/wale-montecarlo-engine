"""
Wale Monte Carlo - Strategy Stress Lab
Desktop entry point: starts the local web server and opens your browser.

Run from source:   python app.py
Frozen (exe):      WaleMonteCarlo.exe        (built via WaleMonteCarlo.spec)
"""

import socket
import sys
import threading
import webbrowser


def find_free_port(preferred: int = 8742) -> int:
    """Use the preferred port if free, otherwise let the OS pick one."""
    for port in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
        except OSError:
            continue
    return 0


def main() -> None:
    print()
    print("  ============================================")
    print("   WALE MONTE CARLO - Strategy Stress Lab")
    print("  ============================================")
    print()
    print("  Loading engine (numpy)...")

    from wale_montecarlo.webapp import create_app

    app = create_app()
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    print(f"  Server running at {url}")
    print("  Opening your browser... (close this window to quit)")
    print()

    threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        # Werkzeug dev server is fine here: single local user, localhost only.
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        # PyInstaller: multiprocessing guard (harmless here, defensive)
        import multiprocessing
        multiprocessing.freeze_support()
    main()
