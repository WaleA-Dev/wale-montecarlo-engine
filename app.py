"""
Wale Monte Carlo - Strategy Stress Lab
Desktop entry point.

Preferred mode: native desktop window (WebView2 via pywebview) - the exe
opens its own application window like any Windows program. If no WebView2
runtime is available, falls back to the default browser.

Run from source:   python app.py [--browser]
Frozen (exe):      WaleMonteCarlo.exe        (built via WaleMonteCarlo.spec)
"""

import os
import socket
import sys
import threading
import time
import urllib.request

APP_TITLE = "Wale Monte Carlo - Strategy Stress Lab"
MIN_SIZE = (1024, 700)
START_SIZE = (1360, 900)


def _fix_std_streams() -> None:
    """In a windowed (no-console) exe, stdout/stderr may be None; give
    logging and print() a safe sink so nothing crashes on write."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def find_free_port(preferred: int = 8742) -> int:
    for port in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
        except OSError:
            continue
    return 0


def start_server(port: int):
    """Run Flask in a daemon thread; return once it answers HTTP."""
    from wale_montecarlo.webapp import create_app

    app = create_app()
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port,
                               debug=False, use_reloader=False, threaded=True),
        daemon=True,
    ).start()

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return url
        except Exception:
            time.sleep(0.15)
    raise RuntimeError("Local server failed to start.")


def run_native_window(url: str) -> bool:
    """Open the app in its own native window. Returns False if unavailable."""
    try:
        import webview
    except Exception:
        return False

    try:
        webview.settings["ALLOW_DOWNLOADS"] = True   # report export
        webview.create_window(
            APP_TITLE,
            url,
            width=START_SIZE[0],
            height=START_SIZE[1],
            min_size=MIN_SIZE,
            background_color="#0d0d0d",
        )
        webview.start()   # blocks until the window is closed
        return True
    except Exception:
        return False


def run_browser_fallback(url: str) -> None:
    import webbrowser
    print(f"  Server running at {url}")
    print("  Opening your browser... (Ctrl+C to quit)")
    webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def main() -> None:
    _fix_std_streams()
    port = find_free_port()
    url = start_server(port)

    force_browser = "--browser" in sys.argv
    if force_browser or not run_native_window(url):
        run_browser_fallback(url)


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        import multiprocessing
        multiprocessing.freeze_support()
    main()
