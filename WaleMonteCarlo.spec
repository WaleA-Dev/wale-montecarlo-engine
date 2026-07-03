# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for WaleMonteCarlo.exe (windowed native-desktop app)
# Build:  pyinstaller WaleMonteCarlo.spec --noconfirm

from PyInstaller.utils.hooks import collect_all

# pywebview needs its packaged WebView2 loader DLLs and pythonnet/clr backend
wv_datas, wv_binaries, wv_hidden = collect_all('webview')
clr_datas, clr_binaries, clr_hidden = collect_all('clr_loader')
net_datas, net_binaries, net_hidden = collect_all('pythonnet')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=wv_binaries + clr_binaries + net_binaries,
    datas=[
        ('wale_montecarlo/webapp/templates', 'wale_montecarlo/webapp/templates'),
        ('wale_montecarlo/webapp/static', 'wale_montecarlo/webapp/static'),
    ] + wv_datas + clr_datas + net_datas,
    hiddenimports=wv_hidden + clr_hidden + net_hidden + ['clr'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy libs present in the dev environment but unused by the app
        'matplotlib', 'pandas', 'scipy', 'PIL', 'tkinter', '_tkinter',
        'IPython', 'jedi', 'pytest', 'notebook', 'PySide6', 'PyQt5',
        'databento', 'yaml', 'tqdm', 'setuptools', 'numpy.f2py',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WaleMonteCarlo',
    icon='assets/icon.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,         # windowed: no console; the native window IS the app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
