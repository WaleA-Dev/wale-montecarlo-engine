# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for WaleMonteCarlo.exe
# Build:  pyinstaller WaleMonteCarlo.spec --noconfirm

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('wale_montecarlo/webapp/templates', 'wale_montecarlo/webapp/templates'),
        ('wale_montecarlo/webapp/static', 'wale_montecarlo/webapp/static'),
    ],
    hiddenimports=[],
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
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,          # console shows server status; closing it quits the app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
