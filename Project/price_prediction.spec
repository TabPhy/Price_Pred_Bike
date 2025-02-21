# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('xgboost')


a = Analysis(
    ['price_prediction.py'],
    pathex=[],
    binaries=[('/Users/tabea/miniforge3/envs/toolbox/lib/libxgboost.dylib', './xgboost/lib'), ('/Users/tabea/miniforge3/envs/toolbox/lib/libpython3.12.dylib', '.')],
    datas=datas,
    hiddenimports=['joblib', 'pandas', 'scikit-learn', 'sklearn.preprocessing', 'gzip', 'sklearn.utils', 'encodings', 'sklearn.ensemble', 'sklearn.model_selection', 'sklearn.metrics', 'sklearn.pipeline', 'sklearn.impute', 'sklearn.compose', 'sklearn.tree', 'sklearn.neighbors', 'sklearn.feature_selection', 'catboost', 'xgboost', 'pickle'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='price_prediction',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='price_prediction',
)
