# PYTHON 3.13 INSTALLATION FIX - Windows Path Length Issue

## Problem Identified

Your machine has Windows AppData path length limitations that prevent pip from installing packages using the default Python distribution from the Microsoft Store.

**Error**: Windows Long Path support is not enabled

**Root Cause**: 
- Using Python from WindowsApps (Microsoft Store version)
- Path: `C:\Users\...AppData\Local\Packages\PythonSoftwareFoundation...` is very long
- This exceeds Windows' traditional 260-character path limit

## Solutions

### SOLUTION 1: Install Official Python (RECOMMENDED)

1. **Remove Microsoft Store Python**:
   - Go to Settings → Apps → Apps & features
   - Find "Python 3.13" 
   - Uninstall it

2. **Download Official Python**:
   - Go to https://www.python.org/downloads/
   - Download Python 3.13 (latest stable)
   - Run installer
   - **IMPORTANT**: Check "Add Python to PATH" during installation

3. **Create venv with official Python**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

### SOLUTION 2: Enable Windows Long Paths

1. **Open Group Policy Editor**:
   - Press `Win + R`, type `gpe dit.msc`, press Enter

2. **Navigate to**:
   - Computer Configuration → Administrative Templates → System → Filesystem

3. **Find "Enable Win32 long paths"**:
   - Double-click it
   - Select "Enabled"
   - Click OK

4. **Restart your computer**

5. **Try installation again**:
   ```bash
   pip install -r requirements.txt
   ```

### SOLUTION 3: Use Conda (Alternative)

If you have Anaconda/Miniconda installed:

```bash
conda create -n rag python=3.13
conda activate rag
pip install -r requirements.txt
```

### SOLUTION 4: Manually Install Key Packages

If other solutions don't work, install one-by-one:

```bash
pip install --user --no-deps langchain
pip install --user --no-deps chromadb
pip install --user --no-deps streamlit
pip install --user --no-deps sentence-transformers
pip install --user --no-deps ollama
pip install --user --no-deps loguru
pip install --user rank-bm25
pip install --user pydantic
```

## Recommended: Use Official Python

The easiest and cleanest solution is to use Python from python.org instead of the Microsoft Store version.

### Steps:
1. Uninstall Python 3.13 from Microsoft Store
2. Download Python 3.13 from https://www.python.org
3. Run the installer (check "Add to PATH")
4. Restart terminal
5. Run: `pip install -r requirements.txt`

This will work without any issues!

## After Installing

Once installed correctly, verify with:

```bash
python --version
pip --version
python -c "import langchain; import streamlit; print('Success!')"
```

## Troubleshooting

If you still get path errors:

1. **Delete user packages cache**:
   ```bash
   rmdir /s "%APPDATA%\pip"
   ```

2. **Clear pip cache**:
   ```bash
   pip cache purge
   ```

3. **Try installing again**:
   ```bash
   pip install -r requirements.txt
   ```

## Questions?

The recommended approach is definitely **SOLUTION 1** - using official Python from python.org instead of Microsoft Store version. It will save you many headaches!

---

**Status**: Solution provided
**Recommended**: Install official Python from python.org
