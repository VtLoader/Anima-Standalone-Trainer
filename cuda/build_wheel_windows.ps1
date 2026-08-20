$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

Set-Location -LiteralPath $ScriptDir

& $PythonBin -m pip install --upgrade pip setuptools wheel ninja
& $PythonBin setup.py bdist_wheel

"Built wheels:"
Get-ChildItem -LiteralPath "dist" -Filter "*.whl" | ForEach-Object { $_.FullName }
