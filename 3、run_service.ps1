# Activate python venv
Set-Location $PSScriptRoot

if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = $PSScriptRoot + "\huggingface"
$Env:TORCH_HOME = $PSScriptRoot + "\torch"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:CUDA_HOME = "${env:CUDA_PATH}"

# More specific espeak setup
$EspeakPath = "C:\Program Files\eSpeak NG"
$Env:PHONEMIZER_ESPEAK_PATH = "$EspeakPath\espeak-ng.exe"
$Env:PHONEMIZER_ESPEAK_LIBRARY = "$EspeakPath\libespeak-ng.dll"
$Env:PATH = "$EspeakPath;" + $Env:PATH

# Add additional debugging
$Env:PHONEMIZER_DEBUG = "1"
$Env:PHONEMIZER_LOGGER_LEVEL = "DEBUG"

# Make Python warnings more visible
$Env:PYTHONWARNINGS = "default"

$Env:PYTHONPATH = $PSScriptRoot

$Env:FLASK_ENV = "development"

# Removed HuggingFace token since we're using local files

python service.py

Read-Host | Out-Null ;
