$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$kernelTestsRoot = Split-Path -Parent $scriptDir
Set-Location $kernelTestsRoot

$pdflatex = (Get-Command pdflatex -ErrorAction SilentlyContinue).Source
if (-not $pdflatex) {
    $pdflatex = "C:\Users\manue\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
}

Write-Host ""
Write-Host "Kernel report build" -ForegroundColor Cyan
Write-Host "Directorio de trabajo: $kernelTestsRoot"
Write-Host ""

Write-Host "==> Generando artefactos estadisticos" -ForegroundColor Yellow
python scripts/build_kernel_report_assets.py
if ($LASTEXITCODE -ne 0) {
    throw "Fallo la generacion de artefactos del reporte"
}

$outDir = Join-Path $kernelTestsRoot "out"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

Write-Host "==> Compilando PDF (pasada 1)" -ForegroundColor Yellow
& $pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$outDir -jobname=Informe_Kernels_Integrado reports/Informe_Kernels_Integrado.tex
if ($LASTEXITCODE -ne 0) {
    throw "Fallo la compilacion LaTeX en la pasada 1"
}

Write-Host "==> Compilando PDF (pasada 2)" -ForegroundColor Yellow
& $pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$outDir -jobname=Informe_Kernels_Integrado reports/Informe_Kernels_Integrado.tex
if ($LASTEXITCODE -ne 0) {
    throw "Fallo la compilacion LaTeX en la pasada 2"
}

Write-Host ""
Write-Host "PDF actualizado: out/Informe_Kernels_Integrado.pdf" -ForegroundColor Green
