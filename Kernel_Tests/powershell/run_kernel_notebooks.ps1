param(
    [string[]]$Notebooks
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$kernelTestsRoot = Split-Path -Parent $scriptDir
Set-Location $kernelTestsRoot

$allNotebooks = @(
    "notebooks/01_kde_kernels_comparativa.ipynb",
    "notebooks/02_kde_gridsize_convergencia.ipynb"
)

$notebooks = if ($Notebooks -and $Notebooks.Count -gt 0) {
    $Notebooks
} else {
    $allNotebooks
}

Write-Host ""
Write-Host "Kernel_Tests pipeline" -ForegroundColor Cyan
Write-Host "Directorio de trabajo: $kernelTestsRoot"
Write-Host ""

foreach ($notebook in $notebooks) {
    Write-Host "==> Ejecutando $notebook" -ForegroundColor Yellow
    python -m nbconvert `
        --to notebook `
        --execute `
        --inplace `
        --ExecutePreprocessor.timeout=-1 `
        --ExecutePreprocessor.kernel_name=python3 `
        $notebook

    if ($LASTEXITCODE -ne 0) {
        throw "Fallo la ejecucion de $notebook"
    }

    Write-Host "<== Terminado $notebook" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Todos los notebooks de Kernel_Tests terminaron correctamente." -ForegroundColor Green
Write-Host "Los resultados quedaron guardados en los mismos .ipynb." -ForegroundColor Green
