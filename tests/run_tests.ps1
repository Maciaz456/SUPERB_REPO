param (
    [Parameter(Mandatory=$true)]
    [string]$pythonPath,

    [switch]$generateHtml = $false,
    [string]$outputFolder = $null
)

try {
    & $pythonPath --version > $null
} catch {
    throw "Unable to run Python at: $pythonPath"
}

$coverageSubCommands = @(
    @('run', '--branch', '-m', 'pytest', $PSScriptRoot),
    @('report', '--omit', 'test*', '--show-missing')
)
if ($generateHtml) {
    if (-not $outputFolder) {
        throw "You must specify an output folder if you want to generate an HTML report."
    }
    if (Test-Path $outputFolder) {
        Remove-Item -Force -Recurse -Path $outputFolder
    }
    $coverageSubCommands += ,@('html', '-d', $outputFolder)
}

foreach ($subcommand in $coverageSubCommands) {
    Write-Output "Running the following coverage subcommand: $($subcommand -join ' ')"
    & $pythonPath -m coverage @subcommand
    Write-Output ""
}
