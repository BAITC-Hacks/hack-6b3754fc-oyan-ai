[CmdletBinding()]
param(
    [string[]]$EnvironmentVariables = @('OPENAI_API_KEY', 'NVIDIA_API_KEY')
)

$ErrorActionPreference = 'Stop'

$toolSpecs = @(
    @{ Name = 'Git'; Command = 'git'; Args = @('--version'); Required = $true },
    @{ Name = 'Python'; Command = 'python'; Args = @('--version'); Required = $false },
    @{ Name = 'Go'; Command = 'go'; Args = @('version'); Required = $false },
    @{ Name = 'Node.js'; Command = 'node'; Args = @('--version'); Required = $false },
    @{ Name = 'uv'; Command = 'uv'; Args = @('--version'); Required = $false },
    @{ Name = 'GitHub CLI'; Command = 'gh'; Args = @('--version'); Required = $false },
    @{ Name = 'Codex CLI'; Command = 'codex'; Args = @('--version'); Required = $false },
    @{ Name = 'Docker'; Command = 'docker'; Args = @('--version'); Required = $false }
)

$results = foreach ($spec in $toolSpecs) {
    $command = Get-Command $spec.Command -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        [pscustomobject]@{ Tool = $spec.Name; Status = if ($spec.Required) { 'MISSING (required)' } else { 'not found (optional)' }; Version = '' }
        continue
    }

    $version = (& $command.Source @($spec.Args) 2>&1 | Select-Object -First 1).ToString().Trim()
    [pscustomobject]@{ Tool = $spec.Name; Status = 'ready'; Version = $version }
}

$results | Format-Table -AutoSize

$hasPython = ($results | Where-Object { $_.Tool -eq 'Python' }).Status -eq 'ready'
$hasGo = ($results | Where-Object { $_.Tool -eq 'Go' }).Status -eq 'ready'
if (-not ($hasPython -or $hasGo)) {
    Write-Error 'Neither Python nor Go is available on PATH.'
}

$gitNameSet = -not [string]::IsNullOrWhiteSpace((git config --global user.name 2>$null))
$gitEmailSet = -not [string]::IsNullOrWhiteSpace((git config --global user.email 2>$null))
Write-Host "Git identity: name=$(if ($gitNameSet) { 'set' } else { 'MISSING' }), email=$(if ($gitEmailSet) { 'set' } else { 'MISSING' })"

foreach ($variableName in $EnvironmentVariables) {
    $value = [Environment]::GetEnvironmentVariable($variableName)
    $status = if ([string]::IsNullOrWhiteSpace($value)) { 'not present in this shell' } else { 'present (value hidden)' }
    Write-Host "$variableName`: $status"
}

Write-Host 'Preflight complete. Missing optional tools are not blockers unless the selected case needs them.'
