[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRepository,

    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$sourceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$targetRoot = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $TargetRepository).Path)

if (-not (Test-Path -LiteralPath (Join-Path $targetRoot '.git'))) {
    throw "Target is not a Git repository: $targetRoot"
}
if ($targetRoot.Equals($sourceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Target repository must be different from the preparation repository.'
}

$fileMap = [ordered]@{
    'AGENTS.md' = 'AGENTS.md'
    '.gitignore' = '.gitignore'
    'starter\skills\hackalem-case-selection\SKILL.md' = '.agents\skills\hackalem-case-selection\SKILL.md'
    'starter\skills\hackalem-source-gate\SKILL.md' = '.agents\skills\hackalem-source-gate\SKILL.md'
    'starter\skills\hackalem-source-gate\references\research.md' = '.agents\skills\hackalem-source-gate\references\research.md'
    'starter\skills\hackalem-source-gate\references\dependencies.md' = '.agents\skills\hackalem-source-gate\references\dependencies.md'
    'docs\CASE_SELECTION.md' = 'docs\CASE_SELECTION.md'
    'docs\CHALLENGE_BRIEF.md' = 'docs\CHALLENGE_BRIEF.md'
    'docs\HACKATHON_RUNBOOK.md' = 'docs\HACKATHON_RUNBOOK.md'
    'docs\TECH_RADAR.md' = 'docs\TECH_RADAR.md'
    'docs\TEAM_PLAYBOOK.md' = 'docs\TEAM_PLAYBOOK.md'
    'docs\PROGRESS_LOG.md' = 'docs\PROGRESS_LOG.md'
    'templates\README.template.md' = 'templates\README.template.md'
    'scripts\preflight.ps1' = 'scripts\preflight.ps1'
}

foreach ($mapping in $fileMap.GetEnumerator()) {
    $source = Join-Path $sourceRoot $mapping.Key
    $destination = Join-Path $targetRoot $mapping.Value
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Starter file is missing: $source"
    }
    if ((Test-Path -LiteralPath $destination) -and -not $Force) {
        Write-Host "SKIP existing: $($mapping.Value)"
        continue
    }

    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force:$Force
    Write-Host "COPY: $($mapping.Value)"
}

Write-Host ''
Write-Host 'Starter infrastructure installed. Review every file, then commit it in the organizer repository after the official start.'
& git -C $targetRoot status --short
