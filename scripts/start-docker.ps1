param(
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [switch]$NoBrowser,
    [switch]$NoToken,
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Install Docker Desktop with the WSL 2 Linux container engine, then run this launcher again.'
}
function Get-DockerReady {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $engine = & docker info --format '{{.OSType}}' 2>$null
    $success = $LASTEXITCODE -eq 0 -and $engine -eq 'linux'
    $ErrorActionPreference = $oldPreference
    return $success
}
if (-not (Get-DockerReady)) {
    $desktopPath = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path -LiteralPath $desktopPath)) { throw 'Start a Docker Linux engine first.' }
    Write-Host 'Starting Docker Desktop...'
    Start-Process -FilePath $desktopPath -WindowStyle Hidden
    $startupDeadline = (Get-Date).AddMinutes(3)
    while (-not (Get-DockerReady)) {
        if ((Get-Date) -gt $startupDeadline) { throw 'Docker did not become ready. Open Docker Desktop and check its WSL 2 status.' }
        Start-Sleep -Seconds 3
    }
}
$env:MODELPORT_PORT = [string]$Port
$composeArguments = @('compose', '-f', 'docker/compose.yml', 'up', '-d')
if ($NoBuild) { $composeArguments += '--no-build' } else { $composeArguments += '--build' }
& docker @composeArguments
if ($LASTEXITCODE -ne 0) { throw "Docker startup failed. If port $Port is occupied, use -Port 8767." }
$url = "http://127.0.0.1:$Port"
$readyDeadline = (Get-Date).AddMinutes(3)
Write-Host 'Waiting for the ModelPort worker...'
while ($true) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$url/health/ready" -TimeoutSec 3
        if ($response.StatusCode -eq 200) { break }
    } catch { }
    if ((Get-Date) -gt $readyDeadline) { throw 'ModelPort readiness timed out. Run: docker compose -f docker/compose.yml logs --tail 100' }
    Start-Sleep -Seconds 2
}
Write-Host "ModelPort is ready: $url"
if (-not $NoToken) {
    Write-Host 'Copy this operator token into the sign-in screen:'
    $tokenResponse = & docker compose -f docker/compose.yml exec -T modelport modelport token
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the operator token.' }
    $credential = $tokenResponse | ConvertFrom-Json
    Write-Host $credential.token
}
if (-not $NoBrowser) { Start-Process $url }
Write-Host 'Use Run CPU demo in the dashboard to test real conversions and benchmarks.'
Write-Host 'Stop with: docker compose -f docker/compose.yml stop'
