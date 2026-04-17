param(
    [string]$Python = "",
    [string]$MsaHost = "127.0.0.1",
    [int]$MsaPort = 8000,
    [int]$FrontendPort = 8001,
    [switch]$Restart,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MsaDir = Join-Path $RootDir "MSA"
$BackendDir = Join-Path $RootDir "Net\user-center-backend-master"
$FrontendDir = Join-Path $RootDir "Net\user-center-frontend-master"

function Quote-PS {
    param([string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Assert-Dir {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Name directory not found: $Path"
    }
}

function Assert-File {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Name file not found: $Path"
    }
}

function Get-PortOwners {
    param([int]$Port)

    $owners = @{}
    $matches = netstat -ano | Select-String -Pattern ":$Port\s+.*LISTENING"
    foreach ($match in $matches) {
        $parts = ($match.ToString() -split "\s+") | Where-Object { $_ }
        if ($parts.Count -gt 0) {
            $processId = [int]$parts[-1]
            if (-not $owners.ContainsKey($processId)) {
                $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                $owners[$processId] = [PSCustomObject]@{
                    Port = $Port
                    ProcessId = $processId
                    ProcessName = if ($process) { $process.ProcessName } else { "unknown" }
                }
            }
        }
    }

    return @($owners.Values)
}

function Start-ServiceWindow {
    param(
        [string]$Title,
        [string]$WorkingDirectory,
        [string]$RunCommand
    )

    Write-Host "Starting $Title"
    Write-Host "  Dir: $WorkingDirectory"
    Write-Host "  Cmd: $RunCommand"

    if ($DryRun) {
        return
    }

    $childScript = @"
`$Host.UI.RawUI.WindowTitle = $(Quote-PS $Title)
Set-Location -LiteralPath $(Quote-PS $WorkingDirectory)
Write-Host "[$Title] Working directory: $WorkingDirectory"
Write-Host "[$Title] Command: $RunCommand"
Write-Host ""
try {
    Invoke-Expression $(Quote-PS $RunCommand)
} catch {
    Write-Host ""
    Write-Host "[$Title] Failed: `$(`$_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""
Write-Host "[$Title] Process exited. Close this window when you are done."
"@

    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childScript))
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        $encodedCommand
    )
}

Assert-Dir $MsaDir "MSA"
Assert-Dir $BackendDir "Backend"
Assert-Dir $FrontendDir "Frontend"
Assert-File (Join-Path $BackendDir "mvnw.cmd") "Maven wrapper"
Assert-File (Join-Path $FrontendDir "package.json") "Frontend package.json"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $VenvPython = Join-Path $MsaDir ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        $Python = $VenvPython
    } else {
        $Python = "python"
    }
}

if (Test-Path -LiteralPath $Python -PathType Leaf) {
    $PythonCall = "& " + (Quote-PS $Python)
} else {
    $PythonCall = $Python
}

$MsaCommand = "$PythonCall -m msa_service.controller.http_server --host $MsaHost --port $MsaPort"
$BackendCommand = ".\mvnw.cmd spring-boot:run"
$FrontendCommand = "`$env:PORT='$FrontendPort'; npm.cmd run start:dev"

Write-Host "MSA-Net one-click startup"
Write-Host "Root: $RootDir"
Write-Host "MSA URL: http://$MsaHost`:$MsaPort"
Write-Host "Backend URL: http://127.0.0.1:8080/api"
Write-Host "Frontend URL: http://127.0.0.1:$FrontendPort"
Write-Host ""

$PortOwners = @()
$PortOwners += Get-PortOwners $MsaPort
$PortOwners += Get-PortOwners 8080
$PortOwners += Get-PortOwners $FrontendPort

if ($PortOwners.Count -gt 0) {
    foreach ($owner in $PortOwners) {
        Write-Warning "Port $($owner.Port) is already in use by $($owner.ProcessName) (PID $($owner.ProcessId))."
    }
}
Write-Host ""

if ($PortOwners.Count -gt 0 -and -not $DryRun) {
    if (-not $Restart) {
        $BlockedPorts = $PortOwners | Select-Object -ExpandProperty Port -Unique
        throw "Cannot start because required port(s) are already in use: $($BlockedPorts -join ', '). Close the old service windows or run .\start-all.bat -Restart."
    }

    $ProcessIds = $PortOwners | Select-Object -ExpandProperty ProcessId -Unique
    foreach ($processId in $ProcessIds) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Stopping existing process $($process.ProcessName) (PID $processId)"
            Stop-Process -Id $processId -Force
        }
    }
    Start-Sleep -Seconds 2
}

Start-ServiceWindow "MSA-Service" $MsaDir $MsaCommand
Start-Sleep -Seconds 1
Start-ServiceWindow "MSA-Net-Backend" $BackendDir $BackendCommand
Start-Sleep -Seconds 1
Start-ServiceWindow "MSA-Net-Frontend" $FrontendDir $FrontendCommand

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. No windows were started."
} else {
    Write-Host "All services were launched in separate windows."
}
