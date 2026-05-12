$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $root "deliverables"
$packageRoot = Join-Path $outputRoot "ecs-package"

if (Test-Path -LiteralPath $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $packageRoot | Out-Null

function Copy-IfExists {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (Test-Path -LiteralPath $Source) {
        $parent = Split-Path -Parent $Destination
        if ($parent -and -not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
    }
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

# Root files
Copy-IfExists (Join-Path $root "docker-compose.yml") (Join-Path $packageRoot "docker-compose.yml")
Copy-IfExists (Join-Path $root "README.md") (Join-Path $packageRoot "README.md")
Copy-IfExists (Join-Path $root ".env.high-concurrency.example") (Join-Path $packageRoot ".env.high-concurrency.example")

# MSA runtime source
Copy-IfExists (Join-Path $root "MSA\Dockerfile.worker") (Join-Path $packageRoot "MSA\Dockerfile.worker")
Copy-IfExists (Join-Path $root "MSA\requirements-standalone.txt") (Join-Path $packageRoot "MSA\requirements-standalone.txt")
Copy-IfExists (Join-Path $root "MSA\config.py") (Join-Path $packageRoot "MSA\config.py")
Copy-IfExists (Join-Path $root "MSA\msa_service") (Join-Path $packageRoot "MSA\msa_service")
Copy-IfExists (Join-Path $root "MSA\GCNet") (Join-Path $packageRoot "MSA\GCNet")
Copy-IfExists (Join-Path $root "MSA\DT-MSA") (Join-Path $packageRoot "MSA\DT-MSA")

# Runtime assets required on ECS
Copy-IfExists (Join-Path $root "MSA\models") (Join-Path $packageRoot "MSA\models")
Copy-IfExists (Join-Path $root "MSA\tools") (Join-Path $packageRoot "MSA\tools")
Copy-IfExists (Join-Path $root "MSA\dataset") (Join-Path $packageRoot "MSA\dataset")

# Backend source
Copy-IfExists (Join-Path $root "Net\user-center-backend-master") (Join-Path $packageRoot "Net\user-center-backend-master")

# Frontend source, excluding local dependencies and logs
$frontendSrc = Join-Path $root "Net\user-center-frontend-master"
$frontendDst = Join-Path $packageRoot "Net\user-center-frontend-master"
Ensure-Dir $frontendDst

$frontendKeep = @(
    "config",
    "docker",
    "public",
    "src",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "jsconfig.json",
    ".editorconfig",
    ".eslintignore",
    ".eslintrc.js",
    ".gitignore",
    ".prettierignore",
    ".prettierrc.js",
    ".stylelintrc.js",
    "Dockerfile",
    "README.md"
)

foreach ($item in $frontendKeep) {
    Copy-IfExists (Join-Path $frontendSrc $item) (Join-Path $frontendDst $item)
}

# Deployment note
$note = @"
Upload this folder to ECS as the project root.

Required before startup:
1. Confirm MSA/models exists and contains checkpoints.
2. Confirm MSA/tools exists and contains Whisper, transformers, OpenFace, MANet, and wav2vec assets.
3. Open ECS security-group port 5020.
4. Optional: open 15673 for RabbitMQ management.

Start command:
docker compose up -d --build

Public URL:
http://<ECS-IP>:5020
"@
$note | Set-Content -LiteralPath (Join-Path $packageRoot "DEPLOY.txt") -Encoding UTF8

Write-Host "Prepared ECS package at: $packageRoot"
