param(
    [Parameter(Position = 0)]
    [ValidateSet("train", "eval", "demo", "web")]
    [string]$Mode = "train",

    [int]$Timesteps = 10000,
    [int]$Episodes = 20,
    [int]$Seed = 0,
    [int]$MaxEpisodes = 0,
    [int]$Fps = 20,
    [int]$MapSize = 64,
    [float]$ObstacleDensity = 0.15,
    [int]$LidarRays = 24,
    [int]$MaxSteps = 300,
    [int]$Port = 8000,
    [string]$Device = "auto",
    [string]$SaveDir = "models",
    [string]$ModelPath = "models\best_model.zip",
    [string]$MapPath = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment not found: $PythonExe"
}

switch ($Mode) {
    "train" {
        & $PythonExe (Join-Path $ProjectRoot "training\train.py") `
            --timesteps $Timesteps `
            --seed $Seed `
            --save_dir (Join-Path $ProjectRoot $SaveDir) `
            --device $Device `
            --map-size $MapSize `
            --obstacle-density $ObstacleDensity `
            --lidar-rays $LidarRays `
            --max-steps $MaxSteps
    }
    "eval" {
        $ResolvedModelPath = Join-Path $ProjectRoot $ModelPath
        $Args = @(
            (Join-Path $ProjectRoot "evaluation\evaluate.py"),
            "--episodes", $Episodes,
            "--seed", $Seed,
            "--model_path", $ResolvedModelPath
        )
        if ($MapPath -ne "") {
            $Args += @("--map_path", (Join-Path $ProjectRoot $MapPath))
        }
        & $PythonExe @Args
    }
    "demo" {
        $ResolvedModelPath = Join-Path $ProjectRoot $ModelPath
        $Args = @(
            (Join-Path $ProjectRoot "demo.py"),
            "--model_path", $ResolvedModelPath,
            "--seed", $Seed,
            "--max_episodes", $MaxEpisodes,
            "--fps", $Fps
        )
        if ($MapPath -ne "") {
            $Args += @("--map_path", (Join-Path $ProjectRoot $MapPath))
        }
        & $PythonExe @Args
    }
    "web" {
        & $PythonExe -m uvicorn web.app:app --host 127.0.0.1 --port $Port --reload
    }
}
