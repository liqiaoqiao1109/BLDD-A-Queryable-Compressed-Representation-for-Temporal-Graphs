$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$solutionRoot = Split-Path $projectRoot -Parent
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.Component.MSBuild -property installationPath
if (-not $vsRoot) { throw 'Visual Studio with MSBuild was not found.' }
$msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\MSBuild.exe'
$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
& $msbuild (Join-Path $solutionRoot 'BLDD.sln') /p:Configuration=Release /p:Platform=x64 /m /v:minimal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$testDir = Join-Path ([System.IO.Path]::GetTempPath()) ('bldd-next-act-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testDir | Out-Null
$graphDir = Join-Path $testDir 'graph'
New-Item -ItemType Directory -Path $graphDir | Out-Null
$exe = Join-Path $solutionRoot 'x64\Release\BLDD.exe'
& $exe compress-save (Join-Path $PSScriptRoot 'next_activation_uvt.txt') $graphDir UVT
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$objects = Get-ChildItem (Join-Path $projectRoot 'x64\Release\*.obj') |
    Where-Object { $_.Name -notin @('Main.obj', 'Application.obj', 'WikiLinkLoad.obj', 'WikiLinkQuery.obj') } |
    ForEach-Object { '"' + $_.FullName + '"' }
$include = Join-Path $solutionRoot 'CUDD\include'
$library = Join-Path $solutionRoot 'x64\Release\CUDD3.0.0.lib'
$testSource = Join-Path $PSScriptRoot 'next_activation_regression.cpp'
$testObject = Join-Path $testDir 'test.obj'
$testExe = Join-Path $testDir 'test.exe'
$command = 'call "' + $vcvars + '" >nul && cl /nologo /O2 /EHsc /MD /std:c++17 ' +
    '/I"' + $projectRoot + '" /I"' + $include + '" /Fo"' + $testObject + '" "' +
    $testSource + '" ' + ($objects -join ' ') + ' "' + $library +
    '" shell32.lib /Fe"' + $testExe + '" /link /LTCG'
cmd /c $command
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $testExe $graphDir
exit $LASTEXITCODE
