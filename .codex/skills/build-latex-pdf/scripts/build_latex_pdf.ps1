[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$TexFiles,

    [string]$SourceRoot,

    [string]$ProjectRoot,

    [string]$OutputRoot,

    [switch]$KeepAuxFiles
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:GeneratedExtensions = @(
    '.abs',
    '.aux',
    '.bcf',
    '.blg',
    '.fdb_latexmk',
    '.fls',
    '.idx',
    '.ilg',
    '.ind',
    '.lof',
    '.log',
    '.lot',
    '.nav',
    '.out',
    '.run.xml',
    '.snm',
    '.toc',
    '.vrb',
    '.xdv'
)

function Get-CommandSource {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    try {
        return (Get-Command $Name -ErrorAction Stop | Select-Object -First 1).Source
    }
    catch {
        return $null
    }
}

function Normalize-FullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $resolved = [System.IO.Path]::GetFullPath($Path)
    return $resolved.TrimEnd([char[]]@([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar))
}

function Test-PathPrefix {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Prefix
    )

    $normalizedPath = Normalize-FullPath -Path $Path
    $normalizedPrefix = Normalize-FullPath -Path $Prefix

    if ($normalizedPath.Equals($normalizedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    $prefixWithSeparator = $normalizedPrefix + [System.IO.Path]::DirectorySeparatorChar
    return $normalizedPath.StartsWith($prefixWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)
}

function Resolve-TexFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Paths
    )

    $expandedPaths = @()
    foreach ($rawPath in $Paths) {
        foreach ($candidatePath in ($rawPath -split ',')) {
            $trimmedPath = $candidatePath.Trim().Trim('"')
            if (-not [string]::IsNullOrWhiteSpace($trimmedPath)) {
                $expandedPaths += $trimmedPath
            }
        }
    }

    $resolved = @()

    foreach ($path in $expandedPaths) {
        $item = Get-Item -LiteralPath $path -ErrorAction Stop

        if ($item.PSIsContainer) {
            throw "Expected a .tex file but received a directory: $path"
        }

        if ($item.Extension -ine '.tex') {
            throw "Expected a .tex file but received: $path"
        }

        $resolved += $item.FullName
    }

    return $resolved
}

function Get-ProjectRootPath {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$TexPaths,

        [string]$ExplicitProjectRoot
    )

    if ($ExplicitProjectRoot) {
        $item = Get-Item -LiteralPath $ExplicitProjectRoot -ErrorAction Stop
        if (-not $item.PSIsContainer) {
            throw "ProjectRoot must be a directory: $ExplicitProjectRoot"
        }
        return $item.FullName
    }

    $anchorDirectory = Split-Path -Parent $TexPaths[0]
    $gitCommand = Get-CommandSource -Name 'git'

    if ($gitCommand) {
        try {
            $gitRoot = & $gitCommand -C $anchorDirectory rev-parse --show-toplevel 2>$null
            if ($LASTEXITCODE -eq 0 -and $gitRoot) {
                return $gitRoot.Trim()
            }
        }
        catch {
        }
    }

    $cursor = $anchorDirectory
    while (-not [string]::IsNullOrWhiteSpace($cursor)) {
        if (Test-Path -LiteralPath (Join-Path $cursor '.git')) {
            return $cursor
        }

        $parent = Split-Path -Parent $cursor
        if ($parent -eq $cursor) {
            break
        }
        $cursor = $parent
    }

    return (Get-Location).Path
}

function Get-SourceRootPath {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$TexPaths,

        [string]$ExplicitSourceRoot
    )

    if ($ExplicitSourceRoot) {
        $item = Get-Item -LiteralPath $ExplicitSourceRoot -ErrorAction Stop
        if (-not $item.PSIsContainer) {
            throw "SourceRoot must be a directory: $ExplicitSourceRoot"
        }

        foreach ($texPath in $TexPaths) {
            if (-not (Test-PathPrefix -Path $texPath -Prefix $item.FullName)) {
                throw "The TeX file '$texPath' is not inside SourceRoot '$($item.FullName)'."
            }
        }

        return $item.FullName
    }

    $directories = $TexPaths | ForEach-Object { Split-Path -Parent $_ } | Sort-Object -Unique
    if ($directories.Count -ne 1) {
        throw "All TeX files must come from the same source directory unless -SourceRoot is provided."
    }

    return $directories[0]
}

function New-TimestampDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BaseDirectory
    )

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $candidate = Join-Path $BaseDirectory $timestamp
    $suffix = 1

    while (Test-Path -LiteralPath $candidate) {
        $candidate = Join-Path $BaseDirectory ('{0}_{1:d2}' -f $timestamp, $suffix)
        $suffix += 1
    }

    New-Item -ItemType Directory -Force -Path $candidate | Out-Null
    return (Get-Item -LiteralPath $candidate).FullName
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,

        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $normalizedRoot = Normalize-FullPath -Path $Root
    $normalizedPath = Normalize-FullPath -Path $Path

    if (-not (Test-PathPrefix -Path $normalizedPath -Prefix $normalizedRoot)) {
        throw "Path '$Path' is not inside root '$Root'."
    }

    return $normalizedPath.Substring($normalizedRoot.Length).TrimStart([char[]]@([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar))
}

function Should-SkipFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FullPath,

        [Parameter(Mandatory = $true)]
        [string]$SourceRootPath,

        [Parameter(Mandatory = $true)]
        [string]$OutputRootPath
    )

    if (Test-PathPrefix -Path $FullPath -Prefix $OutputRootPath) {
        return $true
    }

    $relative = Get-RelativePath -Root $SourceRootPath -Path $FullPath
    $normalizedRelative = $relative -replace '/', '\'

    if ($normalizedRelative -match '(^|\\)\.git(\\|$)' -or
        $normalizedRelative -match '(^|\\)\.venv(\\|$)' -or
        $normalizedRelative -match '(^|\\)__pycache__(\\|$)') {
        return $true
    }

    if ($normalizedRelative -like 'docs\latex_pdfs*') {
        return $true
    }

    if ($normalizedRelative.EndsWith('.synctex.gz', [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    $extension = [System.IO.Path]::GetExtension($FullPath).ToLowerInvariant()
    return $script:GeneratedExtensions -contains $extension
}

function Stage-SourceTree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRootPath,

        [Parameter(Mandatory = $true)]
        [string]$DestinationRootPath,

        [Parameter(Mandatory = $true)]
        [string]$OutputRootPath
    )

    $files = Get-ChildItem -LiteralPath $SourceRootPath -Recurse -File -Force

    foreach ($file in $files) {
        if (Should-SkipFile -FullPath $file.FullName -SourceRootPath $SourceRootPath -OutputRootPath $OutputRootPath) {
            continue
        }

        $relative = Get-RelativePath -Root $SourceRootPath -Path $file.FullName
        $destination = Join-Path $DestinationRootPath $relative
        $destinationDirectory = Split-Path -Parent $destination

        if ($destinationDirectory) {
            New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
        }

        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }
}

function Install-RepoLocalTectonic {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath
    )

    if ($env:OS -ne 'Windows_NT') {
        throw "No CLI LaTeX engine was found. This script can bootstrap standalone Tectonic automatically only on Windows."
    }

    $toolsDirectory = Join-Path $ProjectRootPath '.tools\tectonic'
    $archivePath = Join-Path $toolsDirectory 'tectonic.zip'

    New-Item -ItemType Directory -Force -Path $toolsDirectory | Out-Null

    $release = Invoke-RestMethod -Headers @{ 'User-Agent' = 'Codex' } -Uri 'https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest'
    $asset = $release.assets | Where-Object { $_.name -like '*x86_64-pc-windows-msvc.zip' } | Select-Object -First 1

    if (-not $asset) {
        throw "Unable to locate a Windows standalone Tectonic release."
    }

    Invoke-WebRequest -Headers @{ 'User-Agent' = 'Codex' } -Uri $asset.browser_download_url -OutFile $archivePath
    Expand-Archive -LiteralPath $archivePath -DestinationPath $toolsDirectory -Force

    $tectonicPath = Join-Path $toolsDirectory 'tectonic.exe'
    if (-not (Test-Path -LiteralPath $tectonicPath)) {
        $tectonicPath = (Get-ChildItem -LiteralPath $toolsDirectory -Recurse -File -Filter 'tectonic.exe' | Select-Object -First 1).FullName
    }

    if (-not $tectonicPath) {
        throw "Tectonic download finished, but tectonic.exe was not found."
    }

    return $tectonicPath
}

function Get-LatexEngine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath
    )

    $repoLocalTectonic = Join-Path $ProjectRootPath '.tools\tectonic\tectonic.exe'
    if (Test-Path -LiteralPath $repoLocalTectonic) {
        return [PSCustomObject]@{
            Name = 'tectonic'
            Path = (Get-Item -LiteralPath $repoLocalTectonic).FullName
        }
    }

    foreach ($candidate in @('tectonic', 'latexmk', 'xelatex', 'lualatex', 'pdflatex')) {
        $path = Get-CommandSource -Name $candidate
        if ($path) {
            return [PSCustomObject]@{
                Name = $candidate
                Path = $path
            }
        }
    }

    $installedTectonic = Install-RepoLocalTectonic -ProjectRootPath $ProjectRootPath
    return [PSCustomObject]@{
        Name = 'tectonic'
        Path = $installedTectonic
    }
}

function Invoke-LatexBuild {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Engine,

        [Parameter(Mandatory = $true)]
        [string]$StageDirectory,

        [Parameter(Mandatory = $true)]
        [string]$RelativeTexPath
    )

    $relativeDirectory = Split-Path -Parent $RelativeTexPath
    if ([string]::IsNullOrWhiteSpace($relativeDirectory)) {
        $workingDirectory = $StageDirectory
    }
    else {
        $workingDirectory = Join-Path $StageDirectory $relativeDirectory
    }

    $texLeaf = Split-Path -Leaf $RelativeTexPath

    Push-Location $workingDirectory
    try {
        switch ($Engine.Name) {
            'tectonic' {
                & $Engine.Path --keep-logs -o $workingDirectory $texLeaf
                break
            }
            'latexmk' {
                & $Engine.Path -pdf -interaction=nonstopmode -halt-on-error "-outdir=$workingDirectory" $texLeaf
                break
            }
            'xelatex' {
                foreach ($pass in 1..2) {
                    & $Engine.Path -interaction=nonstopmode -halt-on-error "-output-directory=$workingDirectory" $texLeaf
                    if ($LASTEXITCODE -ne 0) {
                        break
                    }
                }
                break
            }
            'lualatex' {
                foreach ($pass in 1..2) {
                    & $Engine.Path -interaction=nonstopmode -halt-on-error "-output-directory=$workingDirectory" $texLeaf
                    if ($LASTEXITCODE -ne 0) {
                        break
                    }
                }
                break
            }
            'pdflatex' {
                foreach ($pass in 1..2) {
                    & $Engine.Path -interaction=nonstopmode -halt-on-error "-output-directory=$workingDirectory" $texLeaf
                    if ($LASTEXITCODE -ne 0) {
                        break
                    }
                }
                break
            }
            default {
                throw "Unsupported LaTeX engine: $($Engine.Name)"
            }
        }

        if ($LASTEXITCODE -ne 0) {
            throw "Build failed for '$RelativeTexPath' using '$($Engine.Name)'."
        }
    }
    finally {
        Pop-Location
    }
}

function Remove-AuxiliaryFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StageDirectory
    )

    $extensionsToRemove = @(
        '.abs',
        '.aux',
        '.bcf',
        '.blg',
        '.fdb_latexmk',
        '.fls',
        '.idx',
        '.ilg',
        '.ind',
        '.lof',
        '.lot',
        '.nav',
        '.out',
        '.run.xml',
        '.snm',
        '.toc',
        '.vrb',
        '.xdv'
    )

    $files = Get-ChildItem -LiteralPath $StageDirectory -Recurse -File -Force
    foreach ($file in $files) {
        $remove = $false

        if ($file.Name.EndsWith('.synctex.gz', [System.StringComparison]::OrdinalIgnoreCase)) {
            $remove = $true
        }
        elseif ($extensionsToRemove -contains $file.Extension.ToLowerInvariant()) {
            $remove = $true
        }

        if ($remove) {
            Remove-Item -LiteralPath $file.FullName -Force
        }
    }
}

$resolvedTexFiles = Resolve-TexFiles -Paths $TexFiles
$projectRootPath = Get-ProjectRootPath -TexPaths $resolvedTexFiles -ExplicitProjectRoot $ProjectRoot
$projectRootPath = Normalize-FullPath -Path $projectRootPath

$sourceRootPath = Get-SourceRootPath -TexPaths $resolvedTexFiles -ExplicitSourceRoot $SourceRoot
$sourceRootPath = Normalize-FullPath -Path $sourceRootPath

if ($OutputRoot) {
    $outputRootPath = Get-Item -LiteralPath $OutputRoot -ErrorAction SilentlyContinue
    if ($outputRootPath -and -not $outputRootPath.PSIsContainer) {
        throw "OutputRoot must be a directory: $OutputRoot"
    }

    if (-not $outputRootPath) {
        New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    }

    $outputRootPath = Normalize-FullPath -Path $OutputRoot
}
else {
    $outputRootPath = Join-Path $projectRootPath 'docs\latex_pdfs'
    New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null
    $outputRootPath = Normalize-FullPath -Path $outputRootPath
}

$timestampDirectory = New-TimestampDirectory -BaseDirectory $outputRootPath
Stage-SourceTree -SourceRootPath $sourceRootPath -DestinationRootPath $timestampDirectory -OutputRootPath $outputRootPath

$engine = Get-LatexEngine -ProjectRootPath $projectRootPath
$relativeTexPaths = @()
$generatedPdfPaths = @()

foreach ($texPath in $resolvedTexFiles) {
    $relativeTexPath = Get-RelativePath -Root $sourceRootPath -Path $texPath
    $relativeTexPaths += $relativeTexPath

    $stagedTexPath = Join-Path $timestampDirectory $relativeTexPath
    if (-not (Test-Path -LiteralPath $stagedTexPath)) {
        throw "The staged TeX file was not found: $stagedTexPath"
    }

    Invoke-LatexBuild -Engine $engine -StageDirectory $timestampDirectory -RelativeTexPath $relativeTexPath

    $pdfRelativePath = [System.IO.Path]::ChangeExtension($relativeTexPath, '.pdf')
    $generatedPdfPath = Join-Path $timestampDirectory $pdfRelativePath

    if (-not (Test-Path -LiteralPath $generatedPdfPath)) {
        throw "The generated PDF was not found after building '$relativeTexPath'. Expected: $generatedPdfPath"
    }

    $generatedPdfPaths += (Normalize-FullPath -Path $generatedPdfPath)
}

if (-not $KeepAuxFiles) {
    Remove-AuxiliaryFiles -StageDirectory $timestampDirectory
}

[PSCustomObject]@{
    OutputFolder = $timestampDirectory
    SourceRoot = $sourceRootPath
    Engine = $engine.Name
    TeXFiles = $relativeTexPaths
    PDFs = $generatedPdfPaths
} | ConvertTo-Json -Depth 4
