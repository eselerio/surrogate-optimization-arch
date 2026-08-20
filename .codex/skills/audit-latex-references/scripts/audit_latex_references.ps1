[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TexFile,

    [string]$SourceRoot,

    [string]$ProjectRoot,

    [switch]$NoEdit,

    [switch]$RemoveUnverified,

    [switch]$BuildPdf,

    [int]$MinCrossrefScore = 130,

    [int]$Rows = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-AbsolutePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    return $item.FullName
}

function Get-ProjectRootPath {
    param([string]$ExplicitProjectRoot, [string]$AnchorPath)

    if ($ExplicitProjectRoot) {
        return ConvertTo-AbsolutePath -Path $ExplicitProjectRoot
    }

    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        try {
            $root = & $git -C $AnchorPath rev-parse --show-toplevel 2>$null
            if ($LASTEXITCODE -eq 0 -and $root) {
                return $root.Trim()
            }
        }
        catch {
        }
    }

    $cursor = $AnchorPath
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

function Convert-LatexToPlainText {
    param([AllowNull()][string]$Text)

    if (-not $Text) {
        return ''
    }

    $plain = $Text
    $plain = $plain -replace '\\&', '&'
    $plain = $plain -replace '\\_', '_'
    $plain = $plain -replace '--', '-'
    $plain = [regex]::Replace($plain, '\\["''`^~=.cHuv]\{?([A-Za-z])\}?', '$1')

    do {
        $previous = $plain
        $plain = [regex]::Replace($plain, '\\[A-Za-z]+\{([^{}]*)\}', '$1')
    } while ($plain -ne $previous)

    $plain = [regex]::Replace($plain, '\\[A-Za-z]+\*?', ' ')
    $plain = $plain -replace '[{}]', ' '
    $plain = [regex]::Replace($plain, '\s+', ' ').Trim()
    return $plain
}

function Remove-Diacritics {
    param([AllowNull()][string]$Text)

    if (-not $Text) {
        return ''
    }

    $normalized = $Text.Normalize([System.Text.NormalizationForm]::FormD)
    $builder = New-Object System.Text.StringBuilder
    foreach ($char in $normalized.ToCharArray()) {
        $category = [System.Globalization.CharUnicodeInfo]::GetUnicodeCategory($char)
        if ($category -ne [System.Globalization.UnicodeCategory]::NonSpacingMark) {
            [void]$builder.Append($char)
        }
    }

    return $builder.ToString().Normalize([System.Text.NormalizationForm]::FormC)
}

function Normalize-Text {
    param([AllowNull()][string]$Text)

    $plain = Convert-LatexToPlainText -Text $Text
    $plain = Remove-Diacritics -Text $plain
    $plain = $plain.ToLowerInvariant()
    $plain = [regex]::Replace($plain, '[^a-z0-9]+', ' ')
    return [regex]::Replace($plain, '\s+', ' ').Trim()
}

function Normalize-Surname {
    param([AllowNull()][string]$Surname)

    $value = Normalize-Text -Text $Surname
    return ($value -replace '\s+', '')
}

function Get-YearBase {
    param([AllowNull()][string]$Year)

    if ($Year -match '^(?<year>\d{4})') {
        return $Matches['year']
    }

    return $Year
}

function Get-JsonProperty {
    param(
        [AllowNull()]$Object,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if (-not $Object) {
        return $null
    }

    $property = $Object.PSObject.Properties[$Name]
    if ($property) {
        return $property.Value
    }

    return $null
}

function Get-TokenOverlap {
    param(
        [AllowNull()][string]$Source,
        [AllowNull()][string]$Target
    )

    $sourceTokens = @((Normalize-Text -Text $Source) -split ' ' | Where-Object { $_.Length -gt 2 } | Select-Object -Unique)
    if ($sourceTokens.Count -eq 0) {
        return 0.0
    }

    $targetTokens = @{}
    foreach ($token in @((Normalize-Text -Text $Target) -split ' ' | Where-Object { $_.Length -gt 2 })) {
        $targetTokens[$token] = $true
    }

    $hits = 0
    foreach ($token in $sourceTokens) {
        if ($targetTokens.ContainsKey($token)) {
            $hits += 1
        }
    }

    return [double]$hits / [double]$sourceTokens.Count
}

function Get-ReferenceSection {
    param([Parameter(Mandatory = $true)][string]$Text)

    $pattern = '(?ms)^(?<before>.*?^\\section\*\{References\}\s*\r?\n)(?<references>.*?)(?<after>^\\end\{document\}.*)$'
    $match = [regex]::Match($Text, $pattern)
    if (-not $match.Success) {
        throw "Could not find a manual '\section*{References}' block followed by '\end{document}'."
    }

    return [pscustomobject]@{
        Before = $match.Groups['before'].Value
        References = $match.Groups['references'].Value
        After = $match.Groups['after'].Value
    }
}

function Split-ReferenceEntries {
    param([Parameter(Mandatory = $true)][string]$ReferencesBlock)

    $entries = New-Object System.Collections.Generic.List[string]
    $matches = [regex]::Matches($ReferencesBlock, '(?ms)\\noindent\s+.*?(?=(?:\r?\n\s*)*\\noindent\s+|\z)')
    foreach ($match in $matches) {
        $entry = $match.Value.Trim()
        if (-not [string]::IsNullOrWhiteSpace($entry)) {
            $entries.Add($entry)
        }
    }

    if ($entries.Count -eq 0) {
        throw "No reference entries beginning with '\noindent' were found in the References section."
    }

    return @($entries)
}

function Get-ReferenceMetadata {
    param(
        [Parameter(Mandatory = $true)][string]$Entry,
        [Parameter(Mandatory = $true)][int]$Index
    )

    $plain = Convert-LatexToPlainText -Text $Entry
    $plain = [regex]::Replace($plain, '^\s*noindent\s+', '').Trim()
    $match = [regex]::Match($plain, '^(?<authors>.+?)\s+\((?<year>\d{4}[a-z]?)\)\.\s+(?<rest>.+)$')

    if (-not $match.Success) {
        return [pscustomobject]@{
            Index = $Index
            Parsed = $false
            Entry = $Entry
            Authors = ''
            FirstAuthor = ''
            FirstSurname = ''
            Year = ''
            YearBase = ''
            Title = ''
            Container = ''
            ExistingDoi = ''
            ApaWarnings = @('Could not parse author-year-title structure.')
        }
    }

    $authors = $match.Groups['authors'].Value.Trim()
    $year = $match.Groups['year'].Value.Trim()
    $rest = $match.Groups['rest'].Value.Trim()
    $firstAuthor = (($authors -split ',')[0]).Trim()

    $title = ''
    $titleMatch = [regex]::Match($Entry, '\(\d{4}[a-z]?\)\.\s+(?<title>.*?)\.\s+(?:In\s+)?\\textit', [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if ($titleMatch.Success) {
        $title = Convert-LatexToPlainText -Text $titleMatch.Groups['title'].Value
    }
    else {
        $fallbackTitle = [regex]::Match($rest, '^(?<title>.+?)\.\s+')
        if ($fallbackTitle.Success) {
            $title = $fallbackTitle.Groups['title'].Value.Trim()
        }
    }

    $container = ''
    $containerMatch = [regex]::Match($Entry, '(?:In\s+)?\\textit\{(?<container>[^}]*)\}')
    if ($containerMatch.Success) {
        $container = Convert-LatexToPlainText -Text $containerMatch.Groups['container'].Value
    }

    $existingDoi = ''
    $doiMatch = [regex]::Match($Entry, '(?i)\bdoi:\s*(?<doi>10\.\S+?)(?:\s*\.?\s*)$')
    if ($doiMatch.Success) {
        $existingDoi = ($doiMatch.Groups['doi'].Value -replace '\\_', '_').TrimEnd('.')
    }

    $warnings = New-Object System.Collections.Generic.List[string]
    if (-not $Entry.TrimStart().StartsWith('\noindent ')) {
        $warnings.Add("Reference entry should begin with '\noindent'.")
    }
    if ([string]::IsNullOrWhiteSpace($title)) {
        $warnings.Add('Could not parse reference title.')
    }
    if ([string]::IsNullOrWhiteSpace($container)) {
        $warnings.Add('Could not find italicized journal, proceedings, book, or preprint container.')
    }
    if (-not $Entry.TrimEnd().EndsWith('.')) {
        $warnings.Add('Reference entry should end with a period.')
    }

    return [pscustomobject]@{
        Index = $Index
        Parsed = $true
        Entry = $Entry
        Authors = $authors
        FirstAuthor = $firstAuthor
        FirstSurname = $firstAuthor
        Year = $year
        YearBase = Get-YearBase -Year $year
        Title = $title
        Container = $container
        ExistingDoi = $existingDoi
        ApaWarnings = @($warnings)
    }
}

function Get-BodyCitations {
    param([Parameter(Mandatory = $true)][string]$BodyText)

    $body = Convert-LatexToPlainText -Text $BodyText
    $citations = New-Object System.Collections.Generic.List[object]

    function Add-Citation {
        param([string]$Surname, [string]$Year, [string]$Source)

        if ([string]::IsNullOrWhiteSpace($Surname) -or [string]::IsNullOrWhiteSpace($Year)) {
            return
        }

        $citations.Add([pscustomobject]@{
            Surname = $Surname.Trim()
            NormalizedSurname = Normalize-Surname -Surname $Surname
            Year = $Year.Trim()
            YearBase = Get-YearBase -Year $Year
            Source = $Source.Trim()
        })
    }

    foreach ($match in [regex]::Matches($body, '\((?<inside>[^()]*\d{4}[a-z]?[^()]*)\)')) {
        $inside = $match.Groups['inside'].Value
        foreach ($chunk in ($inside -split ';')) {
            $authorMatch = [regex]::Match($chunk, '(?<surname>[A-Z][A-Za-z''.-]+)')
            if (-not $authorMatch.Success) {
                continue
            }

            $surname = $authorMatch.Groups['surname'].Value
            foreach ($yearMatch in [regex]::Matches($chunk, '(?<!\d)(?<year>\d{4}[a-z]?)(?!\d)')) {
                Add-Citation -Surname $surname -Year $yearMatch.Groups['year'].Value -Source $chunk
            }
        }
    }

    foreach ($match in [regex]::Matches($body, '(?<surname>[A-Z][A-Za-z''.-]+)(?:\s+et\s+al\.|\s+and\s+[A-Z][A-Za-z''.-]+|\s+&\s+[A-Z][A-Za-z''.-]+)?\s*\((?<years>\d{4}[a-z]?(?:\s*,\s*\d{4}[a-z]?)*)\)')) {
        $surname = $match.Groups['surname'].Value
        foreach ($yearMatch in [regex]::Matches($match.Groups['years'].Value, '(?<!\d)(?<year>\d{4}[a-z]?)(?!\d)')) {
            Add-Citation -Surname $surname -Year $yearMatch.Groups['year'].Value -Source $match.Value
        }
    }

    $unique = @{}
    $deduped = New-Object System.Collections.Generic.List[object]
    foreach ($citation in $citations) {
        $key = '{0}|{1}' -f $citation.NormalizedSurname, $citation.YearBase
        if (-not $unique.ContainsKey($key)) {
            $unique[$key] = $true
            $deduped.Add($citation)
        }
    }

    return @($deduped | Sort-Object NormalizedSurname, YearBase)
}

function Get-CrossrefMatch {
    param(
        [Parameter(Mandatory = $true)]$Metadata,
        [Parameter(Mandatory = $true)][int]$Rows,
        [Parameter(Mandatory = $true)][int]$MinimumScore
    )

    if (-not $Metadata.Parsed -or [string]::IsNullOrWhiteSpace($Metadata.Title)) {
        return [pscustomobject]@{
            Reliable = $false
            Score = 0
            DOI = ''
            MatchedTitle = ''
            MatchedContainer = ''
            MatchedYear = ''
            MatchedFirstAuthor = ''
            Note = 'No parseable title for Crossref query.'
        }
    }

    $query = '{0} {1} {2} {3}' -f $Metadata.Title, $Metadata.Container, $Metadata.FirstAuthor, $Metadata.YearBase
    $uri = 'https://api.crossref.org/works?query.bibliographic={0}&rows={1}' -f [uri]::EscapeDataString($query), $Rows

    try {
        Start-Sleep -Milliseconds 120
        $response = Invoke-RestMethod -Uri $uri -Method Get -Headers @{
            'User-Agent' = 'codex-latex-reference-audit/1.0 (mailto:metadata@example.invalid)'
        }
    }
    catch {
        return [pscustomobject]@{
            Reliable = $false
            Score = 0
            DOI = ''
            MatchedTitle = ''
            MatchedContainer = ''
            MatchedYear = ''
            MatchedFirstAuthor = ''
            Note = "Crossref query failed: $($_.Exception.Message)"
        }
    }

    $items = @()
    $message = Get-JsonProperty -Object $response -Name 'message'
    if ($message) {
        $itemsValue = Get-JsonProperty -Object $message -Name 'items'
        if ($itemsValue) {
            $items = @($itemsValue)
        }
    }

    $best = $null
    $bestScore = -1.0
    $targetTitle = Normalize-Text -Text $Metadata.Title
    $targetAuthor = Normalize-Surname -Surname $Metadata.FirstSurname

    foreach ($item in $items) {
        $titleValue = Get-JsonProperty -Object $item -Name 'title'
        $candidateTitle = ''
        if ($titleValue) {
            $candidateTitle = [string]@($titleValue)[0]
        }

        $containerValue = Get-JsonProperty -Object $item -Name 'container-title'
        $candidateContainer = ''
        if ($containerValue) {
            $candidateContainer = [string]@($containerValue)[0]
        }

        $candidateYear = ''
        $issued = Get-JsonProperty -Object $item -Name 'issued'
        $dateParts = Get-JsonProperty -Object $issued -Name 'date-parts'
        if ($dateParts) {
            $firstDatePart = @($dateParts)[0]
            if ($firstDatePart) {
                $candidateYear = [string]@($firstDatePart)[0]
            }
        }

        $candidateAuthor = ''
        $authors = Get-JsonProperty -Object $item -Name 'author'
        if ($authors) {
            $first = @($authors)[0]
            $family = Get-JsonProperty -Object $first -Name 'family'
            if ($family) {
                $candidateAuthor = [string]$family
            }
        }

        $score = 100.0 * (Get-TokenOverlap -Source $Metadata.Title -Target $candidateTitle)
        if ((Normalize-Text -Text $candidateTitle) -eq $targetTitle) {
            $score += 30.0
        }

        $containerOverlap = Get-TokenOverlap -Source $Metadata.Container -Target $candidateContainer
        $score += 35.0 * $containerOverlap
        if (-not [string]::IsNullOrWhiteSpace($Metadata.Container)) {
            if ([string]::IsNullOrWhiteSpace($candidateContainer)) {
                $score -= 40.0
            }
            elseif ($containerOverlap -eq 0) {
                $score -= 15.0
            }
        }

        if ($candidateYear) {
            if ($candidateYear -eq $Metadata.YearBase) {
                $score += 20.0
            }
            elseif ($candidateYear -match '^\d{4}$' -and $Metadata.YearBase -match '^\d{4}$') {
                if ([math]::Abs([int]$candidateYear - [int]$Metadata.YearBase) -le 1) {
                    $score += 8.0
                }
                else {
                    $score -= 8.0
                }
            }
        }

        if ($candidateAuthor -and (Normalize-Surname -Surname $candidateAuthor) -eq $targetAuthor) {
            $score += 15.0
        }

        $doi = Get-JsonProperty -Object $item -Name 'DOI'
        if ($doi) {
            $score += 5.0
        }

        $candidateType = Get-JsonProperty -Object $item -Name 'type'
        $referenceLooksPreprint = $Metadata.Container -match '(?i)arxiv|preprint|ssrn|chemrxiv'
        $candidateLooksPreprint = ([string]$candidateType -match '(?i)posted-content|preprint') -or ([string]$doi -match '(?i)chemrxiv|ssrn|preprints')
        if (-not $referenceLooksPreprint -and $candidateLooksPreprint) {
            $score -= 35.0
        }

        if ($score -gt $bestScore) {
            $bestDoi = ''
            if ($doi) {
                $bestDoi = [string]$doi
            }

            $bestScore = $score
            $best = [pscustomobject]@{
                DOI = $bestDoi
                MatchedTitle = $candidateTitle
                MatchedContainer = $candidateContainer
                MatchedYear = $candidateYear
                MatchedFirstAuthor = $candidateAuthor
            }
        }
    }

    if (-not $best) {
        return [pscustomobject]@{
            Reliable = $false
            Score = 0
            DOI = ''
            MatchedTitle = ''
            MatchedContainer = ''
            MatchedYear = ''
            MatchedFirstAuthor = ''
            Note = 'No Crossref candidates returned.'
        }
    }

    $note = 'Low-confidence Crossref match.'
    if ($bestScore -ge $MinimumScore) {
        $note = 'Reliable Crossref match.'
    }

    return [pscustomobject]@{
        Reliable = ($bestScore -ge $MinimumScore)
        Score = [math]::Round($bestScore, 2)
        DOI = $best.DOI
        MatchedTitle = $best.MatchedTitle
        MatchedContainer = $best.MatchedContainer
        MatchedYear = $best.MatchedYear
        MatchedFirstAuthor = $best.MatchedFirstAuthor
        Note = $note
    }
}

function Format-DoiForLatex {
    param([AllowNull()][string]$Doi)

    if (-not $Doi) {
        return ''
    }

    $value = $Doi.Trim()
    $value = $value -replace '_', '\_'
    $value = $value -replace '%', '\%'
    $value = $value -replace '#', '\#'
    $value = $value -replace '&', '\&'
    return $value
}

function Set-EntryDoi {
    param(
        [Parameter(Mandatory = $true)][string]$Entry,
        [Parameter(Mandatory = $true)][string]$Doi
    )

    $doiText = Format-DoiForLatex -Doi $Doi
    $trimmed = $Entry.Trim()
    if ($trimmed -match '(?i)\bdoi:\s*10\.\S+\.?$') {
        return [regex]::Replace($trimmed, '(?i)\bdoi:\s*10\.\S+\.?$', "doi: $doiText.")
    }

    if ($trimmed.EndsWith('.')) {
        return "$trimmed doi: $doiText."
    }

    return "$trimmed. doi: $doiText."
}

function Get-CitationKey {
    param([string]$Surname, [string]$Year)

    return '{0}|{1}' -f (Normalize-Surname -Surname $Surname), (Get-YearBase -Year $Year)
}

$texPath = ConvertTo-AbsolutePath -Path $TexFile
$texDirectory = Split-Path -Parent $texPath
$sourceRootPath = if ($SourceRoot) { ConvertTo-AbsolutePath -Path $SourceRoot } else { $texDirectory }
$projectRootPath = Get-ProjectRootPath -ExplicitProjectRoot $ProjectRoot -AnchorPath $texDirectory
$text = [System.IO.File]::ReadAllText($texPath)
$newline = if ($text -match "`r`n") { "`r`n" } else { "`n" }
$section = Get-ReferenceSection -Text $text
$entries = Split-ReferenceEntries -ReferencesBlock $section.References
$citations = Get-BodyCitations -BodyText $section.Before

$citationKeys = @{}
foreach ($citation in $citations) {
    $citationKeys[(Get-CitationKey -Surname $citation.Surname -Year $citation.YearBase)] = $true
}

$references = New-Object System.Collections.Generic.List[object]
for ($i = 0; $i -lt $entries.Count; $i++) {
    $metadata = Get-ReferenceMetadata -Entry $entries[$i] -Index ($i + 1)
    $key = Get-CitationKey -Surname $metadata.FirstSurname -Year $metadata.YearBase
    $isCited = $metadata.Parsed -and $citationKeys.ContainsKey($key)
    $match = Get-CrossrefMatch -Metadata $metadata -Rows $Rows -MinimumScore $MinCrossrefScore

    $references.Add([pscustomobject]@{
        Index = $metadata.Index
        Entry = $metadata.Entry
        Metadata = $metadata
        IsCited = $isCited
        Crossref = $match
    })
}

$referenceKeys = @{}
foreach ($reference in $references) {
    if ($reference.Metadata.Parsed) {
        $referenceKeys[(Get-CitationKey -Surname $reference.Metadata.FirstSurname -Year $reference.Metadata.YearBase)] = $true
    }
}

$missingBodyCitations = New-Object System.Collections.Generic.List[object]
foreach ($citation in $citations) {
    $key = Get-CitationKey -Surname $citation.Surname -Year $citation.YearBase
    if (-not $referenceKeys.ContainsKey($key)) {
        $missingBodyCitations.Add($citation)
    }
}

$keptEntries = New-Object System.Collections.Generic.List[string]
$removedUncited = New-Object System.Collections.Generic.List[object]
$removedUnverified = New-Object System.Collections.Generic.List[object]
$doiActions = New-Object System.Collections.Generic.List[object]
$unverifiedKept = New-Object System.Collections.Generic.List[object]
$apaWarnings = New-Object System.Collections.Generic.List[object]

foreach ($reference in $references) {
    $metadata = $reference.Metadata
    $crossref = $reference.Crossref

    foreach ($warning in @($metadata.ApaWarnings)) {
        $apaWarnings.Add([pscustomobject]@{
            Reference = ('{0} ({1})' -f $metadata.FirstAuthor, $metadata.Year)
            Warning = $warning
        })
    }

    if (-not $reference.IsCited) {
        $removedUncited.Add([pscustomobject]@{
            Reference = ('{0} ({1})' -f $metadata.FirstAuthor, $metadata.Year)
            Title = $metadata.Title
        })
        if (-not $NoEdit) {
            continue
        }
    }

    if ($reference.IsCited -and -not $crossref.Reliable) {
        $unverifiedKept.Add([pscustomobject]@{
            Reference = ('{0} ({1})' -f $metadata.FirstAuthor, $metadata.Year)
            Title = $metadata.Title
            Score = $crossref.Score
            Note = $crossref.Note
            BestTitle = $crossref.MatchedTitle
            BestDoi = $crossref.DOI
        })

        if ($RemoveUnverified -and -not $NoEdit) {
            $removedUnverified.Add([pscustomobject]@{
                Reference = ('{0} ({1})' -f $metadata.FirstAuthor, $metadata.Year)
                Title = $metadata.Title
                Score = $crossref.Score
                Note = $crossref.Note
            })
            continue
        }
    }

    $entryToKeep = $reference.Entry
    if ($crossref.Reliable -and $crossref.DOI) {
        $existing = $metadata.ExistingDoi
        if (-not $existing -or ((Normalize-Text -Text $existing) -ne (Normalize-Text -Text $crossref.DOI))) {
            $action = if ($existing) { 'updated' } else { 'added' }
            $reportedAction = $action
            if ($NoEdit) {
                $reportedAction = "would-$action"
            }

            $doiActions.Add([pscustomobject]@{
                Reference = ('{0} ({1})' -f $metadata.FirstAuthor, $metadata.Year)
                Action = $reportedAction
                DOI = $crossref.DOI
                Score = $crossref.Score
            })

            if (-not $NoEdit) {
                $entryToKeep = Set-EntryDoi -Entry $entryToKeep -Doi $crossref.DOI
            }
        }
    }

    $keptEntries.Add($entryToKeep)
}

if (-not $NoEdit) {
    $newReferencesBlock = (@($keptEntries) -join ($newline + $newline)) + $newline + $newline
    $newText = $section.Before + $newReferencesBlock + $section.After
    [System.IO.File]::WriteAllText($texPath, $newText)
}

$build = $null
if ($BuildPdf) {
    $builder = Join-Path $projectRootPath '.codex\skills\build-latex-pdf\scripts\build_latex_pdf.ps1'
    if (-not (Test-Path -LiteralPath $builder)) {
        throw "Could not find PDF build script at: $builder"
    }

    $buildOutput = @()
    $buildExitCode = 0
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $buildOutput = & powershell -ExecutionPolicy Bypass -File $builder -SourceRoot $sourceRootPath -TexFiles $texPath 2>&1
        $buildExitCode = $LASTEXITCODE
    }
    catch {
        $buildOutput += $_.Exception.Message
        $buildExitCode = 1
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $buildText = ($buildOutput | ForEach-Object { [string]$_ }) -join $newline
    $pdfs = New-Object System.Collections.Generic.List[string]
    foreach ($pdfMatch in [regex]::Matches($buildText, 'Writing `(?<path>[^`]+\.pdf)`')) {
        $pdfs.Add($pdfMatch.Groups['path'].Value)
    }
    foreach ($jsonPdfMatch in [regex]::Matches($buildText, '"(?<path>[A-Za-z]:\\\\[^"]+?\.pdf)"')) {
        $pdfs.Add(($jsonPdfMatch.Groups['path'].Value -replace '\\\\', '\'))
    }

    $build = [pscustomobject]@{
        ExitCode = $buildExitCode
        PDFs = @($pdfs | Select-Object -Unique)
        Output = $buildText
    }

    Write-Output $buildText
    if ($buildExitCode -ne 0) {
        throw "PDF build failed with exit code $buildExitCode."
    }
}

$reportPath = [System.IO.Path]::ChangeExtension($texPath, '.reference-audit.json')
$totalReferencesAfter = @($keptEntries).Count
if ($NoEdit) {
    $totalReferencesAfter = @($entries).Count
}

$bodyCitationsReport = @($citations | ForEach-Object { $_ })
$missingBodyCitationsReport = @($missingBodyCitations | ForEach-Object { $_ })
$removedUncitedReport = @($removedUncited | ForEach-Object { $_ })
$removedUnverifiedReport = @($removedUnverified | ForEach-Object { $_ })
$doiActionsReport = @($doiActions | ForEach-Object { $_ })
$unverifiedKeptReport = @($unverifiedKept | ForEach-Object { $_ })
$apaWarningsReport = @($apaWarnings | ForEach-Object { $_ })
$referencesReport = @($references | ForEach-Object {
    [pscustomobject]@{
        Index = $_.Index
        Reference = ('{0} ({1})' -f $_.Metadata.FirstAuthor, $_.Metadata.Year)
        Title = $_.Metadata.Title
        IsCited = $_.IsCited
        ExistingDoi = $_.Metadata.ExistingDoi
        CrossrefReliable = $_.Crossref.Reliable
        CrossrefScore = $_.Crossref.Score
        CrossrefDoi = $_.Crossref.DOI
        CrossrefTitle = $_.Crossref.MatchedTitle
        CrossrefContainer = $_.Crossref.MatchedContainer
        CrossrefNote = $_.Crossref.Note
    }
})

$report = [pscustomobject]@{
    TexFile = $texPath
    SourceRoot = $sourceRootPath
    ProjectRoot = $projectRootPath
    Edited = (-not [bool]$NoEdit)
    RemovedUnverifiedMode = [bool]$RemoveUnverified
    TotalBodyCitations = $bodyCitationsReport.Count
    TotalReferencesBefore = @($entries).Count
    TotalReferencesAfter = $totalReferencesAfter
    BodyCitations = $bodyCitationsReport
    MissingBodyCitations = $missingBodyCitationsReport
    RemovedUncited = $removedUncitedReport
    RemovedUnverified = $removedUnverifiedReport
    DoiActions = $doiActionsReport
    CrossrefUnverifiedKept = $unverifiedKeptReport
    ApaWarnings = $apaWarningsReport
    References = $referencesReport
    Build = $build
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8

Write-Output ''
Write-Output 'Reference audit complete.'
Write-Output ("  Tex file: {0}" -f $texPath)
Write-Output ("  Report: {0}" -f $reportPath)
Write-Output ("  Edited: {0}" -f (-not [bool]$NoEdit))
Write-Output ("  Body citations: {0}" -f $bodyCitationsReport.Count)
Write-Output ("  References before: {0}" -f @($entries).Count)
Write-Output ("  References after: {0}" -f $totalReferencesAfter)
Write-Output ("  Missing body citations: {0}" -f $missingBodyCitationsReport.Count)
Write-Output ("  Removed uncited: {0}" -f $removedUncitedReport.Count)
Write-Output ("  Removed unverified: {0}" -f $removedUnverifiedReport.Count)
Write-Output ("  DOI actions: {0}" -f $doiActionsReport.Count)
Write-Output ("  Crossref-unverified cited references kept: {0}" -f $unverifiedKeptReport.Count)
Write-Output ("  APA warnings: {0}" -f $apaWarningsReport.Count)

if ($build -and @($build.PDFs).Count -gt 0) {
    Write-Output ("  PDFs: {0}" -f (@($build.PDFs) -join '; '))
}
