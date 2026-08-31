$ErrorActionPreference = "Stop"
$deckPath = "D:\crop-diversity\deck\crop_diversity.pptx"
$outDir = "D:\crop-diversity\deck\qa_png"

# Slide indices to export. Pass explicit 1-based indices as args to export only
# those slides. With no args, exports every slide in the deck.

# EXPORT RESOLUTION. 3840 x 2160, matching export_qa_case.ps1.
#
# The QA export is what anybody actually looks at when judging whether the deck is sharp, so an
# export coarser than the deck makes a clean frame look soft and sends the next person chasing a
# problem that is not there. A 13.333 in slide at 1920 px is 144 dpi; at 3840 px it runs at 288
# dpi, and this deck's figures are 3,060 px wide on a 10.20 in frame, which the export gives
# 2,938 px. That is within four per cent of what the PNG holds.
#
# 3840 x 2160 is also what a 4K screen shows, so this is the worst case a viewer will ever see
# rather than a laboratory number.
$exportW, $exportH = 3840, 2160
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory $outDir | Out-Null }

$ppt = New-Object -ComObject PowerPoint.Application

# PowerPoint holds an open pptx and blocks the rebuild that produced this file, and it also
# serves a stale copy to Export if the deck was rebuilt underneath it. Close any copy of this
# deck the application already has open before reopening it from disk.
foreach ($p in @($ppt.Presentations)) {
    if ($p.Name -like "crop_diversity*") { $p.Close() }
}

$pres = $ppt.Presentations.Open($deckPath, $true, $false, $false)

if ($args.Count -gt 0) {
    $indices = $args | ForEach-Object { [int]$_ }
} else {
    $indices = 1..$pres.Slides.Count
}

foreach ($i in $indices) {
    $num = "{0:D2}" -f $i
    $outFile = Join-Path $outDir "slide_$num.png"
    $pres.Slides.Item($i).Export($outFile, "PNG", $exportW, $exportH)
}

$pres.Close()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($pres) | Out-Null

Write-Output "Exported $($indices.Count) slides at ${exportW}x${exportH} to $outDir"
