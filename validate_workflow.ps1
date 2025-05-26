# This script validates GitHub Actions workflow YAML files

param(
    [string]$YamlFile = "d:\All Data\Projek Portfolio\SMSML_farhanabdul12\Workflow-CI\.github\workflows\model_retraining.yml"
)

# Check if PowerShell YAML module is installed
if (-not (Get-Module -ListAvailable -Name 'powershell-yaml')) {
    Write-Host "Installing PowerShell YAML module..."
    Install-Module -Name powershell-yaml -Force -Scope CurrentUser
}

# Import the module
Import-Module powershell-yaml

try {
    # Read the YAML file content
    $yamlContent = Get-Content -Path $YamlFile -Raw
    
    # Try to convert it to a PowerShell object (this will fail if syntax is invalid)
    $yamlObject = ConvertFrom-Yaml -Yaml $yamlContent -ErrorAction Stop
    
    Write-Host "✓ YAML syntax is valid!" -ForegroundColor Green
    
    # Additional validation for GitHub Actions specifics
    if (-not $yamlObject.ContainsKey('name')) {
        Write-Host "⚠️ Warning: Workflow does not have a name property" -ForegroundColor Yellow
    }
    
    if (-not $yamlObject.ContainsKey('on')) {
        Write-Host "⚠️ Warning: Workflow does not have triggers ('on' property)" -ForegroundColor Yellow
    }
    
    if (-not $yamlObject.ContainsKey('jobs')) {
        Write-Host "⚠️ Warning: Workflow does not have any jobs" -ForegroundColor Yellow
    }
    
    # Check all jobs have a runs-on property
    foreach ($jobName in $yamlObject.jobs.Keys) {
        $job = $yamlObject.jobs[$jobName]
        if (-not $job.ContainsKey('runs-on')) {
            Write-Host "⚠️ Warning: Job '$jobName' does not have a 'runs-on' property" -ForegroundColor Yellow
        }
    }
    
    Write-Host "All validation checks passed. The workflow file looks good!" -ForegroundColor Green
} catch {
    Write-Host "❌ YAML syntax error: $_" -ForegroundColor Red
    exit 1
}
