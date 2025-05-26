#!/usr/bin/env python3
"""
GitHub Actions Workflow Validator
Validates the YAML syntax of the workflow file
"""

import yaml
import os
import sys

def validate_workflow():
    """Validate the GitHub Actions workflow file"""
    workflow_path = ".github/workflows/model_retraining.yml"
    
    if not os.path.exists(workflow_path):
        print(f"❌ Workflow file not found: {workflow_path}")
        return False
    
    try:
        with open(workflow_path, 'r', encoding='utf-8') as file:
            yaml_content = yaml.safe_load(file)
        
        print("✅ YAML syntax is valid!")
        
        # Check for basic workflow structure
        if 'on' not in yaml_content:
            print("⚠️  Warning: Workflow does not have triggers ('on' property)")
        
        if 'jobs' not in yaml_content:
            print("❌ Error: Workflow does not have jobs")
            return False
        
        # Check if jobs have steps
        for job_name, job_config in yaml_content['jobs'].items():
            if 'steps' not in job_config:
                print(f"❌ Error: Job '{job_name}' does not have steps")
                return False
        
        print("All validation checks passed. The workflow file looks good!")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

if __name__ == "__main__":
    success = validate_workflow()
    sys.exit(0 if success else 1)
