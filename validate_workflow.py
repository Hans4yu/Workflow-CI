#!/usr/bin/env python

import yaml
import sys
import os

def validate_yaml(file_path):
    """Validates a YAML file and checks for GitHub Actions workflow specifics"""
    try:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
            
        print("✅ YAML syntax is valid!")
        
        # Additional validation for GitHub Actions specifics
        if 'name' not in yaml_content:
            print("⚠️  Warning: Workflow does not have a name property")
            
        if 'on' not in yaml_content:
            print("⚠️  Warning: Workflow does not have triggers ('on' property)")
            
        if 'jobs' not in yaml_content:
            print("⚠️  Warning: Workflow does not have any jobs")
        else:
            # Check all jobs have a runs-on property
            for job_name, job in yaml_content['jobs'].items():
                if 'runs-on' not in job:
                    print(f"⚠️  Warning: Job '{job_name}' does not have a 'runs-on' property")
        
        print("\nAll validation checks passed. The workflow file looks good!")
        return True
    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "d:\\All Data\\Projek Portfolio\\SMSML_farhanabdul12\\Workflow-CI\\.github\\workflows\\model_retraining.yml"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
        
    if validate_yaml(file_path):
        sys.exit(0)
    else:
        sys.exit(1)
