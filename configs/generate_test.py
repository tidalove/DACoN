#!/usr/bin/env python3
"""
Generate test configuration files for each model variant and colorize setting.
Creates test-{model}-{setting}.yaml files from test-tiny.yaml template.
"""

import yaml
from pathlib import Path

# List of models
models = [
    "vitt16-distillationv1_vitb16",
    "vitt16-distillationv1_vits16plus",
    "vitt16-notpretrained_vits16",
    "vitt16_vitb16",
    "vitt16_vits16plus",
    "vitt16-distillationv1_vitb16_200e",
    "vitt16-distillationv1_vits16plus_200e",
    "vitt16-notpretrained_vits16_200e",
    "vitt16_vitb16_200e",
    "vitt16_vits16plus_200e",
    "vitt16-distillationv1_vits16",
    "vitt16-notpretrained_vitb16",
    "vitt16-notpretrained_vits16plus",
    "vitt16_vits16",
    "vitt16-distillationv1_vits16_200e",
    "vitt16-notpretrained_vitb16_200e",
    "vitt16-notpretrained_vits16plus_200e",
    "vitt16_vits16_200e"
]

# Define the three colorize settings
colorize_settings = [
    {
        'name': 'consecutive',
        'colorize_type': 'consecutive_frame',
        'ref_shot': None  # Not used for consecutive_frame
    },
    {
        'name': 'keyframe-ref1',
        'colorize_type': 'keyframe',
        'ref_shot': 1
    },
    {
        'name': 'keyframe-ref5',
        'colorize_type': 'keyframe',
        'ref_shot': 5
    }
]

def generate_test_configs(template_file="test-tiny.yaml"):
    """Generate test config files for each model and colorize setting combination."""
    
    # Load the template
    template_path = Path(template_file)
    if not template_path.exists():
        print(f"Error: Template file '{template_file}' not found!")
        return
    
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    total_files = 0
    
    # Generate a config file for each model and each colorize setting
    for model in models:
        for setting in colorize_settings:
            # Create a copy of the config
            model_config = yaml.safe_load(yaml.dump(config))
            
            # Modify the dino_model_type
            if 'network' not in model_config:
                model_config['network'] = {}
            model_config['network']['dino_model_type'] = model
            
            # Modify colorize settings
            model_config['colorize_type'] = setting['colorize_type']
            
            # Set ref_shot only for keyframe mode
            if setting['ref_shot'] is not None:
                model_config['ref_shot'] = setting['ref_shot']
            elif 'ref_shot' in model_config:
                # Remove ref_shot for consecutive_frame if it exists
                del model_config['ref_shot']
            
            # Write to new file
            output_file = f"test-{model}-{setting['name']}.yaml"
            with open(output_file, 'w') as f:
                yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)
            
            print(f"Created: {output_file}")
            total_files += 1
    
    print(f"\nSuccessfully generated {total_files} test configuration files!")
    print(f"({len(models)} models × {len(colorize_settings)} settings)")

if __name__ == "__main__":
    generate_test_configs()
