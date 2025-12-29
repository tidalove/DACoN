#!/usr/bin/env python3
"""
Generate training configuration files for each model variant.
Creates train-{model}.yaml files from train-tiny.yaml template.
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

def generate_train_configs(template_file="train-tiny.yaml"):
    """Generate training config files for each model."""
    
    # Load the template
    template_path = Path(template_file)
    if not template_path.exists():
        print(f"Error: Template file '{template_file}' not found!")
        return
    
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate a config file for each model
    for model in models:
        # Create a copy of the config
        model_config = yaml.safe_load(yaml.dump(config))
        
        # Modify the dino_model_type
        if 'network' not in model_config:
            model_config['network'] = {}
        model_config['network']['dino_model_type'] = model
        
        # Write to new file
        output_file = f"train-{model}.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Created: {output_file}")
    
    print(f"\nSuccessfully generated {len(models)} training configuration files!")

if __name__ == "__main__":
    generate_train_configs()
