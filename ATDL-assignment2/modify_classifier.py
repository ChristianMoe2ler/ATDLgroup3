#!/usr/bin/env python3
"""
Script to modify the pixel_classifier.py MLP architecture.
Usage: python modify_classifier.py [baseline|wider|deeper]
"""
import sys
import shutil

def modify_classifier(variant):
    """Modify the classifier based on the variant."""

    # Backup original
    shutil.copy('src/pixel_classifier.py', 'src/pixel_classifier.py.backup')

    if variant == 'baseline':
        print("Using baseline architecture (no changes)")
        return

    # Read the original file
    with open('src/pixel_classifier.py', 'r') as f:
        lines = f.readlines()

    # Find the __init__ method and replace it
    new_lines = []
    skip_until_next_def = False
    in_init = False
    indent_count = 0

    for i, line in enumerate(lines):
        if 'def __init__(self, numpy_class, dim):' in line:
            in_init = True
            new_lines.append(line)
            new_lines.append('        super(pixel_classifier, self).__init__()\n')

            if variant == 'wider':
                # 2x wider hidden layers
                new_lines.append('        if numpy_class < 30:\n')
                new_lines.append('            self.layers = nn.Sequential(\n')
                new_lines.append('                nn.Linear(dim, 256),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=256),\n')
                new_lines.append('                nn.Linear(256, 64),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=64),\n')
                new_lines.append('                nn.Linear(64, numpy_class)\n')
                new_lines.append('            )\n')
                new_lines.append('        else:\n')
                new_lines.append('            self.layers = nn.Sequential(\n')
                new_lines.append('                nn.Linear(dim, 512),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=512),\n')
                new_lines.append('                nn.Linear(512, 256),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=256),\n')
                new_lines.append('                nn.Linear(256, numpy_class)\n')
                new_lines.append('            )\n')
            elif variant == 'deeper':
                # 4 hidden layers instead of 2
                new_lines.append('        if numpy_class < 30:\n')
                new_lines.append('            self.layers = nn.Sequential(\n')
                new_lines.append('                nn.Linear(dim, 128),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=128),\n')
                new_lines.append('                nn.Linear(128, 96),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=96),\n')
                new_lines.append('                nn.Linear(96, 64),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=64),\n')
                new_lines.append('                nn.Linear(64, 32),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=32),\n')
                new_lines.append('                nn.Linear(32, numpy_class)\n')
                new_lines.append('            )\n')
                new_lines.append('        else:\n')
                new_lines.append('            self.layers = nn.Sequential(\n')
                new_lines.append('                nn.Linear(dim, 256),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=256),\n')
                new_lines.append('                nn.Linear(256, 224),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=224),\n')
                new_lines.append('                nn.Linear(224, 192),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=192),\n')
                new_lines.append('                nn.Linear(192, 128),\n')
                new_lines.append('                nn.ReLU(),\n')
                new_lines.append('                nn.BatchNorm1d(num_features=128),\n')
                new_lines.append('                nn.Linear(128, numpy_class)\n')
                new_lines.append('            )\n')

            skip_until_next_def = True
            continue

        if skip_until_next_def:
            # Skip lines until we find the next method definition
            if line.strip().startswith('def ') and 'def __init__' not in line:
                skip_until_next_def = False
                new_lines.append('\n')
                new_lines.append(line)
            continue

        new_lines.append(line)

    # Write the modified file
    with open('src/pixel_classifier.py', 'w') as f:
        f.writelines(new_lines)

    print(f"Modified classifier to {variant} architecture")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python modify_classifier.py [baseline|wider|deeper]")
        sys.exit(1)

    variant = sys.argv[1].lower()
    if variant not in ['baseline', 'wider', 'deeper']:
        print("Error: variant must be 'baseline', 'wider', or 'deeper'")
        sys.exit(1)

    modify_classifier(variant)
