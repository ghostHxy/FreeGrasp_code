#!/usr/bin/env python3
"""
Script to fix NumPy compatibility issues in transforms3d package.

This script fixes the deprecated np.float usage in transforms3d/quaternions.py
that causes AttributeError: module 'numpy' has no attribute 'float'

Usage:
    python fix_numpy_compatibility.py
"""

import os
import re
import sys
from pathlib import Path

def find_transforms3d_path():
    """Find the transforms3d package path in the current Python environment."""
    try:
        import transforms3d
        package_path = Path(transforms3d.__file__).parent
        return package_path
    except ImportError:
        # Try common conda/pip installation paths
        import site
        for site_packages in site.getsitepackages():
            candidate = Path(site_packages) / 'transforms3d'
            if candidate.exists():
                return candidate
        
        # Check user site-packages
        user_site = Path(site.getusersitepackages())
        candidate = user_site / 'transforms3d'
        if candidate.exists():
            return candidate
        
        return None

def fix_quaternions_file(file_path):
    """Fix np.float usage in transforms3d/quaternions.py"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix np.float (standalone, not np.float32/64)
    # Pattern: np.float followed by ), , or whitespace
    patterns = [
        (r'\bnp\.float\)', 'np.float64)'),
        (r'\bnp\.float,', 'np.float64,'),
        (r'\bnp\.float\s', 'np.float64 '),
        (r'\bnp\.float$', 'np.float64'),  # end of line
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content == original_content:
        print(f"✅ No changes needed in {file_path}")
        return True
    
    # Backup original file
    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"📝 Created backup: {backup_path}")
    
    # Write fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed {file_path}")
    
    # Show what was changed
    diff_lines = []
    for orig_line, new_line in zip(original_content.splitlines(), content.splitlines()):
        if orig_line != new_line:
            diff_lines.append(f"  - {orig_line}")
            diff_lines.append(f"  + {new_line}")
    
    if diff_lines:
        print("\nChanges made:")
        print("\n".join(diff_lines[:10]))  # Show first 10 changes
        if len(diff_lines) > 10:
            print(f"  ... and {len(diff_lines) - 10} more changes")
    
    return True

def main():
    """Main function to fix NumPy compatibility issues."""
    print("=" * 60)
    print("NumPy Compatibility Fix Script")
    print("=" * 60)
    print()
    
    # Find transforms3d package
    transforms3d_path = find_transforms3d_path()
    if transforms3d_path is None:
        print("❌ Could not find transforms3d package.")
        print("   Please install it first: pip install transforms3d")
        sys.exit(1)
    
    print(f"📦 Found transforms3d at: {transforms3d_path}")
    print()
    
    # Fix quaternions.py
    quaternions_file = transforms3d_path / 'quaternions.py'
    if quaternions_file.exists():
        print(f"🔧 Fixing {quaternions_file.name}...")
        if fix_quaternions_file(quaternions_file):
            print("✅ Successfully fixed quaternions.py")
        else:
            print("❌ Failed to fix quaternions.py")
            sys.exit(1)
    else:
        print(f"⚠️  File not found: {quaternions_file}")
        print("   The package structure might be different.")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✅ Fix completed!")
    print()
    print("To verify the fix, run:")
    print("  python -c 'import transforms3d.quaternions; print(\"OK\")'")
    print("=" * 60)

if __name__ == '__main__':
    main()

