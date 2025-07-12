import platform
import subprocess
import sys
from typing import Dict, Any

class SystemChecker:
    """System information and hardware validation"""
    
    def __init__(self):
        self.platform = platform.system()
        self.architecture = platform.machine()
    
    def get_system_specs(self) -> Dict[str, Any]:
        """Get system specifications"""
        specs = {
            'platform': self.platform,
            'architecture': self.architecture,
            'python_version': sys.version,
            'processor': platform.processor(),
        }
        
        # Add macOS specific info
        if self.platform == 'Darwin':
            try:
                result = subprocess.run(['sw_vers'], capture_output=True, text=True)
                specs['macos_version'] = result.stdout.strip()
            except:
                specs['macos_version'] = 'Unknown'
        
        return specs
    
    def run_system_checks(self) -> Dict[str, Any]:
        """Run system validation checks"""
        results = {
            'system_specs': self.get_system_specs(),
            'timestamp': platform.platform()
        }
        return results

if __name__ == "__main__":
    checker = SystemChecker()
    specs = checker.get_system_specs()
    print("System Specifications:")
    for key, value in specs.items():
        print(f"  {key}: {value}")