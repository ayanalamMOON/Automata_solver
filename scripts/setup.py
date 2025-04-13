import os
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required system dependencies are installed"""
    dependencies = ['python', 'docker', 'docker-compose', 'redis-cli']
    missing = []
    
    for dep in dependencies:
        try:
            subprocess.run([dep, '--version'], capture_output=True)
        except FileNotFoundError:
            missing.append(dep)
    
    return missing

def setup_virtual_env():
    """Set up Python virtual environment and install dependencies"""
    try:
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
        
        # Activate virtual environment
        if sys.platform == 'win32':
            activate_script = '.venv\\Scripts\\activate'
        else:
            activate_script = '.venv/bin/activate'
        
        # Install dependencies
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'backend/requirements.txt'
        ], check=True)
        
        print("Virtual environment setup complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up virtual environment: {e}")
        return False

def setup_pre_commit():
    """Install and configure pre-commit hooks"""
    try:
        subprocess.run(['pre-commit', 'install'], check=True)
        subprocess.run(['pre-commit', 'autoupdate'], check=True)
        print("Pre-commit hooks installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up pre-commit: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_template = Path('backend/.env.template')
    env_file = Path('backend/.env')
    
    if not env_file.exists() and env_template.exists():
        env_file.write_text(env_template.read_text())
        print(".env file created from template")
        return True
    return False

def main():
    # Check system dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Please install these dependencies before proceeding")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('backend/tests', exist_ok=True)
    os.makedirs('.github/workflows', exist_ok=True)
    
    # Setup virtual environment
    if not setup_virtual_env():
        sys.exit(1)
    
    # Setup pre-commit
    if not setup_pre_commit():
        print("Warning: Pre-commit setup failed")
    
    # Create .env file
    create_env_file()
    
    print("\nSetup complete! Next steps:")
    print("1. Configure your .env file in the backend directory")
    print("2. Start the development environment with: docker-compose up")
    print("3. Run tests with: cd backend && pytest")

if __name__ == '__main__':
    main()