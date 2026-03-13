import subprocess
import os

repo_url = "https://github.com/lemeryfertitta/BoardLib.git"
repo_dir = "BoardLib"

board_name = "kilter"   # change if needed
db_path = "kilter_board.db"
username = "kiltergpt4900"  # input your username
password = "COMP4900!"

# Clone repo if missing
if not os.path.exists(repo_dir):
    print("Cloning BoardLib...")
    subprocess.run(["git", "clone", repo_url], check=True)

# Install BoardLib
os.chdir(repo_dir)

print("Installing BoardLib...")
subprocess.run(["pip", "install", "."], check=True)

# Build database
print("Downloading board database...")

cmd = "python -m boardlib database kilter kilter_board.db --username kiltergpt4900"

subprocess.run(cmd, shell=True, check=True)

print("Database created:", os.path.abspath(db_path))