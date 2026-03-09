import subprocess
import os

repo_url = "https://github.com/lemeryfertitta/BoardLib.git"
repo_dir = "BoardLib"

board_name = "kilter"   # change if needed
db_path = "kilter_board.db"
username = "<kilter_board_username>"  # input your username

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

subprocess.run([
    "boardlib",
    "database",
    board_name,
    db_path,
    "--username",
    username
], check=True)

print("Database created:", os.path.abspath(db_path))