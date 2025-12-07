"""
Build Report Script

This script:
1. Caches to cache/build_report/build_report
2. Copies all ./latex files to the cache directory
3. Creates a symlink from cache/build_report/build_report/v0/cache to ./cache
   (allows importing figures built by other stages)
4. Compiles the LaTeX document
"""

import pathlib
import shutil
import subprocess
import logging
import os

# SETUP =================================================================================
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
LATEX_SRC = PROJECT_ROOT / "latex"
CACHE_DIR = PROJECT_ROOT / "cache" / "build_report" / "build_report"

# Setup logging
CACHE_DIR.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CACHE_DIR / 'build_report.log', mode='w'),
        logging.StreamHandler()
    ]
)

def copy_latex_files():
    """Copy all latex files to the cache directory."""
    logging.info(f"Copying latex files from {LATEX_SRC} to {CACHE_DIR}")

    if not LATEX_SRC.exists():
        logging.error(f"Latex source directory does not exist: {LATEX_SRC}")
        return False

    # Copy the entire latex directory structure
    for item in LATEX_SRC.iterdir():
        dest = CACHE_DIR / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            logging.info(f"Copied directory: {item.name}")
        else:
            shutil.copy2(item, dest)
            logging.info(f"Copied file: {item.name}")

    return True


def create_cache_symlink():
    """Create symlink from cache/build_report/build_report/v0/cache to ./cache."""
    symlink_path = CACHE_DIR / "v0" / "cache"
    target_path = PROJECT_ROOT / "cache"

    logging.info(f"Creating symlink: {symlink_path} -> {target_path}")

    # Remove existing symlink or directory if it exists
    if symlink_path.is_symlink():
        symlink_path.unlink()
        logging.info("Removed existing symlink")
    elif symlink_path.exists():
        shutil.rmtree(symlink_path)
        logging.info("Removed existing directory")

    # Create the symlink
    symlink_path.symlink_to(target_path, target_is_directory=True)
    logging.info("Symlink created successfully")

    return True


def compile_latex():
    """Compile the LaTeX document using pdflatex."""
    build_dir = CACHE_DIR / "v0"
    main_tex = build_dir / "main.tex"

    if not main_tex.exists():
        logging.error(f"main.tex not found at {main_tex}")
        return False

    logging.info(f"Compiling LaTeX document in {build_dir}")

    # Run pdflatex twice for references, then bibtex, then pdflatex twice more
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            logging.info(f"Ran: {' '.join(cmd)}")
            if result.returncode != 0:
                # Log warnings but don't fail - LaTeX often has non-zero exit codes for warnings
                logging.warning(f"Command returned non-zero exit code: {result.returncode}")
                if result.stderr:
                    logging.warning(f"stderr: {result.stderr[:500]}")
        except FileNotFoundError:
            logging.error(f"Command not found: {cmd[0]}. Is LaTeX installed?")
            return False

    # Check if PDF was generated
    pdf_path = build_dir / "main.pdf"
    if pdf_path.exists():
        logging.info(f"PDF generated successfully: {pdf_path}")
        return True
    else:
        logging.error("PDF was not generated")
        return False


def main():
    logging.info("=" * 60)
    logging.info("Starting build report process")
    logging.info("=" * 60)

    # Step 1: Copy latex files
    if not copy_latex_files():
        logging.error("Failed to copy latex files")
        return 1

    # Step 2: Create cache symlink
    if not create_cache_symlink():
        logging.error("Failed to create cache symlink")
        return 1

    # Step 3: Compile LaTeX
    if not compile_latex():
        logging.warning("LaTeX compilation had issues (this may be normal if LaTeX is not installed)")

    logging.info("=" * 60)
    logging.info("Build report process complete")
    logging.info(f"Output directory: {CACHE_DIR / 'v0'}")
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
