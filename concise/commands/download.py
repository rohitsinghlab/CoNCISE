import typer
from pathlib import Path
from typing_extensions import Annotated

DEMO_PATH = "https://github.com/rohitsinghlab/CoNCISE/raw/refs/heads/main/demo/demo.zip"

def main(download_path: Annotated[Path, typer.Argument()] = "./"):
    """
    Download the demo data and config files.
    """
    import requests
    import zipfile
    import os

    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    demo_zip = download_path / "demo.zip"
    response = requests.get(DEMO_PATH)
    demo_zip.write_bytes(response.content)

    with zipfile.ZipFile(demo_zip, "r") as zip_ref:
        zip_ref.extractall(download_path)

    os.remove(demo_zip)
    print(f"Demo data and config files downloaded to {download_path}")