import glob
import logging
import platform
import urllib
from pathlib import Path
from subprocess import check_output

import cv2
import numpy as np
import pkg_resources as pkg
import torch
from cerberusdet.utils.general import colorstr, emojis, get_user_config_dir, is_colab, is_docker

LOGGER = logging.getLogger(__name__)
USER_CONFIG_DIR = get_user_config_dir()  # settings dir


def check_font(font_path: str, progress: bool = False) -> None:
    """
    Download font file to the user's configuration directory if it does not already exist.

    Args:
        font_path (str): Path to font file.
        progress (bool): If True, display a progress bar during the download.

    Returns:
        None
    """
    font = Path(font_path)

    # Destination path for the font file
    file = USER_CONFIG_DIR / font.name

    # Check if font file exists at the source or destination path
    if not font.exists() and not file.exists():
        # Download font file
        url = f"https://ultralytics.com/assets/{font.name}"
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        bool: True if the string is composed only of ASCII characters, False otherwise.
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


def check_online():
    # Check internet connectivity
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


def check_git_status(err_msg=", for updates see https://github.com/ai-forever/CerberusDet"):
    # Recommend 'git pull' if code is out of date
    print(colorstr("github: "), end="")
    try:
        assert Path(".git").exists(), "skipping check (not a git repository)"
        assert not is_docker(), "skipping check (Docker image)"
        assert check_online(), "skipping check (offline)"

        cmd = "git fetch && git config --get remote.origin.url"
        url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip(".git")  # git fetch
        branch = check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()  # checked out
        n = int(check_output(f"git rev-list {branch}..origin/master --count", shell=True))  # commits behind
        if n > 0:
            s = (
                f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. "
                f"Use 'git pull' to update or 'git clone {url}' to download latest."
            )
        else:
            s = f"up to date with {url} ✅"
        print(emojis(s))  # emoji-safe
    except Exception as e:
        print(f"{e}{err_msg}")


def check_python(minimum="3.6.2"):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name="Python ")


def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    assert result, f"{name}{minimum} required by YOLOv5, but {name}{current} is currently installed"


def check_requirements(requirements="requirements.txt", exclude=()):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr("red", "bold", "requirements:")
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        if not file.exists():
            print(f"{prefix} {file.resolve()} not found, check failed.")
            return
        requirements = [f"{x.name}{x.specifier}" for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception:  # DistributionNotFound or VersionConflict if requirements not met
            print(f"{prefix} {r} not found and is required by YOLOv5, attempting auto-update...")
            try:
                assert check_online(), f"'pip install {r}' skipped (offline)"
                print(check_output(f"pip install '{r}'", shell=True).decode())
                n += 1
            except Exception as e:
                print(f"{prefix} {e}")

    if n:  # if packages updated
        source = file.resolve() if "file" in locals() else requirements
        s = (
            f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n"
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        )
        print(emojis(s))  # emoji-safe


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not is_docker(), "cv2.imshow() is disabled in Docker environments"
        assert not is_colab(), "cv2.imshow() is disabled in Google Colab environments"
        cv2.imshow("test", np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f"WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}")
        return False


def check_file(file):
    # Search/download file (if necessary) and return path
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == "":  # exists
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = str(Path(file)).replace(":/", "://")  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file)).name.split("?")[0]  # '%2F' to '/', split https://url.com/file.txt?auth
        print(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, file)
        assert Path(file).exists() and Path(file).stat().st_size > 0, f"File download failed: {url}"  # check
        return file
    else:  # search
        files = glob.glob("./**/" + file, recursive=True)  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file
