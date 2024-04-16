import csv
import inspect
import os
import os.path as p

import pandas as pd  # type: ignore

ROOT_DIR = p.realpath(p.join(p.dirname(__file__), ".."))
LOG_DIR = p.join(ROOT_DIR, "log")
PREAMBLE = "[PY]"


def mkdir_p(fpath: str) -> None:
    if p.exists(fpath):
        return
    parent, folder = p.split(fpath)
    mkdir_p(parent)
    os.mkdir(fpath)


def now_ts() -> str:
    return str(pd.Timestamp.now())


def get_current_dir():
    current_script = inspect.stack()[1][0]
    path = os.path.abspath(inspect.getsourcefile(current_script))
    return os.path.dirname(path)


def write_to_log(logname: str, text: str) -> None:
    stamped_text = f"{now_ts()}: {PREAMBLE} {text}"
    if not logname.endswith(".log"):
        logname += ".log"
    fpath = p.join(LOG_DIR, logname)
    write_to_file(fpath, stamped_text)


def write_to_file(fpath: str, text: str) -> None:
    with open(fpath, "a") as f:
        f.write(text)
        f.write("\n")


def write_to_csv(file_path):
    def decorator(inner_function):
        def wrapper(*args, **kwargs):
            try:
                result = inner_function(*args, **kwargs)
                with open(file_path, "a", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(result.keys()) if csv_file.tell() == 0 else None
                    csv_writer.writerow(result.values())
                return result
            except Exception:
                print("Failed to write item to csv")

        return wrapper

    return decorator
