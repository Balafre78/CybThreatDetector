import warnings

from workflow import run_cli


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run_cli()
