import argparse
from pathlib import Path
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow yüklü değil. Lütfen: pip install pyarrow", file=sys.stderr)
    sys.exit(1)


def inspect_paths(data_root: Path, rel_paths):
    for rel in rel_paths:
        p = (data_root / rel).resolve()
        try:
            schema = pq.read_schema(p)
            names = schema.names
            types = [str(schema.field(n).type) for n in names]
            print(f"{p} -> {names}")
            print("  dtypes:")
            for n, t in zip(names, types):
                print(f"    - {n}: {t}")
        except Exception as e:
            print(f"{p} schema read error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Parquet şema inceleme aracı")
    parser.add_argument("--data-root", default=r"E:/Bigteam/Project/data", help="Veri kök dizini")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[
            r"content/search_log.parquet",
            r"content/sitewide_log.parquet",
            r"content/top_terms_log.parquet",
            r"user/fashion_sitewide_log.parquet",
        ],
        help="Kök dizine göre göreli parquet yolları",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    inspect_paths(data_root, args.paths)


if __name__ == "__main__":
    main()
