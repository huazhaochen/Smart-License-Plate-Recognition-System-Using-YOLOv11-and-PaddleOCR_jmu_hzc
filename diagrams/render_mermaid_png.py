from pathlib import Path
import requests


BASE_DIR = Path(__file__).resolve().parent
FILES = [
    ("class_diagram.mmd", "class_diagram.png"),
    ("sequence_static.mmd", "sequence_static.png"),
    ("sequence_stream.mmd", "sequence_stream.png"),
]


def render_one(src_name: str, out_name: str) -> None:
    src_path = BASE_DIR / src_name
    out_path = BASE_DIR / out_name
    mermaid_code = src_path.read_text(encoding="utf-8")

    resp = requests.post(
        "https://kroki.io/mermaid/png",
        data=mermaid_code.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        timeout=60,
    )
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"[OK] {src_name} -> {out_name}")


def main() -> None:
    for src, out in FILES:
        render_one(src, out)


if __name__ == "__main__":
    main()
