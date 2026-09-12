import argparse
import logging
import tarfile
import uuid
from pathlib import Path


DEFAULT_IMAGES_DIR = Path("data/images")
DEFAULT_OUTPUT = Path("data/photos.tar.gz")

logger = logging.getLogger(__name__)


def iter_wine_dirs(images_dir: Path):
    for path in sorted(images_dir.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        try:
            uuid.UUID(path.name)
        except ValueError:
            continue
        else:
            yield path


def add_file(archive: tarfile.TarFile, source: Path, archive_name: str) -> None:
    info = archive.gettarinfo(str(source), arcname=archive_name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with source.open("rb") as file_obj:
        archive.addfile(info, file_obj)


def build_archive(images_dir: Path, output: Path) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f"{output.name}.tmp")
    wine_count = 0
    file_count = 0

    try:
        with tarfile.open(tmp_output, "w:gz") as archive:
            for wine_dir in iter_wine_dirs(images_dir):
                added_for_wine = 0

                main_files = sorted((wine_dir / "convert_jpg").glob("*.jpg"))
                if main_files:
                    add_file(archive, main_files[0], f"{wine_dir.name}/main.jpg")
                    added_for_wine += 1
                    file_count += 1

                for image_path in sorted((wine_dir / "from_yandex").glob("*.jpg")):
                    add_file(archive, image_path, f"{wine_dir.name}/{image_path.name}")
                    added_for_wine += 1
                    file_count += 1

                if added_for_wine:
                    wine_count += 1

        tmp_output.replace(output)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise

    return wine_count, file_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MinIO seed archive from data/images.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    wines, files = build_archive(images_dir=args.images_dir, output=args.output)
    logger.info("Wrote %s files for %s wines to %s", files, wines, args.output)
