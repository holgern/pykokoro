import json
from concurrent.futures import ThreadPoolExecutor

from pykokoro.runtime.cache import DiskCache


def test_disk_cache_corrupted_entry_returns_none(tmp_path):
    cache = DiskCache(tmp_path)
    key = "bad-entry"
    cache_path = cache._path(key)
    cache_path.write_text("{bad json", encoding="utf-8")

    assert cache.get(key) is None
    assert not cache_path.exists()


def test_disk_cache_atomic_write_threads(tmp_path):
    cache = DiskCache(tmp_path)
    key = "shared"

    def write_value(value: int) -> None:
        cache.set(key, {"value": value})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_value, range(50)))

    data = json.loads(cache._path(key).read_text(encoding="utf-8"))
    assert data["value"] in range(50)
