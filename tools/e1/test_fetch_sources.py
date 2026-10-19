"""Offline tests for pinned source download, resume, and immutable publication."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_sources as fetch


class Response(io.BytesIO):
    def __init__(self, data, status=200, **headers):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data)), **headers}
        self.read_calls = 0

    def read(self, *args):
        self.read_calls += 1
        return super().read(*args)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="e1-source-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "weights.gguf"
        self.data = b"pinned public source fixture"
        self.sha = hashlib.sha256(self.data).hexdigest()

    def test_fetch_verifies_all_bytes_before_publication(self):
        opener = mock.Mock(return_value=Response(self.data))
        fetch.fetch_file(self.path, "https://example.com/model", len(self.data), self.sha, opener)
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertFalse(self.path.with_suffix(".gguf.partial").exists())
        fetch.fetch_file(self.path, "https://example.com/model", len(self.data), self.sha, opener)
        self.assertEqual(opener.call_count, 1)

    def test_resume_validates_range_and_rehashes_prefix(self):
        partial = self.path.with_suffix(".gguf.partial")
        partial.write_bytes(self.data[:7])
        headers = {"Content-Range": f"bytes 7-{len(self.data)-1}/{len(self.data)}"}
        opener = mock.Mock(return_value=Response(self.data[7:], 206, **headers))
        fetch.fetch_file(self.path, "https://example.com/model", len(self.data), self.sha, opener)
        self.assertEqual(opener.call_args.args[0].get_header("Range"), f"bytes=7-{len(self.data)-1}")
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_resume_rejects_ignored_or_changed_range_before_body(self):
        partial = self.path.with_suffix(".gguf.partial")
        partial.write_bytes(self.data[:7])
        for response in (Response(self.data), Response(self.data[7:], 206)):
            with self.subTest(status=response.status), self.assertRaises(ValueError):
                fetch.fetch_file(self.path, "https://example.com/model", len(self.data), self.sha,
                                 lambda *a, **kw: response)
            self.assertEqual(response.read_calls, 0)
            self.assertEqual(partial.read_bytes(), self.data[:7])

    def test_corrupt_partial_never_publishes(self):
        partial = self.path.with_suffix(".gguf.partial")
        partial.write_bytes(b"!" * len(self.data))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            fetch.fetch_file(self.path, "https://example.com/model", len(self.data), self.sha)
        self.assertFalse(self.path.exists())

    def test_empty_interrupted_download_restarts(self):
        self.path.with_suffix(".gguf.partial").touch()
        fetch.fetch_file(self.path, "https://example.com/model", len(self.data), self.sha,
                         lambda *a, **kw: Response(self.data))
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_published_files_and_symlinks_are_not_overwritten(self):
        fetch.publish_bytes(self.path, self.data)
        fetch.publish_bytes(self.path, self.data)
        with self.assertRaises(ValueError):
            fetch.publish_bytes(self.path, b"other")
        link = self.root / "link"
        link.symlink_to(self.path)
        with self.assertRaises(ValueError):
            fetch.publish_bytes(link, self.data)
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_fetch_done_requires_every_verified_source(self):
        pins = {"a": (len(self.data), self.sha), "b": (len(self.data), self.sha)}
        (self.root / "a").write_bytes(self.data)
        args = type("Args", (), {"source": self.root, "command": "verify"})()
        with mock.patch.object(fetch, "PINNED", pins):
            with self.assertRaises(FileNotFoundError):
                fetch.run(args)
            self.assertFalse((self.root / "FETCH-DONE").exists())
            (self.root / "b").write_bytes(self.data)
            fetch.run(args)
        self.assertEqual((self.root / "FETCH-DONE").read_bytes(), b"DONE\n")


if __name__ == "__main__":
    unittest.main()
