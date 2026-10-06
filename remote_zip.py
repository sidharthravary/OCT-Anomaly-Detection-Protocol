"""remote_zip.py - read members of a large remote zip via HTTP range requests (no full download).

Used by 15_external_kermany.py to fetch only the Kermany OCT test images from the 8.4 GB Mendeley
archive (ZhangLabData.zip, CC BY 4.0).
"""
import io
import urllib.request
import zipfile

KERMANY_URL = ("https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
               "810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded")
UA = {"User-Agent": "Mozilla/5.0"}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a small read-ahead buffer."""

    def __init__(self, url, block=1 << 20):
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()                       # resolve the redirect once
            self.size = int(r.headers["Content-Length"])
        self.pos, self.block = 0, block
        self.buf_start, self.buf = 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={start}-{end - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except OSError:
                if attempt == 4:
                    raise

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.pos + n
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            if n >= self.block:
                data = self._fetch(self.pos, end)
                self.pos = end
                return data
            self.buf_start = self.pos
            self.buf = self._fetch(self.pos, min(self.size, self.pos + self.block))
        i = self.pos - self.buf_start
        data = self.buf[i:i + n]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_remote_zip(url=KERMANY_URL):
    return zipfile.ZipFile(HttpRangeFile(url))
