"""Small offline check for the existing gzip transport surface."""
import gzip
import io
import os
import random
import tempfile
from pathlib import Path
import smart_open

payload = bytes(range(256)) * 800
compressed = gzip.compress(payload[:60000], mtime=0) + gzip.compress(b'', mtime=0) + gzip.compress(payload[60000:], mtime=0)
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'sample.gz'
    path.write_bytes(compressed)
    with smart_open.open(path, 'rb') as actual, gzip.GzipFile(fileobj=io.BytesIO(compressed)) as reference:
        rng=random.Random(14)
        for _ in range(80):
            offset=rng.randrange(len(payload)+500)
            assert actual.seek(offset)==reference.seek(offset)
            size=rng.randrange(1,1300)
            assert actual.read(size)==reference.read(size)
            assert actual.tell()==reference.tell()
        actual.seek(0); reference.seek(0)
        a=bytearray(257); b=bytearray(257)
        assert actual.readinto(a)==reference.readinto(b)
        assert a==b
        assert actual.seek(-20,os.SEEK_END)==reference.seek(-20,os.SEEK_END)
        assert actual.read()==reference.read()
    bad=bytearray(compressed);bad[-8]^=1
    path.write_bytes(bad)
    try:
        with smart_open.open(path,'rb') as f: f.read()
    except gzip.BadGzipFile:
        pass
    else:
        raise AssertionError('CRC corruption accepted')
print('PASS: 80 seek/read sequences, multi-member gzip, empty member, readinto, SEEK_END and bad CRC')
