# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import random
import struct
import tarfile
from pathlib import Path

import numpy as np


class LazyShuffledRange:
    """
    Generates a permutation of ``range(n)`` lazily using a Feistel cipher,
    without materializing the full index list. Each element is computed on
    the fly in O(1) time and the object itself uses O(1) memory regardless
    of ``n``.

    The technique is known as *cycle-walking* format-preserving encryption:
    a Feistel network is a bijection on ``[0, 2^k)``, and repeatedly applying
    it until the output falls within ``[0, n)`` restricts it to a bijection
    on the desired domain.

    Args:
        n: Size of the range to permute.
        rng: A ``random.Random`` instance used to derive round keys.
    """

    def __init__(self, n: int, rng: random.Random):
        self.n = n
        if n <= 1:
            return
        # Round up to an even number of bits so we can split evenly
        bits = (n - 1).bit_length()
        if bits < 2:
            bits = 2
        if bits % 2:
            bits += 1
        self._half = bits // 2
        self._mask = (1 << self._half) - 1
        self._rounds = 6
        self._keys = [rng.getrandbits(64) for _ in range(self._rounds)]

    def _permute_one(self, x: int) -> int:
        left = (x >> self._half) & self._mask
        right = x & self._mask
        for key in self._keys:
            left, right = right, left ^ (((right * 2654435761) ^ key) >> 32 & self._mask)
        return (left << self._half) | right

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        n = self.n
        if n <= 0:
            return
        if n == 1:
            yield 0
            return
        for i in range(n):
            x = i
            while True:
                x = self._permute_one(x)
                if x < n:
                    yield x
                    break


class IndexedJSONLReader:
    def __init__(self, jsonl_path: Path | str, idx_path: Path | str | None = None):
        self.jsonl_path = Path(jsonl_path)
        if idx_path is None:
            idx_path = self.jsonl_path.with_suffix(self.jsonl_path.suffix + '.idx')
        self.idx_path = Path(idx_path)
        assert self.jsonl_path.exists(), f"JSONL file not found: {self.jsonl_path}"
        assert self.idx_path.exists(), f"Index file not found: {self.idx_path}"

        # 2. Use Memory Mapping (Zero RAM Footprint)
        # 'mode="r"': Open in read-only mode to prevent corruption
        # 'dtype="<u8"': Little-endian unsigned 64-bit integer (same as Energon)
        # This maps the file on disk directly to virtual memory.
        self.offsets = np.memmap(self.idx_path, dtype=np.dtype('<u8'), mode='r')

        # 3. Determine Dataset Length
        # If the index includes a sentinel (EOF offset) at the end, the number of samples is N-1.
        # We verify this by checking if the last offset matches the JSONL file size.
        self.jsonl_size = os.path.getsize(self.jsonl_path)

        # Check if the last offset points to the end of the file (Sentinel exists)
        if self.offsets[-1] == self.jsonl_size:
            self._len = self.offsets.shape[0] - 1
        else:
            # No sentinel; strictly N offsets
            self._len = self.offsets.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Support negative indexing
        if idx < 0:
            idx += self._len

        if idx < 0 or idx >= self._len:
            raise IndexError("Index out of bounds")

        # 4. Efficient Retrieval
        # Accessing self.offsets[idx] reads just 8 bytes from disk (or OS cache)
        start_offset = self.offsets[idx]

        # Calculate length dynamically
        if idx + 1 < self.offsets.shape[0]:
            end_offset = self.offsets[idx + 1]
        else:
            # Fallback for the very last item if no sentinel exists in the .idx
            end_offset = self.jsonl_size

        length = end_offset - start_offset

        # 5. Read Content
        with open(self.jsonl_path, 'rb') as f:
            f.seek(start_offset)
            data = f.read(length)

        return json.loads(data.decode('utf-8'))


def create_index(jsonl_path, idx_path):
    """
    Creates a raw binary index file compatible with Megatron-Energon (CrudeJsonlDataset).

    Format:
    - No header/magic bytes.
    - Sequence of Little-Endian Unsigned 64-bit Integers (uint64).
    - [Offset_0, Offset_1, ..., Offset_N, File_Size]

    Args:
        jsonl_path: Path to the source .jsonl file
        idx_path: Path where the .idx file will be saved
    """
    # 'rb': Read binary (exact byte counts, includes \n)
    # 'wb': Write binary
    with open(jsonl_path, 'rb') as f_in, open(idx_path, 'wb') as f_out:

        current_offset = 0

        # 1. Buffer for writing offsets
        # Writing small 8-byte chunks is slow; we accumulate ~8MB chunks in memory first.
        write_buffer = bytearray()

        # 2. Add the first offset (0)
        write_buffer.extend(struct.pack('<Q', current_offset))

        # 3. Stream the file line-by-line
        # Using the file iterator in binary mode is memory-efficient and fast in CPython.
        for line in f_in:
            # Advance offset by the length of the line (bytes)
            current_offset += len(line)

            # Pack the NEW offset (start of next line / EOF)
            write_buffer.extend(struct.pack('<Q', current_offset))

            # 4. Flush buffer periodically (e.g., whenever it exceeds 8MB)
            if len(write_buffer) > 8 * 1024 * 1024:
                f_out.write(write_buffer)
                write_buffer.clear()

        # 5. Write any remaining offsets
        if write_buffer:
            f_out.write(write_buffer)


def _read_tar_member(f):
    """
    Read a single tar member (header + data) at the current file position.
    Skips PAX extended headers transparently.
    Returns (member_name, data_bytes).
    """
    while True:
        header_buf = f.read(512)
        if len(header_buf) < 512 or header_buf == b'\0' * 512:
            raise EOFError("End of tar archive or unexpected EOF")

        info = tarfile.TarInfo.frombuf(header_buf, tarfile.ENCODING, "surrogateescape")
        data = f.read(info.size)
        if len(data) < info.size:
            raise EOFError("Unexpected end of tar file while reading data")

        # Skip padding to next 512-byte boundary
        remainder = info.size % 512
        if remainder:
            f.seek(512 - remainder, 1)

        # Skip PAX extended headers; re-read to get the actual file
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            continue

        return info.name, data


class IndexedTarSampleReader:
    """
    Provides random access to samples in a WebDataset tar archive using an index file.
    Each sample consists of a pair of files with the same basename: ``N.json`` + ``N.<audio_ext>``.

    The index file has the same format as for ``IndexedJSONLReader``: a sequence of
    little-endian unsigned 64-bit offsets, optionally followed by a sentinel equal to
    the tar file size.

    Each offset points to the start of the first tar member header for that sample (the
    ``.json`` entry).
    """

    def __init__(self, tar_path: str | Path, idx_path: str | Path | None = None):
        self.tar_path = str(tar_path)
        if idx_path is None:
            idx_path = self.tar_path + '.idx'
        self.idx_path = str(idx_path)
        assert os.path.exists(self.tar_path), f"Tar file not found: {self.tar_path}"
        assert os.path.exists(self.idx_path), f"Index file not found: {self.idx_path}"

        self.offsets = np.memmap(self.idx_path, dtype=np.dtype('<u8'), mode='r')
        self.tar_size = os.path.getsize(self.tar_path)

        if self.offsets[-1] == self.tar_size:
            self._len = self.offsets.shape[0] - 1
        else:
            self._len = self.offsets.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError("Index out of bounds")

        offset = int(self.offsets[idx])
        with open(self.tar_path, 'rb') as f:
            f.seek(offset)
            json_name, json_bytes = _read_tar_member(f)
            audio_name, audio_bytes = _read_tar_member(f)

        return json.loads(json_bytes), audio_bytes, audio_name


def create_tar_index(tar_path, idx_path):
    """
    Creates a raw binary index file for a WebDataset tar archive.

    Samples in the tar consist of consecutive members with the same basename
    (e.g. ``0.json``, ``0.wav``).  The index stores the byte offset of the first
    member of each sample, followed by a sentinel equal to the tar file size.

    Format is identical to :func:`create_index`: little-endian ``uint64`` values.
    """
    offsets = []
    prev_stem = None
    with tarfile.open(tar_path, 'r:') as tar:
        for member in tar:
            stem = Path(member.name).stem
            if stem != prev_stem:
                offsets.append(member.offset)
                prev_stem = stem

    with open(idx_path, 'wb') as f:
        write_buffer = bytearray()
        for off in offsets:
            write_buffer.extend(struct.pack('<Q', off))
        # Sentinel: tar file size
        write_buffer.extend(struct.pack('<Q', os.path.getsize(tar_path)))
        f.write(write_buffer)
