from __future__ import annotations

__version__ = "1.4.7"

import functools
import itertools
import json
import multiprocessing as mp
import os
import pickle
import re
import stat
import sys
import tarfile
import textwrap
import time
import zipfile
from argparse import ArgumentParser
from collections import Counter, defaultdict
from http.client import HTTPSConnection
from io import SEEK_END, StringIO
from itertools import islice
from json import loads
from math import log
from pathlib import Path
from signal import signal, SIGPIPE, SIG_DFL
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import morfessor
import numpy as np
import pycld2 as cld2
from icu import BreakIterator, Locale
from loguru import logger
from numpy import float32

__author__: str = "Rami Al-Rfou"
__email__: str = "rmyeid@gmail.com"
__version__: str = "15.10.03"

WORKERS: int = 6

data_path: List[Path] = [Path("~").expanduser()]

LOGFORMAT: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} {level} {file.name}:{line} {message}"


def _open(file_: Any, mode: str = "r") -> Any:
    if isinstance(file_, (str, Path)):
        p = Path(file_)
        if p.suffix in {".bz2", ".gz"}:
            s = tarfile.open(str(p))
            return s.extractfile(s.next())
        return open(str(p), mode)
    return file_


def _print(text: Any) -> None:
    print(text)


def pretty_list(items: Iterable[Any], cols: int = 3) -> str:
    text: List[str] = []
    width: int = 24
    col_width: str = "{" + ":<" + str(width) + "} "
    for i, lang in enumerate(items):
        lang = str(lang)
        if len(lang) > width:
            lang = lang[: width - 3] + "..."
        text.append("{:>3}. ".format(i + 1))
        text.append(col_width.format(lang))
        if (i + 1) % cols == 0:
            text.append("\n")
    return "".join(text)


class Sequence:
    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("This Sequence is Empty")
        if not isinstance(text, str):
            raise ValueError("This is not unicode text instead {}".format(type(text)))
        self.__text: str = text
        self.idx: List[int] = [0, len(self.text)]

    @property
    def text(self) -> str:
        return self.__text

    def __iter__(self) -> Iterator[str]:
        for start, end in zip(self.idx[:-1], self.idx[1:]):
            yield self.text[start:end]

    def tokens(self) -> List[str]:
        return [x.strip() for x in self if x.strip()]

    def __str__(self) -> str:
        return "\n".join(self.tokens())

    def split(self, sequence: "Sequence") -> Iterator["Sequence"]:
        major_idx: List[int] = sequence.idx
        idx2: int = 0
        for start, end in zip(major_idx[:-1], major_idx[1:]):
            idx1 = self.idx.index(start, idx2)
            idx2 = self.idx.index(end, idx2)
            seq = Sequence(self.text[start:end])
            seq.idx = [x - start for x in self.idx[idx1:idx2]]
            yield seq

    def __len__(self) -> int:
        return len(self.idx) - 1

    def empty(self) -> bool:
        return not self.text.strip()


class TokenSequence(list):
    def sliding_window(self, width: int = 2, padding: Any = None) -> Iterator[Tuple[Any, ...]]:
        seq: List[Any] = list(self)
        if padding is not None:
            pad = [padding for _ in range(width // 2)]
            seq = pad + seq + pad
        args = [islice(seq, i, None) for i in range(width)]
        for x in zip(*args):
            yield x


class TextFile:
    def __init__(self, file: Any, delimiter: str = "\n") -> None:
        self.name: Any = file
        self.delimiter: str = delimiter
        self.open_file: Any = open(file, "r")
        self.buf: StringIO = StringIO()

    def iter_delimiter(self, byte_size: int = 8192) -> Iterator[str]:
        partial: str = ""
        while True:
            read_chars = self.read(byte_size)
            if not read_chars:
                break
            partial += read_chars
            lines = partial.split(self.delimiter)
            partial = lines.pop()
            for line in lines:
                yield line + self.delimiter
        if partial:
            yield partial

    def __iter__(self) -> Iterator[str]:
        for l in self.iter_delimiter():
            yield l

    def iter_chunks(self, chunksize: int) -> Iterator[List[str]]:
        chunk: List[str] = []
        for i, l in enumerate(self):
            chunk.append(l)
            if i % chunksize == chunksize - 1:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _append_to_buf(self, contents: str) -> None:
        oldpos: int = self.buf.tell()
        self.buf.seek(0, SEEK_END)
        self.buf.write(contents)
        self.buf.seek(oldpos)

    def peek(self, size: int) -> str:
        contents: str = self.open_file.read(size)
        self._append_to_buf(contents)
        return contents

    def read(self, size: Optional[int] = None) -> str:
        if size is None:
            return self.buf.read() + self.open_file.read()
        contents: str = self.buf.read(size)
        if len(contents) < size:
            contents += self.open_file.read(size - len(contents))
        return contents

    def readline(self) -> str:
        line: str = self.buf.readline()
        if not line.endswith("\n"):
            line += self.open_file.readline()
        return line

    def apply(
        self,
        func: Callable[[List[str]], Any],
        job_size: int = 10000,
    ) -> Iterator[Any]:
        pool = mp.Pool(WORKERS)
        try:
            for result in pool.imap_unordered(func, self.iter_chunks(job_size)):
                yield result
        finally:
            pool.close()
            pool.join()


class TextFiles(TextFile):
    def __init__(self, files: List[Any], delimiter: str = "\n") -> None:
        if isinstance(files[0], (str, Path)):
            self.files: List[TextFile] = [TextFile(f) for f in files]
        else:
            self.files = files
        self.delimiter = delimiter
        self.buf = StringIO()
        self.i: int = 0
        self.open_file = self.files[self.i].open_file

    def readline(self) -> str:
        raise NotImplementedError("Future work")

    def peek(self, size: int) -> str:
        self.open_file.seek(0)
        contents: str = self.open_file.read(size)
        self.open_file.seek(0)
        return contents

    def read(self, size: Optional[int] = None) -> str:
        content: str = super().read(size)
        if not content and self.i < len(self.files) - 1:
            self.i += 1
            self.buf = StringIO()
            self.open_file = self.files[self.i].open_file
            return self.read(size)
        return content

    @property
    def names(self) -> List[Any]:
        return [f.name for f in self.files]


class cached_property:
    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func: Callable[[Any], Any] = func

    def __get__(self, obj: Any, cls: Any) -> Any:
        if obj is None:
            return self
        value = self.func(obj)
        obj.__dict__[self.func.__name__] = value
        return value


def memoize(obj: Callable[..., Any]) -> Callable[..., Any]:
    cache: Dict[Any, Any] = {}

    @functools.wraps(obj)
    def memoizer(*args: Any, **kwargs: Any) -> Any:
        key = tuple(list(args) + sorted(kwargs.items()))
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer


isoLangs: Dict[str, Dict[str, str]] = {
    "ab": {"name": "Abkhaz", "nativeName": "аҧсуа"},
    "aa": {"name": "Afar", "nativeName": "Afaraf"},
    "af": {"name": "Afrikaans", "nativeName": "Afrikaans"},
    "ak": {"name": "Akan", "nativeName": "Akan"},
    "sq": {"name": "Albanian", "nativeName": "Shqip"},
    "am": {"name": "Amharic", "nativeName": "አማርኛ"},
    "ar": {"name": "Arabic", "nativeName": "العربية"},
    "hy": {"name": "Armenian", "nativeName": "Հայերեն"},
    "az": {"name": "Azerbaijani", "nativeName": "azərbaycan dili"},
    "eu": {"name": "Basque", "nativeName": "euskara, euskera"},
    "be": {"name": "Belarusian", "nativeName": "Беларуская"},
    "bn": {"name": "Bengali", "nativeName": "বাংলা"},
    "bs": {"name": "Bosnian", "nativeName": "bosanski jezik"},
    "bg": {"name": "Bulgarian", "nativeName": "български език"},
    "my": {"name": "Burmese", "nativeName": "ဗမာစာ"},
    "ca": {"name": "Catalan; Valencian", "nativeName": "Català"},
    "zh": {"name": "Chinese", "nativeName": "中文 (Zhōngwén), 汉语, 漢語"},
    "hr": {"name": "Croatian", "nativeName": "hrvatski"},
    "cs": {"name": "Czech", "nativeName": "česky, čeština"},
    "da": {"name": "Danish", "nativeName": "dansk"},
    "nl": {"name": "Dutch", "nativeName": "Nederlands, Vlaams"},
    "en": {"name": "English", "nativeName": "English"},
    "eo": {"name": "Esperanto", "nativeName": "Esperanto"},
    "et": {"name": "Estonian", "nativeName": "eesti, eesti keel"},
    "fi": {"name": "Finnish", "nativeName": "suomi, suomen kieli"},
    "fr": {"name": "French", "nativeName": "français, langue française"},
    "gl": {"name": "Galician", "nativeName": "Galego"},
    "ka": {"name": "Georgian", "nativeName": "ქართული"},
    "de": {"name": "German", "nativeName": "Deutsch"},
    "el": {"name": "Greek, Modern", "nativeName": "Ελληνικά"},
    "gu": {"name": "Gujarati", "nativeName": "ગુજરાતી"},
    "ht": {"name": "Haitian; Haitian Creole", "nativeName": "Kreyòl ayisyen"},
    "he": {"name": "Hebrew (modern)", "nativeName": "עברית"},
    "hi": {"name": "Hindi", "nativeName": "हिन्दी, हिंदी"},
    "hu": {"name": "Hungarian", "nativeName": "Magyar"},
    "id": {"name": "Indonesian", "nativeName": "Bahasa Indonesia"},
    "ga": {"name": "Irish", "nativeName": "Gaeilge"},
    "it": {"name": "Italian", "nativeName": "Italiano"},
    "ja": {"name": "Japanese", "nativeName": "日本語 (にほんご／にっぽんご)"},
    "kn": {"name": "Kannada", "nativeName": "ಕನ್ನಡ"},
    "kk": {"name": "Kazakh", "nativeName": "Қазақ тілі"},
    "km": {"name": "Khmer", "nativeName": "ភាសាខ្មែរ"},
    "ko": {"name": "Korean", "nativeName": "한국어 (韓國語), 조선말 (朝鮮語)"},
    "ku": {"name": "Kurdish", "nativeName": "Kurdî, كوردی‎"},
    "la": {"name": "Latin", "nativeName": "latine, lingua latina"},
    "lv": {"name": "Latvian", "nativeName": "latviešu valoda"},
    "lt": {"name": "Lithuanian", "nativeName": "lietuvių kalba"},
    "mk": {"name": "Macedonian", "nativeName": "македонски јазик"},
    "ms": {"name": "Malay", "nativeName": "bahasa Melayu, بهاس ملايو‎"},
    "ml": {"name": "Malayalam", "nativeName": "മലയാളം"},
    "mt": {"name": "Maltese", "nativeName": "Malti"},
    "mr": {"name": "Marathi (Marāṭhī)", "nativeName": "मराठी"},
    "mn": {"name": "Mongolian", "nativeName": "монгол"},
    "ne": {"name": "Nepali", "nativeName": "नेपाली"},
    "no": {"name": "Norwegian", "nativeName": "Norsk"},
    "fa": {"name": "Persian", "nativeName": "فارسی"},
    "pl": {"name": "Polish", "nativeName": "polski"},
    "ps": {"name": "Pashto, Pushto", "nativeName": "پښتو"},
    "pt": {"name": "Portuguese", "nativeName": "Português"},
    "ro": {"name": "Romanian, Moldavian, Moldovan", "nativeName": "română"},
    "ru": {"name": "Russian", "nativeName": "русский язык"},
    "sa": {"name": "Sanskrit (Saṁskṛta)", "nativeName": "संस्कृतम्"},
    "sr": {"name": "Serbian", "nativeName": "српски језик"},
    "sk": {"name": "Slovak", "nativeName": "slovenčina"},
    "sl": {"name": "Slovene", "nativeName": "slovenščina"},
    "so": {"name": "Somali", "nativeName": "Soomaaliga, af Soomaali"},
    "es": {"name": "Spanish; Castilian", "nativeName": "español, castellano"},
    "su": {"name": "Sundanese", "nativeName": "Basa Sunda"},
    "sw": {"name": "Swahili", "nativeName": "Kiswahili"},
    "sv": {"name": "Swedish", "nativeName": "svenska"},
    "ta": {"name": "Tamil", "nativeName": "தமிழ்"},
    "te": {"name": "Telugu", "nativeName": "తెలుగు"},
    "th": {"name": "Thai", "nativeName": "ไทย"},
    "tr": {"name": "Turkish", "nativeName": "Türkçe"},
    "uk": {"name": "Ukrainian", "nativeName": "українська"},
    "ur": {"name": "Urdu", "nativeName": "اردو"},
    "vi": {"name": "Vietnamese", "nativeName": "Tiếng Việt"},
    "cy": {"name": "Welsh", "nativeName": "Cymraeg"},
    "yi": {"name": "Yiddish", "nativeName": "ייִדיש"},
    "yo": {"name": "Yoruba", "nativeName": "Yorùbá"},
}


class Error(Exception):
    pass


class UnknownLanguage(Error):
    pass


class Language:
    def __init__(self, choice: Tuple[str, str, int, int]) -> None:
        basic_name, code, confidence, bytesize = choice
        self.locale: Locale = Locale(code)
        self.confidence: float = float(confidence)
        self.read_bytes: int = int(bytesize)

    @property
    def name(self) -> str:
        return self.locale.getDisplayLanguage()

    @property
    def code(self) -> str:
        return self.locale.getName()

    def __str__(self) -> str:
        return "name: {:<12}code: {:<9}confidence: {:>5.1f} read bytes:{:>6}".format(
            self.name, self.code, self.confidence, self.read_bytes
        )

    @staticmethod
    def from_code(code: str) -> "Language":
        return Language(("", code, 100, 0))


class Detector:
    def __init__(self, text: str, quiet: bool = False) -> None:
        self.__text: str = text
        self.reliable: bool = True
        self.quiet: bool = quiet
        self.languages: List[Language] = []
        self.language: Language
        self.detect(text)

    @staticmethod
    def supported_languages() -> List[str]:
        return [name.capitalize() for name, code in cld2.LANGUAGES if not name.startswith("X_")]

    def detect(self, text: str) -> Language:
        t: bytes = text.encode("utf-8")
        reliable, index, top_3_choices = cld2.detect(t, bestEffort=False)
        if not reliable:
            self.reliable = False
            reliable, index, top_3_choices = cld2.detect(t, bestEffort=True)
            if not reliable and not self.quiet:
                raise UnknownLanguage("Try passing a longer snippet of text")
            logger.warning("Detector is not able to detect the language reliably.")
        self.languages = [Language(x) for x in top_3_choices]
        self.language = self.languages[0]
        return self.language

    def __str__(self) -> str:
        text: str = "Prediction is reliable: {}\n".format(self.reliable)
        text += "\n".join(["Language {}: {}".format(i + 1, str(l)) for i, l in enumerate(self.languages)])
        return text


class ComparableMixin:
    def _compare(self, other: Any, method: Callable[[Any, Any], Any]) -> Any:
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            return NotImplemented

    def __lt__(self, other: Any) -> Any:
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other: Any) -> Any:
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other: Any) -> Any:
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other: Any) -> Any:
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other: Any) -> Any:
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other: Any) -> Any:
        return self._compare(other, lambda s, o: s != o)


class BlobComparableMixin(ComparableMixin):
    def _compare(self, other: Any, method: Callable[[Any, Any], Any]) -> Any:
        if isinstance(other, (str, bytes)):
            return method(self._cmpkey(), other)
        return super()._compare(other, method)


class StringlikeMixin:
    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        text = str(self)
        return '{cls}("{text}")'.format(cls=class_name, text=text)

    def __str__(self) -> str:
        return self._strkey()

    def __len__(self) -> int:
        return len(self._strkey())

    def __iter__(self) -> Iterator[str]:
        return iter(self._strkey())

    def __contains__(self, sub: Any) -> bool:
        return sub in self._strkey()

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, int):
            return self._strkey()[index]
        return self.__class__(self._strkey()[index])

    def find(self, sub: str, start: int = 0, end: int = sys.maxsize) -> int:
        return self._strkey().find(sub, start, end)

    def rfind(self, sub: str, start: int = 0, end: int = sys.maxsize) -> int:
        return self._strkey().rfind(sub, start, end)

    def index(self, sub: str, start: int = 0, end: int = sys.maxsize) -> int:
        return self._strkey().index(sub, start, end)

    def rindex(self, sub: str, start: int = 0, end: int = sys.maxsize) -> int:
        return self._strkey().rindex(sub, start, end)

    def startswith(self, prefix: str, start: int = 0, end: int = sys.maxsize) -> bool:
        return self._strkey().startswith(prefix, start, end)

    def endswith(self, suffix: str, start: int = 0, end: int = sys.maxsize) -> bool:
        return self._strkey().endswith(suffix, start, end)

    starts_with = startswith
    ends_with = endswith

    def title(self) -> Any:
        return self.__class__(self._strkey().title())

    def format(self, *args: Any, **kwargs: Any) -> Any:
        return self.__class__(self._strkey().format(*args, **kwargs))

    def split(self, sep: Optional[str] = None, maxsplit: int = -1) -> Any:
        return self._strkey().split(sep, maxsplit)

    def strip(self, chars: Optional[str] = None) -> Any:
        return self.__class__(self._strkey().strip(chars))

    def upper(self) -> Any:
        return self.__class__(self._strkey().upper())

    def lower(self) -> Any:
        return self.__class__(self._strkey().lower())

    def join(self, iterable: Iterable[str]) -> Any:
        return self.__class__(self._strkey().join(iterable))

    def replace(self, old: str, new: str, count: int = -1) -> Any:
        return self.__class__(self._strkey().replace(old, new, count))


def _count_words(lines: List[str]) -> Counter:
    words = [w for l in lines for w in l.strip().split()]
    return Counter(words)


def _count_words_from_file(filename: Any) -> Counter:
    return CountedVocabulary.from_textfile(filename).word_count


class VocabularyBase:
    def __init__(self, words: Optional[Iterable[str]] = None) -> None:
        words = self.sanitize_words(words or [])
        self.word_id: Dict[str, int] = {w: i for i, w in enumerate(sorted(words))}
        self.id_word: Dict[int, str] = {i: w for w, i in self.word_id.items()}

    def sanitize_words(self, words: Iterable[Any]) -> List[str]:
        _words: List[str] = []
        for w in words:
            if isinstance(w, bytes):
                _words.append(w.decode("utf-8"))
            else:
                _words.append(w)
        return _words

    def __iter__(self) -> Iterator[str]:
        for w, i in sorted(self.word_id.items(), key=lambda wc: wc[1]):
            yield w

    @property
    def words(self) -> List[str]:
        return list(self)

    def __str__(self) -> str:
        return "\n".join(self.words)

    def __getitem__(self, key: Any) -> int:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        return self.word_id[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.word_id

    def __delitem__(self, key: Any) -> None:
        del self.word_id[key]
        self.id_word = dict(enumerate(self.words))
        self.word_id = {w: i for i, w in self.id_word.items()}

    def __len__(self) -> int:
        return len(self.word_id)

    def get(self, k: Any, default: Any = None) -> Any:
        try:
            return self[k]
        except KeyError:
            return default

    def getstate(self) -> Any:
        return list(self.words)

    @classmethod
    def from_vocabfile(cls, filename: Any) -> "VocabularyBase":
        words = [x.strip() for x in _open(filename, "r").read().splitlines()]
        return cls(words=words)


class OrderedVocabulary(VocabularyBase):
    def __init__(self, words: Optional[Iterable[str]] = None) -> None:
        words = self.sanitize_words(words or [])
        self.word_id = {w: i for i, w in enumerate(words)}
        self.id_word = {i: w for w, i in self.word_id.items()}

    def most_frequent(self, k: int) -> "OrderedVocabulary":
        return OrderedVocabulary(words=self.words[:k])


class CountedVocabulary(OrderedVocabulary):
    def __init__(self, word_count: Optional[Any] = None) -> None:
        if word_count is None:
            word_count = {}
        if isinstance(word_count, dict):
            word_count = word_count.items()
        sorted_counts = list(sorted(word_count, key=lambda wc: wc[1], reverse=True))
        words = [w for w, c in sorted_counts]
        super().__init__(words=words)
        self.word_count: Dict[str, int] = dict(sorted_counts)

    @staticmethod
    def from_textfiles(files: TextFiles, job_size: int = 1000) -> "CountedVocabulary":
        c: Counter = Counter()
        pool = mp.Pool(WORKERS)
        try:
            for counter_ in pool.imap_unordered(_count_words_from_file, files.names):
                c.update(counter_)
        finally:
            pool.close()
            pool.join()
        return CountedVocabulary(word_count=c)

    @classmethod
    def from_textfile(cls, textfile: Any, job_size: int = 1000) -> "CountedVocabulary":
        c: Counter = Counter()
        if isinstance(textfile, (str, Path)):
            textfile = TextFile(textfile)
        for result in textfile.apply(_count_words, job_size):
            c.update(result)
        return CountedVocabulary(word_count=c)

    def most_frequent(self, k: int) -> "CountedVocabulary":
        word_count = {w: self.word_count[w] for w in self.words[:k]}
        return CountedVocabulary(word_count=word_count)

    def min_count(self, n: int = 1) -> "CountedVocabulary":
        word_count = {w: c for w, c in self.word_count.items() if c >= n}
        return CountedVocabulary(word_count=word_count)

    def __str__(self) -> str:
        return "\n".join(["{}\t{}".format(w, self.word_count[w]) for w in self.words])

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)
        self.word_count = {w: self.word_count[w] for w in self}

    def getstate(self) -> Any:
        words = list(self.words)
        counts = [self.word_count[w] for w in words]
        return (words, counts)

    @staticmethod
    def from_vocabfile(filename: Any) -> "CountedVocabulary":
        word_count = [x.strip().split() for x in _open(filename, "r").read().splitlines()]
        word_count = {w: int(c) for w, c in word_count}
        return CountedVocabulary(word_count=word_count)


class Embedding:
    def __init__(self, vocabulary: VocabularyBase, vectors: Any) -> None:
        self.vocabulary: VocabularyBase = vocabulary
        self.vectors: np.ndarray = np.asarray(vectors)
        if len(self.vocabulary) != self.vectors.shape[0]:
            raise ValueError(
                "Vocabulary has {} items but we have {} vectors".format(len(vocabulary), self.vectors.shape[0])
            )

    def __getitem__(self, k: Any) -> np.ndarray:
        return self.vectors[self.vocabulary[k]]

    def __contains__(self, k: Any) -> bool:
        return k in self.vocabulary

    def __delitem__(self, k: Any) -> None:
        index = self.vocabulary[k]
        del self.vocabulary[k]
        self.vectors = np.delete(self.vectors, index, 0)

    def __len__(self) -> int:
        return len(self.vocabulary)

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        for w in self.vocabulary:
            yield w, self[w]

    @property
    def words(self) -> List[str]:
        return self.vocabulary.words

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.vectors.shape

    def apply_expansion(self, expansion: Callable[[VocabularyBase], VocabularyBase]) -> None:
        self.vocabulary = expansion(self.vocabulary)

    def get(self, k: Any, default: Any = None) -> Any:
        try:
            return self[k]
        except KeyError:
            return default

    def most_frequent(self, k: int, inplace: bool = False) -> "Embedding":
        vocabulary = self.vocabulary.most_frequent(k)
        vectors = np.asarray([self[w] for w in vocabulary])
        if inplace:
            self.vocabulary = vocabulary
            self.vectors = vectors
            return self
        return Embedding(vectors=vectors, vocabulary=vocabulary)

    def normalize_words(self, ord: Optional[int] = 2, inplace: bool = False) -> "Embedding":
        if ord == 2:
            ord = None
        vectors = self.vectors.T / np.linalg.norm(self.vectors, ord, axis=1)
        if inplace:
            self.vectors = vectors.T
            return self
        return Embedding(vectors=vectors.T, vocabulary=self.vocabulary)

    def nearest_neighbors(self, word: str, top_k: int = 10) -> List[str]:
        point = self[word]
        diff = self.vectors - point
        distances = np.linalg.norm(diff, axis=1)
        top_ids = distances.argsort()[1 : top_k + 1]
        return [self.vocabulary.id_word[i] for i in top_ids]

    def distances(self, word: str, words: Iterable[str]) -> np.ndarray:
        point = self[word]
        vectors = np.asarray([self[w] for w in words])
        diff = vectors - point
        return np.linalg.norm(diff, axis=1)

    @staticmethod
    def from_gensim(model: Any) -> "Embedding":
        word_count: Dict[str, int] = {}
        vectors: List[np.ndarray] = []
        for word, vocab in sorted(model.vocab.items(), key=lambda item: -item[1].count):
            vectors.append(model.syn0[vocab.index])
            word_count[word] = vocab.count
        vocab = CountedVocabulary(word_count=word_count)
        vectors_arr = np.asarray(vectors)
        return Embedding(vocabulary=vocab, vectors=vectors_arr)

    @staticmethod
    def from_word2vec_vocab(fvocab: Any) -> CountedVocabulary:
        counts: Dict[str, int] = {}
        with _open(fvocab) as fin:
            for line in fin:
                word, count = str(line).strip().split()
                counts[word] = int(count)
        return CountedVocabulary(word_count=counts)

    @staticmethod
    def _from_word2vec_binary(fname: Any) -> Tuple[List[str], np.ndarray]:
        with _open(fname, "rb") as fin:
            words: List[str] = []
            header = fin.readline().decode("utf-8")
            vocab_size, layer1_size = list(map(int, header.split()))
            vectors = np.zeros((vocab_size, layer1_size), dtype=float32)
            binary_len = np.dtype(float32).itemsize * layer1_size
            for line_no in range(vocab_size):
                word: List[bytes] = []
                while True:
                    ch = fin.read(1)
                    if ch == b" ":
                        break
                    if ch != b"\n":
                        word.append(ch)
                words.append(b"".join(word).decode("utf-8"))
                vectors[line_no, :] = np.frombuffer(fin.read(binary_len), dtype=float32)
            return words, vectors

    @staticmethod
    def _from_word2vec_text(fname: Any) -> Tuple[List[str], np.ndarray]:
        with _open(fname, "rb") as fin:
            words: List[str] = []
            header = fin.readline().decode("utf-8")
            vocab_size, layer1_size = list(map(int, header.split()))
            vectors: List[List[float]] = []
            for line_no, line in enumerate(fin):
                try:
                    parts = line.decode("utf-8").strip().split()
                except Exception as e:
                    logger.warning("We ignored line number {} because of errors in parsing\n{}".format(line_no, e))
                    continue
                if len(parts) == layer1_size + 1:
                    word, weights = parts[0], list(map(float32, parts[1:]))
                elif len(parts) == layer1_size + 2:
                    word, weights = " ".join(parts[:2]), list(map(float32, parts[2:]))
                else:
                    logger.warning(
                        "We ignored line number {} because of unrecognized number of columns {}".format(
                            line_no, parts[:-layer1_size]
                        )
                    )
                    continue
                words.append(word)
                vectors.append(weights)
            vectors_arr = np.asarray(vectors, dtype=np.float32)
            return words, vectors_arr

    @staticmethod
    def from_word2vec(
        fname: Any,
        fvocab: Optional[Any] = None,
        binary: bool = False,
    ) -> "Embedding":
        vocabulary: Optional[VocabularyBase] = None
        if fvocab is not None:
            logger.info("loading word counts from {}".format(fvocab))
            vocabulary = Embedding.from_word2vec_vocab(fvocab)
        logger.info("loading projection weights from {}".format(fname))
        if binary:
            words, vectors = Embedding._from_word2vec_binary(fname)
        else:
            words, vectors = Embedding._from_word2vec_text(fname)
        if not vocabulary:
            vocabulary = OrderedVocabulary(words=words)
        return Embedding(vocabulary=vocabulary, vectors=vectors)

    @staticmethod
    def load(fname: Any) -> "Embedding":
        content = _open(fname).read()
        state = pickle.loads(content, encoding="latin1")
        voc, vec = state
        if len(voc) == 2:
            words, counts = voc
            word_count = dict(zip(words, counts))
            vocab: VocabularyBase = CountedVocabulary(word_count=word_count)
        else:
            vocab = OrderedVocabulary(voc)
        return Embedding(vocabulary=vocab, vectors=vec)

    def save(self, fname: Any) -> None:
        vec = self.vectors
        voc = self.vocabulary.getstate()
        state = (voc, vec)
        with open(str(fname), "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


class VocabExpander(OrderedVocabulary):
    def __init__(
        self,
        vocabulary: VocabularyBase,
        formatters: List[Callable[[str], str]],
        strategy: str,
    ) -> None:
        super().__init__(vocabulary.words)
        self.strategy: str = strategy
        self._vocab: VocabularyBase = vocabulary
        self.aux_word_id: Dict[str, Any] = defaultdict(list)
        self.formatters: List[Callable[[str], str]] = formatters
        self.expand(formatters)
        self.aux_id_word: Dict[Any, str] = {id_: w for w, id_ in self.aux_word_id.items()}

    def __getitem__(self, key: Any) -> Any:
        try:
            return self._vocab[key]
        except KeyError:
            try:
                return self.aux_word_id[key]
            except KeyError:
                return self.approximate_ids(key)

    def __contains__(self, key: Any) -> bool:
        return (key in self._vocab) or (key in self.aux_word_id) or bool(self.approximate(key))

    def __len__(self) -> int:
        return len(self._vocab) + len(self.aux_word_id)

    def __delitem__(self, key: Any) -> None:
        raise NotImplementedError("It is quite complex, let us do it in the future")

    def format(self, w: str) -> List[str]:
        return [f(w) for f in self.formatters]

    def approximate(self, w: str) -> Dict[str, Any]:
        f = lambda key: (key in self._vocab) or (key in self.aux_word_id)
        return {w_: self[w_] for w_ in self.format(w) if f(w_)}

    def approximate_ids(self, key: str) -> Any:
        ids = [id_ for w, id_ in self.approximate(key).items()]
        if not ids:
            raise KeyError("{} not found".format(key))
        if self.strategy == "most_frequent":
            return min(ids)
        return tuple(sorted(ids))

    def _expand(self, formatter: Callable[[str], str]) -> None:
        for w in self.word_id:
            w_ = formatter(w)
            if w_ not in self._vocab:
                id_ = self.word_id[w]
                self.aux_word_id[w_].append(id_)

    def expand(self, formatters: List[Callable[[str], str]]) -> None:
        for formatter in formatters:
            self._expand(formatter)
        if self.strategy == "average":
            self.aux_word_id = {w: tuple(sorted(ids)) for w, ids in self.aux_word_id.items()}
        elif self.strategy == "most_frequent":
            self.aux_word_id = {w: min(ids) for w, ids in self.aux_word_id.items()}
        else:
            raise ValueError("A strategy is needed")
        words_added = self.aux_word_id.keys()
        old_no = len(self._vocab)
        new_no = len(self.aux_word_id)
        logger.info("We have {} original words.".format(old_no))
        logger.info("Added {} new words.".format(new_no))
        logger.info("The new total number of words is {}".format(len(self)))
        logger.debug("Words added\n{}\n".format(" ".join(words_added)))


class CaseExpander(VocabExpander):
    def __init__(self, vocabulary: VocabularyBase, strategy: str = "most_frequent") -> None:
        formatters = [lambda x: x.lower(), lambda x: x.title(), lambda x: x.upper()]
        super().__init__(vocabulary=vocabulary, formatters=formatters, strategy=strategy)


class DigitExpander(VocabExpander):
    def __init__(self, vocabulary: VocabularyBase, strategy: str = "most_frequent") -> None:
        pattern = re.compile("[0-9]", flags=re.UNICODE)
        formatters = [lambda x: pattern.sub("#", x)]
        super().__init__(vocabulary=vocabulary, formatters=formatters, strategy=strategy)


class ExceptionBase(Exception):
    pass


class LanguageNotSupported(ExceptionBase):
    pass


class TaskNotSupported(ExceptionBase):
    pass


def is_writable(path_: Path) -> bool:
    if not path_.exists():
        return False
    if hasattr(os, "getuid"):
        statdata = path_.stat()
        perm = stat.S_IMODE(statdata.st_mode)
        if perm & 0o002:
            return True
        elif statdata.st_uid == os.getuid() and (perm & 0o200):
            return True
        elif (statdata.st_gid in [os.getgid()] + os.getgroups()) and (perm & 0o020):
            return True
        return False
    return True


class Package:
    def __init__(
        self,
        id: str,
        url: str,
        name: Optional[str] = None,
        subdir: str = "",
        size: Optional[int] = None,
        filename: str = "",
        task: str = "",
        language: str = "",
        attrs: Optional[Any] = None,
        **kw: Any,
    ) -> None:
        self.id: str = id
        self.name: str = name or id
        self.subdir: str = subdir
        self.url: str = url
        self.size: int = int(size or 0)
        self.filename: str = filename
        self.task: str = task
        self.language: str = language
        self.attrs: Optional[Any] = attrs
        self.__dict__.update(kw)

    @staticmethod
    def fromcsobj(csobj: Dict[str, Any]) -> "Package":
        attrs = csobj
        id_ = attrs["id"].split("/")
        id = ".".join(id_[1:3])
        name = attrs["name"].replace("/", ".")
        subdir = str(Path(attrs["name"]).parent)
        url = attrs["mediaLink"]
        size = attrs["size"]
        filename = attrs["name"]
        task = subdir.split("/")[0]
        language = subdir.split("/")[1]
        return Package(**locals())

    def __lt__(self, other: "Package") -> bool:
        return self.id < other.id

    def __repr__(self) -> str:
        return "<Package %s>" % self.id


class Collection:
    def __init__(self, id: str, children: List[Any], name: Optional[str] = None, **kw: Any) -> None:
        self.id: str = id
        self.name: str = name or id
        self.children: List[Any] = children
        self.packages: Optional[Iterable[Package]] = None
        self.__dict__.update(kw)

    def __lt__(self, other: "Collection") -> bool:
        return self.id < other.id

    def __repr__(self) -> str:
        return "<Collection %s>" % self.id


class DownloaderMessage:
    pass


class StartCollectionMessage(DownloaderMessage):
    def __init__(self, collection: Collection) -> None:
        self.collection = collection


class FinishCollectionMessage(DownloaderMessage):
    def __init__(self, collection: Collection) -> None:
        self.collection = collection


class StartPackageMessage(DownloaderMessage):
    def __init__(self, package: Package) -> None:
        self.package = package


class FinishPackageMessage(DownloaderMessage):
    def __init__(self, package: Package) -> None:
        self.package = package


class StartDownloadMessage(DownloaderMessage):
    def __init__(self, package: Package) -> None:
        self.package = package


class FinishDownloadMessage(DownloaderMessage):
    def __init__(self, package: Package) -> None:
        self.package = package


class StartUnzipMessage(DownloaderMessage):
    def __init__(self, package: Package) -> None:
        self.package = package


class FinishUnzipMessage(DownloaderMessage):
    def __init__(self, package: Package) -> None:
        self.package = package


class UpToDateMessage(DownloaderMessage):
    def __init__(self, package: Package) -> None:
        self.package = package


class StaleMessage(DownloaderMessage):
    def __init__(self, package: Package) -> None:
        self.package = package


class ErrorMessage(DownloaderMessage):
    def __init__(self, package: Optional[Package], message: Any) -> None:
        self.package = package
        self.message: str = str(message) if isinstance(message, Exception) else message


class ProgressMessage(DownloaderMessage):
    def __init__(self, progress: float) -> None:
        self.progress = progress


class SelectDownloadDirMessage(DownloaderMessage):
    def __init__(self, download_dir: Path) -> None:
        self.download_dir = download_dir


class Downloader:
    INDEX_TIMEOUT: int = 60 * 60
    DEFAULT_SOURCE: str = "mirror"
    DEFAULT_URL: str = "http://whoisbigger.com/polyglot/"
    INSTALLED: str = "installed"
    NOT_INSTALLED: str = "not installed"
    STALE: str = "out of date"
    PARTIAL: str = "partial"
    LANG_PREFIX: str = "LANG:"
    TASK_PREFIX: str = "TASK:"

    def __init__(
        self,
        server_index_url: Optional[str] = None,
        source: Optional[str] = None,
        download_dir: Optional[Path] = None,
    ) -> None:
        self._url: str = server_index_url or self.DEFAULT_URL
        self._source: str = source or self.DEFAULT_SOURCE
        self._collections: Dict[str, Collection] = {}
        self._packages: Dict[str, Package] = {}
        self._download_dir: Optional[Path] = download_dir
        self._index: Optional[Any] = None
        self._index_timestamp: Optional[float] = None
        self._status_cache: Dict[str, str] = {}
        self._errors: Optional[bool] = None
        if self._download_dir is None:
            self._download_dir = self.default_download_dir()

    def list(
        self,
        download_dir: Optional[Path] = None,
        show_packages: bool = False,
        show_collections: bool = True,
        header: bool = True,
        more_prompt: bool = False,
        skip_installed: bool = False,
    ) -> None:
        lines = 0
        if download_dir is None:
            download_dir = self._download_dir
            print("Using default data directory (%s)" % download_dir)
        if header:
            print("=" * (26 + len(self._url)))
            print(" Data server index for <%s>" % self._url)
            print("=" * (26 + len(self._url)))
            lines += 3
        stale = partial = False
        categories: List[str] = []
        if show_packages:
            categories.append("packages")
        if show_collections:
            categories.append("collections")
        for category in categories:
            print("%s:" % category.capitalize())
            lines += 1
            for info in sorted(getattr(self, category)(), key=str):
                status = self.status(info, download_dir)
                if status == self.INSTALLED and skip_installed:
                    continue
                if status == self.STALE:
                    stale = True
                if status == self.PARTIAL:
                    partial = True
                prefix = {self.INSTALLED: "*", self.STALE: "-", self.PARTIAL: "P", self.NOT_INSTALLED: " "}[status]
                name = textwrap.fill("-" * 27 + (info.name or info.id), 75, subsequent_indent=27 * " ")[27:]
                print("  [%s] %s %s" % (prefix, info.id.ljust(20, "."), name))
                lines += len(name.split("\n"))
                if more_prompt and lines > 20:
                    user_input = input("Hit Enter to continue: ")
                    if user_input.lower() in ("x", "q"):
                        return
                    lines = 0
            print()
        msg = "([*] marks installed packages"
        if stale:
            msg += "; [-] marks out-of-date or corrupt packages"
        if partial:
            msg += "; [P] marks partially installed collections"
        print(textwrap.fill(msg + ")", subsequent_indent=" ", width=76))

    def packages(self) -> Iterable[Package]:
        self._update_index()
        return self._packages.values()

    def corpora(self) -> List[Package]:
        self._update_index()
        return [pkg for (id, pkg) in self._packages.items() if pkg.subdir == "corpora"]

    def models(self) -> List[Package]:
        self._update_index()
        return [pkg for (id, pkg) in self._packages.items() if pkg.subdir != "corpora"]

    def collections(self) -> Iterable[Collection]:
        self._update_index()
        return self._collections.values()

    def _info_or_id(self, info_or_id: Any) -> Any:
        if isinstance(info_or_id, str):
            return self.info(info_or_id)
        return info_or_id

    def incr_download(
        self,
        info_or_id: Any,
        download_dir: Optional[Path] = None,
        force: bool = False,
    ) -> Iterator[DownloaderMessage]:
        if download_dir is None:
            download_dir = self._download_dir
            yield SelectDownloadDirMessage(download_dir)
        if isinstance(info_or_id, (list, tuple)):
            for msg in self._download_list(info_or_id, download_dir, force):
                yield msg
            return
        try:
            info = self._info_or_id(info_or_id)
        except (IOError, ValueError) as e:
            yield ErrorMessage(None, "Error loading %s: %s" % (info_or_id, e))
            return
        if isinstance(info, Collection):
            yield StartCollectionMessage(info)
            for msg in self.incr_download(info.children, download_dir, force):
                yield msg
            yield FinishCollectionMessage(info)
        else:
            for msg in self._download_package(info, download_dir, force):
                yield msg

    def _num_packages(self, item: Any) -> int:
        if isinstance(item, Package):
            return 1
        return len(list(item.packages or []))

    def _download_list(
        self,
        items: List[Any],
        download_dir: Path,
        force: bool,
    ) -> Iterator[DownloaderMessage]:
        for i in range(len(items)):
            try:
                items[i] = self._info_or_id(items[i])
            except (IOError, ValueError) as e:
                yield ErrorMessage(items[i], e)
                return
        num_packages = sum(self._num_packages(item) for item in items)
        progress: float = 0
        for i, item in enumerate(items):
            if isinstance(item, Package):
                delta = 1.0 / num_packages
            else:
                delta = float(len(list(item.packages))) / num_packages
            for msg in self.incr_download(item, download_dir, force):
                if isinstance(msg, ProgressMessage):
                    yield ProgressMessage(progress + msg.progress * delta)
                else:
                    yield msg
            progress += 100 * delta

    def _download_package(
        self,
        info: Package,
        download_dir: Path,
        force: bool,
    ) -> Iterator[DownloaderMessage]:
        yield StartPackageMessage(info)
        yield ProgressMessage(0)
        status = self.status(info, download_dir)
        if not force and status == self.INSTALLED:
            yield UpToDateMessage(info)
            yield ProgressMessage(100)
            yield FinishPackageMessage(info)
            return
        self._status_cache.pop(info.id, None)
        filepath = download_dir / info.filename
        if filepath.exists():
            if status == self.STALE:
                yield StaleMessage(info)
            filepath.unlink()
        subdir_path = download_dir / info.subdir
        if not subdir_path.exists():
            subdir_path.mkdir(parents=True)
        yield StartDownloadMessage(info)
        yield ProgressMessage(5)
        try:
            infile = urlopen(info.url)
            with open(str(filepath), "wb") as outfile:
                num_blocks = max(1, float(info.size) / (1024 * 16))
                for block in itertools.count():
                    s = infile.read(1024 * 16)
                    outfile.write(s)
                    if not s:
                        break
                    if block % 2 == 0:
                        yield ProgressMessage(min(80, 5 + 75 * (block / num_blocks)))
            infile.close()
        except IOError as e:
            yield ErrorMessage(info, "Error downloading %r from <%s>:\n  %s" % (info.id, info.url, e))
            return
        yield FinishDownloadMessage(info)
        yield ProgressMessage(80)
        if info.filename.endswith(".zip"):
            zipdir = download_dir / info.subdir
            if getattr(info, "unzip", False) or (zipdir / info.id).exists():
                yield StartUnzipMessage(info)
                for msg in _unzip_iter(filepath, zipdir, verbose=False):
                    msg.package = info
                    yield msg
                yield FinishUnzipMessage(info)
        yield FinishPackageMessage(info)

    def download(
        self,
        info_or_id: Optional[Any] = None,
        download_dir: Optional[Path] = None,
        quiet: bool = False,
        force: bool = False,
        prefix: str = "[polyglot_data] ",
        halt_on_error: bool = True,
        raise_on_error: bool = False,
    ) -> bool:
        if info_or_id is None:
            if download_dir is not None:
                self._download_dir = download_dir
            self._interactive_download()
            return True

        if isinstance(info_or_id, str):
            info_or_id = str(info_or_id)

        def show(s: str, prefix2: str = "") -> None:
            print(textwrap.fill(s, initial_indent=prefix + prefix2, subsequent_indent=prefix + prefix2 + " " * 4))

        for msg in self.incr_download(info_or_id, download_dir, force):
            if isinstance(msg, ErrorMessage):
                show(msg.message)
                if raise_on_error:
                    raise ValueError(msg.message)
                if halt_on_error:
                    return False
                self._errors = True
                if not quiet:
                    print("Error installing package. Retry? [n/y/e]")
                    choice = input().strip()
                    if choice in ["y", "Y"]:
                        if not self.download(
                            msg.package.id, download_dir, quiet, force, prefix, halt_on_error, raise_on_error
                        ):
                            return False
                    elif choice in ["e", "E"]:
                        return False
            if not quiet:
                if isinstance(msg, StartCollectionMessage):
                    show("Downloading collection %r" % msg.collection.id)
                    prefix += "   | "
                    print(prefix)
                elif isinstance(msg, FinishCollectionMessage):
                    print(prefix)
                    prefix = prefix[:-4]
                    if self._errors:
                        show("Downloaded collection %r with errors" % msg.collection.id)
                    else:
                        show("Done downloading collection %s" % msg.collection.id)
                elif isinstance(msg, StartPackageMessage):
                    show("Downloading package %s to %s..." % (msg.package.id, download_dir))
                elif isinstance(msg, UpToDateMessage):
                    show("Package %s is already up-to-date!" % msg.package.id, "  ")
                elif isinstance(msg, StartUnzipMessage):
                    show("Unzipping %s." % msg.package.filename, "  ")
                elif isinstance(msg, SelectDownloadDirMessage):
                    download_dir = msg.download_dir
        return True

    def is_stale(self, info_or_id: Any, download_dir: Optional[Path] = None) -> bool:
        return self.status(info_or_id, download_dir) == self.STALE

    def is_installed(self, info_or_id: Any, download_dir: Optional[Path] = None) -> bool:
        return self.status(info_or_id, download_dir) == self.INSTALLED

    def clear_status_cache(self, id: Optional[str] = None) -> None:
        if id is None:
            self._status_cache.clear()
        else:
            self._status_cache.pop(id, None)

    def status(self, info_or_id: Any, download_dir: Optional[Path] = None) -> str:
        if download_dir is None:
            download_dir = self._download_dir
        info = self._info_or_id(info_or_id)
        if isinstance(info, Collection):
            pkg_status = [self.status(pkg.id) for pkg in info.packages]
            if self.STALE in pkg_status:
                return self.STALE
            elif self.PARTIAL in pkg_status:
                return self.PARTIAL
            elif self.INSTALLED in pkg_status and self.NOT_INSTALLED in pkg_status:
                return self.PARTIAL
            elif self.NOT_INSTALLED in pkg_status:
                return self.NOT_INSTALLED
            return self.INSTALLED
        filepath = download_dir / info.filename
        if download_dir != self._download_dir:
            return self._pkg_status(info, filepath)
        if info.id not in self._status_cache:
            self._status_cache[info.id] = self._pkg_status(info, filepath)
        return self._status_cache[info.id]

    def _pkg_status(self, info: Package, filepath: Path) -> str:
        if not filepath.exists():
            return self.NOT_INSTALLED
        try:
            filestat = filepath.stat()
        except OSError:
            return self.NOT_INSTALLED
        if filestat.st_size != int(info.size):
            return self.STALE
        if filepath.suffix == ".zip":
            unzipdir = filepath.with_suffix("")
            if not unzipdir.exists():
                return self.INSTALLED
            if not unzipdir.is_dir():
                return self.STALE
            unzipped_size = sum(f.stat().st_size for f in unzipdir.rglob("*") if f.is_file())
            if unzipped_size != getattr(info, "unzipped_size", unzipped_size):
                return self.STALE
        return self.INSTALLED

    def update(self, quiet: bool = False, prefix: str = "[polyglot_data] ") -> None:
        self.clear_status_cache()
        for pkg in self.packages():
            if self.status(pkg) == self.STALE:
                self.download(pkg, quiet=quiet, prefix=prefix)

    def _update_index(self, url: Optional[str] = None) -> None:
        if not (
            self._index is None or url is not None or time.time() - (self._index_timestamp or 0) > self.INDEX_TIMEOUT
        ):
            return
        self._url = url or self._url
        source = self._source
        assert source in ("google", "mirror")
        if source == "google":
            host = "www.googleapis.com"
            conn = HTTPSConnection(host)
            conn.request("GET", "/storage/v1/b/{}/o".format(self._url))
            r1 = conn.getresponse()
            data_bytes = r1.read()
        else:
            index_url = self._url.rstrip("/") + "/index.json"
            data_bytes = urlopen(index_url).read()
        data = loads(data_bytes.decode("utf-8"))
        objs = data["items"]
        self._index_timestamp = time.time()
        packages = [Package.fromcsobj(p) for p in objs]
        self._packages = {p.id: p for p in packages}
        langs: Dict[str, List[Package]] = defaultdict(list)
        for k, package in self._packages.items():
            langs[package.language].append(package)
        tasks: Dict[str, List[Package]] = defaultdict(list)
        for k, package in self._packages.items():
            tasks[package.task].append(package)
        collections: List[Collection] = []
        for lang, children in langs.items():
            name1 = Locale(lang).getDisplayLanguage()
            try:
                name2 = isoLangs[lang]["name"]
            except Exception:
                name2 = None
            if name1 and name1 != lang:
                name = name1
            elif name2:
                name = name2
            else:
                name = lang
            id = "{}{}".format(Downloader.LANG_PREFIX, lang)
            name = "{:<20} packages and models".format(name)
            collections.append(Collection(id=id, name=name, children=children))
        for task, children in tasks.items():
            id = "{}{}".format(Downloader.TASK_PREFIX, task)
            collections.append(Collection(id=id, name=task, children=children))
        self._collections = {c.id: c for c in collections}
        for collection in self._collections.values():
            for i, child_id in enumerate(collection.children):
                if child_id in self._packages:
                    collection.children[i] = self._packages[child_id]
                if child_id in self._collections:
                    collection.children[i] = self._collections[child_id]
        for collection in self._collections.values():
            packages_map: Dict[str, Package] = {}
            queue: List[Any] = [collection]
            for child in queue:
                if isinstance(child, Collection):
                    queue.extend(child.children)
                else:
                    packages_map[child.id] = child
            collection.packages = packages_map.values()
        self._status_cache.clear()

    def index(self) -> Any:
        self._update_index()
        return self._index

    def info(self, id: str) -> Any:
        if id in self._packages:
            return self._packages[id]
        if id in self._collections:
            return self._collections[id]
        self._update_index()
        if id in self._packages:
            return self._packages[id]
        if id in self._collections:
            return self._collections[id]
        raise ValueError("Package %r not found in index" % id)

    def get_collection(self, lang: Optional[str] = None, task: Optional[str] = None) -> Collection:
        if lang:
            id = "{}{}".format(Downloader.LANG_PREFIX, lang)
        elif task:
            id = "{}{}".format(Downloader.TASK_PREFIX, task)
        else:
            raise ValueError("You should pass either the task or the lang")
        try:
            return self.info(id)
        except ValueError:
            if lang:
                raise LanguageNotSupported("Language {} is not supported".format(id))
            raise TaskNotSupported("Task {} is not supported".format(id))

    def supported_language(self, lang: str) -> bool:
        try:
            self.get_collection(lang=lang)
            return True
        except LanguageNotSupported:
            return False

    def supported_languages(self, task: Optional[str] = None) -> List[str]:
        if task:
            collection = self.get_collection(task=task)
            return [isoLangs[x.id.split(".")[1]]["name"] for x in collection.packages]
        return [x.name.split()[0] for x in self.collections() if Downloader.LANG_PREFIX in x.id]

    def supported_languages_table(self, task: str, cols: int = 3) -> str:
        languages = self.supported_languages(task)
        return pretty_list(languages)

    def supported_tasks(self, lang: Optional[str] = None) -> List[str]:
        if lang:
            collection = self.get_collection(lang=lang)
            return [x.id.split(".")[0] for x in collection.packages]
        return [x.name.split()[0] for x in self.collections() if Downloader.TASK_PREFIX in x.id]

    def _get_url(self) -> str:
        return self._url

    def _set_url(self, url: str) -> None:
        original_url = self._url
        try:
            self._update_index(url)
        except Exception:
            self._url = original_url
            raise

    url = property(_get_url, _set_url)

    def default_download_dir(self) -> Path:
        for dir_ in data_path:
            if dir_.exists() and is_writable(dir_):
                return dir_
        if sys.platform == "win32" and "APPDATA" in os.environ:
            homedir = Path(os.environ["APPDATA"])
        else:
            homedir = Path.home()
            if str(homedir) == "~/":
                raise ValueError("Could not find a default download directory")
        return homedir / "polyglot_data"

    def _get_download_dir(self) -> Optional[Path]:
        return self._download_dir

    def _set_download_dir(self, download_dir: Path) -> None:
        self._download_dir = download_dir
        self._status_cache.clear()

    download_dir = property(_get_download_dir, _set_download_dir)

    def _interactive_download(self) -> None:
        DownloaderShell(self).run()


class DownloaderShell:
    def __init__(self, dataserver: Downloader) -> None:
        self._ds: Downloader = dataserver

    def _simple_interactive_menu(self, *options: str) -> None:
        print("-" * 75)
        spc = (68 - sum(len(o) for o in options)) // (len(options) - 1) * " "
        print("  " + spc.join(options))
        print("-" * 75)

    def run(self) -> None:
        print("Polyglot Downloader")
        while True:
            self._simple_interactive_menu("d) Download", "l) List", " u) Update", "c) Config", "h) Help", "q) Quit")
            user_input = input("Downloader> ").strip()
            if not user_input:
                print()
                continue
            command = user_input.lower().split()[0]
            args = user_input.split()[1:]
            try:
                if command == "l":
                    print()
                    self._ds.list(self._ds.download_dir, header=False, more_prompt=True)
                elif command == "h":
                    self._simple_interactive_help()
                elif command == "c":
                    self._simple_interactive_config()
                elif command in ("q", "x"):
                    return
                elif command == "d":
                    self._simple_interactive_download(args)
                elif command == "u":
                    self._simple_interactive_update()
                else:
                    print("Command %r unrecognized" % user_input)
            except HTTPError as e:
                print("Error reading from server: %s" % e)
            except URLError as e:
                print("Error connecting to server: %s" % e.reason)
            print()

    def _simple_interactive_download(self, args: List[str]) -> None:
        if args:
            for arg in args:
                try:
                    self._ds.download(arg, prefix="  ")
                except (IOError, ValueError) as e:
                    print(e)
            return
        while True:
            print()
            print("Download which package (l=list; x=cancel)?")
            user_input = input("  Identifier> ")
            if user_input.lower() == "l":
                self._ds.list(self._ds.download_dir, header=False, more_prompt=True, skip_installed=True)
                continue
            elif user_input.lower() in ("x", "q", ""):
                return
            elif user_input:
                for id in user_input.split():
                    try:
                        self._ds.download(id, prefix="  ")
                    except (IOError, ValueError) as e:
                        print(e)
                break

    def _simple_interactive_update(self) -> None:
        while True:
            stale_packages: List[Tuple[str, str]] = []
            for info in sorted(self._ds.packages(), key=str):
                if self._ds.status(info) == self._ds.STALE:
                    stale_packages.append((info.id, info.name))
            print()
            if stale_packages:
                print("Will update following packages (o=ok; x=cancel)")
                for pid, pname in stale_packages:
                    name = textwrap.fill("-" * 27 + pname, 75, subsequent_indent=27 * " ")[27:]
                    print("  [ ] %s %s" % (pid.ljust(20, "."), name))
                print()
                user_input = input("  Identifier> ")
                if user_input.lower() == "o":
                    for pid, pname in stale_packages:
                        try:
                            self._ds.download(pid, prefix="  ")
                        except (IOError, ValueError) as e:
                            print(e)
                    break
                elif user_input.lower() in ("x", "q", ""):
                    return
            else:
                print("Nothing to update.")
                return

    def _simple_interactive_help(self) -> None:
        print()
        print("Commands:")
        print("  d) Download a package or collection   u) Update out of date packages")
        print("  l) List packages & collections      h) Help")
        print("  c) View & Modify Configuration      q) Quit")

    def _show_config(self) -> None:
        print()
        print("Data Server:")
        print("  - URL: <%s>" % self._ds.url)
        print("  - %d Package Collections Available" % len(list(self._ds.collections())))
        print("  - %d Individual Packages Available" % len(list(self._ds.packages())))
        print()
        print("Local Machine:")
        print("  - Data directory: %s" % self._ds.download_dir)

    def _simple_interactive_config(self) -> None:
        self._show_config()
        while True:
            print()
            self._simple_interactive_menu("s) Show Config", "u) Set Server URL", "d) Set Data Dir", "m) Main Menu")
            user_input = input("Config> ").strip().lower()
            if user_input == "s":
                self._show_config()
            elif user_input == "d":
                new_dl_dir = input("  New Directory> ").strip().lower()
                if new_dl_dir in ("", "x", "q"):
                    print("  Cancelled!")
                elif Path(new_dl_dir).is_dir():
                    self._ds.download_dir = Path(new_dl_dir)
                else:
                    print("Directory %r not found!  Create it first." % new_dl_dir)
            elif user_input == "u":
                new_url = input("  New URL> ").strip().lower()
                if new_url in ("", "x", "q"):
                    print("  Cancelled!")
                else:
                    if not new_url.startswith("http://"):
                        new_url = "http://" + new_url
                    try:
                        self._ds.url = new_url
                    except Exception as e:
                        print("Error reading <%r>:\n  %s" % (new_url, e))
            elif user_input == "m":
                break


def unzip(filename: Path, root: Path, verbose: bool = True) -> None:
    for message in _unzip_iter(filename, root, verbose):
        if isinstance(message, ErrorMessage):
            raise Exception(message)


def _unzip_iter(filename: Path, root: Path, verbose: bool = True) -> Iterator[DownloaderMessage]:
    if verbose:
        sys.stdout.write("Unzipping %s" % filename.name)
        sys.stdout.flush()
    try:
        zf = zipfile.ZipFile(str(filename))
    except zipfile.BadZipFile:
        yield ErrorMessage(filename, "Error with downloaded zip file")
        return
    except Exception as e:
        yield ErrorMessage(filename, e)
        return
    namelist = zf.namelist()
    dirlist: Set[str] = set()
    for x in namelist:
        if x.endswith("/"):
            dirlist.add(x)
        else:
            dirlist.add(x.rsplit("/", 1)[0] + "/")
    filelist = [x for x in namelist if not x.endswith("/")]
    if not root.exists():
        root.mkdir()
    for dirname in sorted(dirlist):
        pieces = dirname[:-1].split("/")
        for i in range(len(pieces)):
            dirpath = root.joinpath(*pieces[: i + 1])
            if not dirpath.exists():
                dirpath.mkdir()
    for i, filename_ in enumerate(filelist):
        filepath = root.joinpath(*filename_.split("/"))
        with open(str(filepath), "wb") as outfile:
            try:
                contents = zf.read(filename_)
            except Exception as e:
                yield ErrorMessage(filename_, e)
                return
            outfile.write(contents)
        if verbose and (i * 10 // len(filelist) > (i - 1) * 10 // len(filelist)):
            sys.stdout.write(".")
            sys.stdout.flush()
    if verbose:
        print()


def _find_packages(root: Path) -> Iterator[Tuple[Any, zipfile.ZipFile, str]]:
    for xmlfile in sorted(root.rglob("*.xml")):
        relpath = str(xmlfile.parent.relative_to(root))
        zipfilename = xmlfile.with_suffix(".zip")
        try:
            zf = zipfile.ZipFile(str(zipfilename))
        except Exception as e:
            raise ValueError("Error reading file %r!\n%s" % (zipfilename, e))
        yield xmlfile, zf, relpath


downloader: Downloader = Downloader()


def download_shell() -> None:
    DownloaderShell(downloader).run()


def update() -> None:
    downloader.update()


polyglot_path: Path = Path(data_path[0]).expanduser().resolve() / "polyglot_data"

resource_dir: Dict[str, str] = {
    "cw_embeddings": "embeddings2",
    "sgns_embeddings": "sgns2",
    "visualization": "tsne2",
    "wiki_vocab": "counts2",
    "sentiment": "sentiment2",
}


def locate_resource(name: str, lang: str, filter: Optional[Any] = None) -> Path:
    task_dir = resource_dir.get(name, name)
    package_id = "{}.{}".format(task_dir, lang)
    p = polyglot_path / task_dir / lang
    if not p.is_dir():
        if downloader.status(package_id) != downloader.INSTALLED:
            raise ValueError(
                "This resource is available in the index "
                "but not downloaded, yet. Try to run\n\n"
                "polyglot download {}".format(package_id)
            )
    return p / next(iter(p.iterdir())).name


@memoize
def load_embeddings(lang: str = "en", task: str = "embeddings", type: str = "cw") -> Embedding:
    src_dir = "_".join((type, task)) if type else task
    p = locate_resource(src_dir, lang)
    e = Embedding.load(p)
    if type == "cw":
        e.apply_expansion(CaseExpander)
        e.apply_expansion(DigitExpander)
    if type == "sgns":
        e.apply_expansion(CaseExpander)
    return e


@memoize
def load_vocabulary(lang: str = "en", type: str = "wiki") -> CountedVocabulary:
    src_dir = "{}_vocab".format(type)
    p = locate_resource(src_dir, lang)
    return CountedVocabulary.from_vocabfile(p)


@memoize
def load_ner_model(lang: str = "en", version: str = "2") -> Any:
    src_dir = "ner{}".format(version)
    p = locate_resource(src_dir, lang)
    fh = _open(p)
    try:
        return pickle.load(fh)
    except UnicodeDecodeError:
        fh.seek(0)
        return pickle.load(fh, encoding="latin1")


@memoize
def load_pos_model(lang: str = "en", version: str = "2") -> Dict[str, Any]:
    src_dir = "pos{}".format(version)
    p = locate_resource(src_dir, lang)
    fh = _open(p)
    return dict(np.load(fh))


@memoize
def load_morfessor_model(lang: str = "en", version: str = "2") -> Any:
    src_dir = "morph{}".format(version)
    p = locate_resource(src_dir, lang)
    file_handler = _open(p)
    tmp_file_ = NamedTemporaryFile(delete=False)
    tmp_file_.write(file_handler.read())
    tmp_file_.close()
    io = morfessor.MorfessorIO()
    model = io.read_any_model(tmp_file_.name)
    Path(tmp_file_.name).unlink()
    return model


@memoize
def load_transliteration_table(lang: str = "en", version: str = "2") -> Any:
    src_dir = "transliteration{}".format(version)
    p = locate_resource(src_dir, lang)
    file_handler = _open(p)
    return pickle.load(file_handler)


NER_ID_TAG: Dict[int, str] = {0: "O", 1: "I-PER", 2: "I-LOC", 3: "I-ORG"}

POS_TAG_ID: Dict[str, int] = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "AUX": 3,
    "CONJ": 4,
    "DET": 5,
    "INTJ": 6,
    "NOUN": 7,
    "NUM": 8,
    "PART": 9,
    "PRON": 10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM": 14,
    "VERB": 15,
    "X": 16,
}
POS_ID_TAG: Dict[int, str] = {v: k for k, v in POS_TAG_ID.items()}


class TaggerBase:
    PAD: str = "<PAD>"
    START: str = "<S>"
    END: str = "</S>"
    UNK: str = "<UNK>"

    def __init__(self, lang: str = "en") -> None:
        self.lang: str = lang
        self.embeddings: Embedding
        self.model: Any
        self.predictor: Callable[[np.ndarray], np.ndarray] = self._load_network()
        self.ID_TAG: Dict[Any, str] = {}
        self.add_bias: bool = True
        self.context: int = 2

    @staticmethod
    def ngrams(sequence: List[str], n: int) -> Iterator[List[str]]:
        seq = (n - 1) * [TaggerBase.PAD] + [TaggerBase.START] + sequence + [TaggerBase.END] + (n - 1) * [TaggerBase.PAD]
        for i in range(n, n + len(sequence)):
            yield seq[i - n : i + n + 1]

    def _load_network(self) -> Callable[[np.ndarray], np.ndarray]:
        raise NotImplementedError()

    def annotate(self, sent: Iterable[str]) -> Iterator[Tuple[str, str]]:
        preds: List[str] = []
        words: List[str] = []
        for word, fv in self.sent2examples(sent):
            probs = self.predictor(fv)
            tags = probs.argsort()
            tag = self.ID_TAG[tags[-1]]
            words.append(word)
            preds.append(tag)
        return zip(words, preds)

    def sent2examples(self, sent: Iterable[str]) -> Iterator[Tuple[str, np.ndarray]]:
        words = [w if w in self.embeddings else TaggerBase.UNK for w in sent]
        ngrams = TaggerBase.ngrams(words, self.context)
        for word, ngram in zip(sent, ngrams):
            fv = np.array([self.embeddings[w] for w in ngram]).flatten()
            if self.add_bias:
                fv = np.hstack((fv, np.array(1)))
            yield word, fv


class NEChunker(TaggerBase):
    def __init__(self, lang: str = "en") -> None:
        super().__init__(lang=lang)
        self.ID_TAG = NER_ID_TAG

    def _load_network(self) -> Callable[[np.ndarray], np.ndarray]:
        self.embeddings = load_embeddings(self.lang, type="cw")
        self.embeddings.normalize_words(inplace=True)
        self.model = load_ner_model(lang=self.lang, version=2)
        first_layer, second_layer = self.model

        def predict_proba(input_: np.ndarray) -> np.ndarray:
            hidden = np.tanh(np.dot(first_layer, input_))
            hidden = np.hstack((hidden, np.ones((hidden.shape[0], 1))))
            output = (second_layer * hidden).sum(axis=1)
            output_ = 1.0 / (1.0 + np.exp(-output))
            return output_ / output_.sum()

        return predict_proba


class POSTagger(TaggerBase):
    def __init__(self, lang: str = "en") -> None:
        super().__init__(lang=lang)
        self.ID_TAG = POS_ID_TAG
        self.add_bias = False

    def _load_network(self) -> Callable[[np.ndarray], np.ndarray]:
        self.embeddings = load_embeddings(self.lang, type="cw")
        self.model = load_pos_model(lang=self.lang, version=2)

        def predict_proba(input_: np.ndarray) -> np.ndarray:
            hidden = np.tanh(np.dot(input_, self.model["W1"]) + self.model["b1"])
            output = np.dot(hidden, self.model["W2"]) + self.model["b2"]
            scores = np.exp(output)
            return scores / scores.sum()

        return predict_proba


@memoize
def get_pos_tagger(lang: str = "en") -> POSTagger:
    return POSTagger(lang=lang)


@memoize
def get_ner_tagger(lang: str = "en") -> NEChunker:
    return NEChunker(lang=lang)


class Breaker:
    def __init__(self, locale: str) -> None:
        self.locale: Locale = Locale(locale)
        self.breaker: Optional[Any] = None

    def transform(self, sequence: Sequence) -> Sequence:
        seq = Sequence(sequence.text)
        seq.idx = [0]
        for segment in sequence:
            offset = seq.idx[-1]
            self.breaker.setText(segment)
            seq.idx.extend([offset + x for x in self.breaker])
        return seq


class SentenceTokenizer(Breaker):
    def __init__(self, locale: str = "en") -> None:
        super().__init__(locale)
        self.breaker = BreakIterator.createSentenceInstance(self.locale)


class WordTokenizer(Breaker):
    def __init__(self, locale: str = "en") -> None:
        super().__init__(locale)
        self.breaker = BreakIterator.createWordInstance(self.locale)


class Transliterator:
    def __init__(self, source_lang: str = "en", target_lang: str = "en") -> None:
        self.source_lang: str = source_lang
        self.target_lang: str = target_lang
        self.decoder: Callable[[str], str] = self._decoder()
        self.encoder: Callable[[str], str] = self._encoder()

    def _decoder(self) -> Callable[[str], str]:
        if self.target_lang == "en":
            return Transliterator._dummy_coder
        weights = load_transliteration_table(self.target_lang)
        return Transliterator._transliterate_string(weights["decoder"])

    def _encoder(self) -> Callable[[str], str]:
        if self.source_lang == "en":
            return Transliterator._dummy_coder
        weights = load_transliteration_table(self.source_lang)
        return Transliterator._transliterate_string(weights["encoder"])

    @staticmethod
    def _dummy_coder(word: str) -> str:
        return word

    def transliterate(self, word: str) -> str:
        encoded_word = self.encoder(word)
        decoded_word = self.decoder(word)
        return decoded_word

    @staticmethod
    def _transliterate_string(
        weight: Dict[Any, float],
        ngram1: int = 6,
        ngram2: int = 6,
    ) -> Callable[[str], str]:
        def translate_string(word: str) -> str:
            unlimited5 = 99999
            word = word.lower().strip()
            best_source_string: List[str] = []
            best_target_string: List[str] = []
            best_string_cost: List[int] = []
            for i in range(len(word) + 1):
                best_source_string.append("")
                best_target_string.append("")
                best_string_cost.append(unlimited5)
            best_string_cost[0] = 0
            for i in range(1, len(word) + 1):
                for j in range(1, ngram1 + 1):
                    if i >= j:
                        piece = word[i - j : i]
                        for item in weight:
                            if item[0].strip() == piece:
                                vfinal = -log(weight[item])
                                if best_string_cost[i - j] < unlimited5:
                                    tmp_string_cost = best_string_cost[i - j] + vfinal
                                    if tmp_string_cost < best_string_cost[i]:
                                        tmp_source_string = best_source_string[i - j] + piece
                                        tmp_target_string = best_target_string[i - j] + item[1].strip()
                                        best_source_string[i] = tmp_source_string
                                        best_target_string[i] = tmp_target_string
                                        best_string_cost[i] = tmp_string_cost
            return best_target_string[len(word)]

        return translate_string


class BaseBlob(StringlikeMixin, BlobComparableMixin):
    def __init__(self, text: Any) -> None:
        if not isinstance(text, (str, bytes)):
            raise TypeError(
                "The `text` argument passed to `__init__(text)` must be a unicode string, not {}".format(type(text))
            )
        self.raw: str = text if isinstance(text, str) else text.decode("utf-8")
        self.string: str = self.raw
        self.__lang: Optional[Language] = None

    @cached_property
    def detected_languages(self) -> Detector:
        return Detector(self.raw, quiet=True)

    @property
    def language(self) -> Language:
        if self.__lang is None:
            self.__lang = self.detected_languages.language
        return self.__lang

    @language.setter
    def language(self, value: str) -> None:
        self.__lang = Language.from_code(value)

    @property
    def word_tokenizer(self) -> WordTokenizer:
        return WordTokenizer(locale=self.language.code)

    @property
    def words(self) -> "WordList":
        return self.tokens

    @cached_property
    def tokens(self) -> "WordList":
        seq = self.word_tokenizer.transform(Sequence(self.raw))
        return WordList(seq.tokens(), parent=self, language=self.language.code)

    def tokenize(self, tokenizer: Optional[Breaker] = None) -> "WordList":
        t = tokenizer if tokenizer is not None else self.word_tokenizer
        return WordList(t.transform(Sequence(self.raw)).tokens(), parent=self)

    @cached_property
    def polarity(self) -> float:
        scores = [w.polarity for w in self.words if w.polarity != 0]
        return sum(scores) / float(len(scores)) if scores else 0.0

    @cached_property
    def ne_chunker(self) -> NEChunker:
        return get_ner_tagger(lang=self.language.code)

    @cached_property
    def pos_tagger(self) -> POSTagger:
        return get_pos_tagger(lang=self.language.code)

    @cached_property
    def morpheme_analyzer(self) -> Any:
        return load_morfessor_model(lang=self.language.code)

    def transliterate(self, target_language: str = "en") -> "WordList":
        return WordList([w.transliterate(target_language) for w in self.words], language=target_language, parent=self)

    @cached_property
    def morphemes(self) -> "WordList":
        words, _score = self.morpheme_analyzer.viterbi_segment(self.raw)
        return WordList(words, language=self.language.code, parent=self)

    @cached_property
    def entities(self) -> List["Chunk"]:
        chunks: List[Chunk] = []
        words = list(self.words)
        prev_tag = "O"
        start = 0
        tag = "O"
        i = -1
        for i, (w, tag) in enumerate(self.ne_chunker.annotate(words)):
            if tag != prev_tag:
                if prev_tag != "O":
                    chunks.append(Chunk(words[start:i], start, i, tag=prev_tag, parent=self))
                start = i
                prev_tag = tag
        if prev_tag != "O" and i >= 0:
            chunks.append(Chunk(words[start : i + 1], start, i + 1, tag=prev_tag, parent=self))
        return chunks

    @cached_property
    def pos_tags(self) -> List[Tuple["Word", str]]:
        tagged_words: List[Tuple[Word, str]] = []
        for word, t in self.pos_tagger.annotate(self.words):
            word.pos_tag = t
            tagged_words.append((word, t))
        return tagged_words

    @cached_property
    def word_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for word in self.words:
            counts[word] += 1
        return counts

    def ngrams(self, n: int = 3) -> List["WordList"]:
        if n <= 0:
            return []
        return [WordList(self.words[i : i + n], parent=self) for i in range(len(self.words) - n + 1)]

    def detect_language(self) -> str:
        return self.language.code

    def _cmpkey(self) -> str:
        return self.raw

    def _strkey(self) -> str:
        return self.raw

    def __hash__(self) -> int:
        return hash(self._cmpkey())

    def __add__(self, other: Any) -> "BaseBlob":
        if isinstance(other, (str, bytes)):
            return self.__class__(self.raw + other)
        elif isinstance(other, BaseBlob):
            return self.__class__(self.raw + other.raw)
        raise TypeError("Operands must be either strings or {} objects".format(self.__class__.__name__))

    def split(self, sep: Optional[str] = None, maxsplit: int = -1) -> "WordList":
        return WordList(self._strkey().split(sep, maxsplit), parent=self)


class Word(str):
    def __new__(cls, string: str, language: Optional[str] = None, pos_tag: Optional[str] = None) -> "Word":
        return super().__new__(cls, string)

    def __init__(self, string: str, language: Optional[str] = None, pos_tag: Optional[str] = None) -> None:
        self.string: str = string
        self.pos_tag: Optional[str] = pos_tag
        self.__lang: Optional[str] = language

    def __repr__(self) -> str:
        return repr(self.string)

    def __str__(self) -> str:
        return self.string

    @cached_property
    def morpheme_analyzer(self) -> Any:
        return load_morfessor_model(lang=self.language)

    @cached_property
    def morphemes(self) -> "WordList":
        words, _score = self.morpheme_analyzer.viterbi_segment(self.string)
        return WordList(words, parent=self, language=self.language)

    @cached_property
    def detected_languages(self) -> Detector:
        return Detector(self.string, quiet=True)

    @property
    def language(self) -> str:
        if self.__lang is None:
            self.__lang = self.detected_languages.language.code
        return self.__lang

    @language.setter
    def language(self, value: str) -> None:
        self.__lang = value

    @property
    def vector(self) -> np.ndarray:
        embeddings = load_embeddings(lang=self.language, type="sgns", task="embeddings")
        return embeddings[self.string]

    @property
    def neighbors(self) -> List[str]:
        embeddings = load_embeddings(lang=self.language, type="sgns", task="embeddings")
        return embeddings.nearest_neighbors(self.string)

    @property
    def polarity(self) -> float:
        embeddings = load_embeddings(lang=self.language, type="", task="sentiment")
        return embeddings.get(self.string, [0])[0]

    def detect_language(self) -> str:
        return self.language

    def transliterate(self, target_language: str = "en") -> str:
        t = Transliterator(source_lang=self.language, target_lang=target_language)
        return t.transliterate(self.string)


class WordList(list):
    def __init__(
        self,
        collection: Iterable[Any],
        parent: Optional[Any] = None,
        language: str = "en",
    ) -> None:
        self._collection: List[Word] = [Word(w, language=language) for w in collection]
        self.parent: Optional[Any] = parent
        super().__init__(self._collection)

    def __str__(self) -> str:
        return str(self._collection)

    def __repr__(self) -> str:
        return "{}({!r})".format(self.__class__.__name__, self._collection)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return self.__class__(self._collection[key])
        return self._collection[key]

    def __iter__(self) -> Iterator[Word]:
        return iter(self._collection)

    def count(self, strg: str, case_sensitive: bool = False, *args: Any, **kwargs: Any) -> int:
        if not case_sensitive:
            return [word.lower() for word in self].count(strg.lower(), *args, **kwargs)
        return self._collection.count(strg, *args, **kwargs)

    def append(self, obj: Any) -> None:
        if isinstance(obj, (str, bytes)):
            self._collection.append(Word(obj))
        else:
            self._collection.append(obj)

    def extend(self, iterable: Iterable[Any]) -> "WordList":
        for e in iterable:
            self._collection.append(Word(e) if isinstance(e, (str, bytes)) else e)
        return self

    def upper(self) -> "WordList":
        return self.__class__([word.upper() for word in self])

    def lower(self) -> "WordList":
        return self.__class__([word.lower() for word in self])


class Chunk(WordList):
    def __init__(
        self,
        subsequence: Iterable[Any],
        start_index: int = 0,
        end_index: Optional[int] = None,
        tag: str = "",
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(collection=subsequence, parent=parent)
        self.start: int = start_index
        self.end: int = end_index if end_index is not None else len(self._collection) - 1
        self.tag: str = tag if tag else self.__class__.__name__

    def __repr__(self) -> str:
        return "{}({!r})".format(self.tag, self._collection)

    @cached_property
    def positive_sentiment(self) -> float:
        pos, _neg = self._sentiment()
        return pos

    @cached_property
    def negative_sentiment(self) -> float:
        _pos, neg = self._sentiment()
        return neg

    def _sentiment(self, distance: bool = True) -> Tuple[float, float]:
        sum_pos = 0.0
        sum_neg = 0.0
        text = self.parent
        entity_positions = range(self.start, self.end)
        non_entity_positions = set(range(len(text.words))).difference(entity_positions)
        if not distance:
            non_entity_polarities = np.array([text.words[i].polarity for i in non_entity_positions])
            sum_pos = float(sum(non_entity_polarities == 1))
            sum_neg = float(sum(non_entity_polarities == -1))
        else:
            polarities = np.array([w.polarity for w in text.words])
            polarized_positions = np.argwhere(polarities != 0).flatten()
            polarized_non_entity_positions = non_entity_positions.intersection(polarized_positions)
            sentence_len = len(text.words)
            for i in polarized_non_entity_positions:
                min_dist = min(abs(self.start - i), abs(self.end - i))
                if text.words[i].polarity == 1:
                    sum_pos += 1.0 - (min_dist - 1.0) / (2.0 * sentence_len)
                else:
                    sum_neg += 1.0 - (min_dist - 1.0) / (2.0 * sentence_len)
        return (sum_pos, sum_neg)


class Sentence(BaseBlob):
    def __init__(self, sentence: str, start_index: int = 0, end_index: Optional[int] = None) -> None:
        super().__init__(sentence)
        self.start: int = start_index
        self.end: int = end_index if end_index is not None else len(sentence) - 1

    @property
    def dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "start_index": self.start,
            "end_index": self.end,
            "entities": self.entities,
            "polarity": self.polarity,
        }


class Text(BaseBlob):
    def __init__(self, text: str) -> None:
        super().__init__(text)

    def __str__(self) -> str:
        if len(self.raw) > 1000:
            return "{}{}{}".format(self.raw[:500], "...", self.raw[-500:])
        return self.raw

    @property
    def sentences(self) -> List[Sentence]:
        return self._create_sentence_objects()

    @property
    def raw_sentences(self) -> List[str]:
        return [sentence.raw for sentence in self.sentences]

    @property
    def serialized(self) -> List[Dict[str, Any]]:
        return [sentence.dict for sentence in self.sentences]

    def to_json(self, *args: Any, **kwargs: Any) -> str:
        return json.dumps(self.serialized, *args, **kwargs)

    @property
    def json(self) -> str:
        return self.to_json()

    def _create_sentence_objects(self) -> List[Sentence]:
        sentence_objects: List[Sentence] = []
        sent_tokenizer = SentenceTokenizer(locale=self.language.code)
        seq = sent_tokenizer.transform(Sequence(self.raw))
        for start_index, end_index in zip(seq.idx[:-1], seq.idx[1:]):
            sent = seq.text[start_index:end_index].strip()
            if not sent:
                continue
            s = Sentence(sent, start_index=start_index, end_index=end_index)
            s.detected_languages = self.detected_languages
            sentence_objects.append(s)
        return sentence_objects


signal(SIGPIPE, SIG_DFL)


def _cli_vocab_counter(args: Any) -> None:
    if isinstance(args.input, TextFiles):
        v = CountedVocabulary.from_textfiles(args.input)
    else:
        v = CountedVocabulary.from_textfile(args.input)
    if args.min_count > 1:
        v = v.min_count(args.min_count)
    if args.most_freq > 0:
        v = v.most_frequent(args.most_freq)
    print(v)


def _cli_detect(args: Any) -> None:
    for l in args.input:
        if l.strip():
            _print("{:<20}{}".format(Detector(l).language.name, l.strip()))


def _cli_tag(tagger: TaggerBase, args: Any) -> None:
    for l in args.input:
        words = l.strip().split()
        line_annotations = ["{:<16}{:<5}".format(w, p) for w, p in tagger.annotate(words)]
        _print("\n".join(line_annotations))
        _print("")


def _cli_transliterate(args: Any) -> None:
    t = Transliterator(source_lang=args.lang, target_lang=args.target)
    for l in args.input:
        words = l.strip().split()
        line_annotations = ["{:<16}{:<16}".format(w, t.transliterate(w)) for w in words]
        _print("\n".join(line_annotations))
        _print("")


def _cli_morphemes(args: Any) -> None:
    morfessor_model = load_morfessor_model(lang=args.lang)
    for l in args.input:
        words = l.strip().split()
        morphemes = [(w, "_".join(morfessor_model.viterbi_segment(w)[0])) for w in words]
        line_annotations = ["{:<16}{:<5}".format(w, p) for w, p in morphemes]
        _print("\n".join(line_annotations))
        _print("")


def _cli_ner_chunk(args: Any) -> None:
    chunker = NEChunker(lang=args.lang)
    _cli_tag(chunker, args)


def _cli_pos_tag(args: Any) -> None:
    tagger = POSTagger(lang=args.lang)
    _cli_tag(tagger, args)


def _cli_cat(args: Any) -> None:
    for l in args.input:
        _print(l.strip())


def _cli_download(args: Any) -> None:
    dl = Downloader(server_index_url=args.server_index_url)
    if args.packages:
        for pkg_id in args.packages:
            rv = dl.download(
                info_or_id=pkg_id,
                download_dir=Path(args.dir) if args.dir else None,
                quiet=args.quiet,
                force=args.force,
                halt_on_error=args.halt_on_error,
            )
            if rv is False and args.halt_on_error:
                break
    else:
        dl.download(
            download_dir=Path(args.dir) if args.dir else None,
            quiet=args.quiet,
            force=args.force,
            halt_on_error=args.halt_on_error,
        )


def _cli_segment(args: Any) -> None:
    lang = args.lang
    w_tokenizer = WordTokenizer(locale=lang)
    s_tokenizer = SentenceTokenizer(locale=lang)
    if args.only_sent:
        for l in args.input:
            seq = Sequence(l)
            if not seq.empty():
                _print(s_tokenizer.transform(seq))
    elif args.only_word:
        for l in args.input:
            seq = Sequence(l)
            if not seq.empty():
                _print(w_tokenizer.transform(seq))
    else:
        for l in args.input:
            seq = Sequence(l)
            sents = s_tokenizer.transform(seq)
            words = w_tokenizer.transform(seq)
            for tokenized_sent in words.split(sents):
                if not tokenized_sent.empty():
                    _print(" ".join(tokenized_sent.tokens()))


def _remove_escape(text: str) -> str:
    return text.encode("utf8").decode("unicode-escape")


def _cli_debug(type_: Any, value: Any, tb: Any) -> None:
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb

        traceback.print_exception(type_, value, tb)
        print("\n")
        pdb.pm()


def main(argv: Optional[List[str]] = None) -> Any:
    parser = ArgumentParser("polyglot", conflict_handler="resolve")
    subparsers = parser.add_subparsers(title="tools", description="multilingual tools for all languages")
    parser.add_argument("--lang", default="detect", help="Language to be processed")
    parser.add_argument(
        "--delimiter", default="\n", help="Delimiter that separates documents, records or even sentences."
    )
    parser.add_argument("-l", "--log", dest="log", help="log verbosity level", default="INFO")
    parser.add_argument(
        "--debug", dest="debug", action="store_true", default=False, help="drop a debugger if an exception is raised."
    )

    detector = subparsers.add_parser("detect", help="Detect the language(s) used in text.")
    detector.set_defaults(func=_cli_detect)

    morph = subparsers.add_parser("morph")
    morph.set_defaults(func=_cli_morphemes)

    tokenizer = subparsers.add_parser("tokenize", help="Tokenize text into sentences and words.")
    group1 = tokenizer.add_mutually_exclusive_group()
    group1.add_argument(
        "--only-sent", default=False, action="store_true", help="Segment sentences without word tokenization"
    )
    group1.add_argument(
        "--only-word", default=False, action="store_true", help="Tokenize words without sentence segmentation"
    )
    tokenizer.set_defaults(func=_cli_segment)

    downloader_p = subparsers.add_parser("download", help="Download polyglot resources and models.")
    downloader_p.add_argument("packages", nargs="*", help="packages to be downloaded")
    downloader_p.add_argument("--dir", dest="dir", help="download package to directory DIR", metavar="DIR")
    downloader_p.add_argument("--quiet", dest="quiet", action="store_true", default=False, help="work quietly")
    downloader_p.add_argument(
        "--force", dest="force", action="store_true", default=False, help="download even if already installed"
    )
    downloader_p.add_argument(
        "--exit-on-error", dest="halt_on_error", action="store_true", default=False, help="exit if an error occurs"
    )
    downloader_p.add_argument("--url", dest="server_index_url", default=None, help="download server index url")
    downloader_p.set_defaults(func=_cli_download)

    counter = subparsers.add_parser("count", help="Count words frequency in a corpus.")
    group1 = counter.add_mutually_exclusive_group()
    group1.add_argument("--min-count", type=int, default=1, help="Ignore all words that appear <= min_freq.")
    group1.add_argument("--most-freq", type=int, default=-1, help="Consider only the most frequent k words.")
    counter.set_defaults(func=_cli_vocab_counter)

    catter = subparsers.add_parser("cat", help="Print the contents of the input file to the screen.")
    catter.set_defaults(func=_cli_cat)

    ner = subparsers.add_parser("ner", help="Named entity recognition chunking.")
    ner.set_defaults(func=_cli_ner_chunk)

    pos = subparsers.add_parser("pos", help="Part of Speech tagger.")
    pos.set_defaults(func=_cli_pos_tag)

    transliterator = subparsers.add_parser("transliteration", help="Rewriting the input in the target language script.")
    transliterator.add_argument("--target", default="en", help="Language code of the target language.")
    transliterator.set_defaults(func=_cli_transliterate)

    subparsers.add_parser("sentiment", help="Classify text to positive and negative polarity.")

    for name, subparser in parser._subparsers._group_actions[0].choices.items():
        if name == "download":
            continue
        subparser.add_argument("--input", nargs="*", type=TextFile, default=[TextFile(sys.stdin.fileno())])

    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log.upper(), format=LOGFORMAT)

    if args.debug:
        sys.excepthook = _cli_debug

    language_agnostic = {_cli_detect, _cli_vocab_counter}

    if args.func != _cli_download:
        if len(args.input) > 1:
            args.input = TextFiles(args.input)
        else:
            args.input = args.input[0]
        if args.lang == "detect" and args.func not in language_agnostic:
            header = 4096
            text = args.input.peek(header)
            lang = Detector(text).language
            args.lang = lang.code
            logger.info("Language {} is detected while reading the first {} bytes.".format(lang.name, lang.read_bytes))
        args.delimiter = _remove_escape(args.delimiter)
        args.input.delimiter = args.delimiter

    return args.func(args)


__all__ = [
    "Sequence",
    "TokenSequence",
    "TextFile",
    "TextFiles",
    "Detector",
    "Language",
    "UnknownLanguage",
    "CountedVocabulary",
    "OrderedVocabulary",
    "VocabularyBase",
    "Embedding",
    "CaseExpander",
    "DigitExpander",
    "Downloader",
    "Package",
    "Collection",
    "downloader",
    "load_embeddings",
    "load_vocabulary",
    "load_ner_model",
    "load_pos_model",
    "load_morfessor_model",
    "load_transliteration_table",
    "NEChunker",
    "POSTagger",
    "get_pos_tagger",
    "get_ner_tagger",
    "SentenceTokenizer",
    "WordTokenizer",
    "Transliterator",
    "BaseBlob",
    "Word",
    "WordList",
    "Chunk",
    "Sentence",
    "Text",
    "cached_property",
    "memoize",
    "__author__",
    "__email__",
    "__version__",
]


if __name__ == "__main__":
    sys.exit(main())
