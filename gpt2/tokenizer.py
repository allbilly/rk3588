"""GPT-2 byte BPE, including Unicode pre-tokenization, using only the stdlib."""
import json
from pathlib import Path
import unicodedata


def byte_symbols():
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    symbols = values[:]
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            symbols.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, symbols)))


def character_class(char):
    category = unicodedata.category(char)
    if category.startswith("L"):
        return "letter"
    if category.startswith("N"):
        return "number"
    return "other"


def whitespace(char):
    # Python includes four ASCII information separators in isspace(); GPT-2's
    # regex uses Unicode White_Space, which excludes those byte values.
    return char.isspace() and char not in "\x1c\x1d\x1e\x1f"


def pieces(text):
    """Equivalent to GPT-2's Unicode regex, without the third-party regex module."""
    index = 0
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while index < len(text):
        start = index
        match = next((suffix for suffix in contractions if text.startswith(suffix, index)), None)
        if match is not None:
            index += len(match)
        elif whitespace(text[index]) and not (text[index] == " " and index + 1 < len(text) and not whitespace(text[index + 1])):
            while index < len(text) and whitespace(text[index]):
                index += 1
            # The greedy \s+(?!\S) branch leaves the last whitespace before a
            # nonspace for the next regex branch whenever two or more exist.
            if index < len(text) and index - start > 1:
                index -= 1
        else:
            if text[index] == " ":
                index += 1
            category = character_class(text[index])
            index += 1
            while index < len(text) and not whitespace(text[index]) and character_class(text[index]) == category:
                index += 1
        yield text[start:index]


class Tokenizer:
    def __init__(self, directory):
        directory = Path(directory)
        self.vocabulary = json.loads((directory / "vocab.json").read_text())
        self.inverse = {value: key for key, value in self.vocabulary.items()}
        lines = (directory / "merges.txt").read_text().splitlines()
        self.ranks = {tuple(line.split()): rank for rank, line in enumerate(lines[1:]) if line}
        self.byte_encoder = byte_symbols()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.cache = {}
        self.eos = self.vocabulary["<|endoftext|>"]

    def bpe(self, word):
        if word in self.cache:
            return self.cache[word]
        symbols = tuple(word)
        while len(symbols) > 1:
            pairs = set(zip(symbols, symbols[1:]))
            pair = min(pairs, key=lambda value: self.ranks.get(value, float("inf")))
            if pair not in self.ranks:
                break
            merged = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and symbols[index:index + 2] == pair:
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = tuple(merged)
        self.cache[word] = symbols
        return symbols

    def encode(self, text):
        result = []
        # GPT-2's designated end-of-text string is a special token.
        sections = text.split("<|endoftext|>")
        for index, section in enumerate(sections):
            if index:
                result.append(self.eos)
            for piece in pieces(section):
                word = "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
                result.extend(self.vocabulary[symbol] for symbol in self.bpe(word))
        return result

    def decode(self, tokens):
        word = "".join(self.inverse[token] for token in tokens)
        return bytes(self.byte_decoder[symbol] for symbol in word).decode("utf-8", errors="replace")
