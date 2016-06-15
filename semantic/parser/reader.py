import re
import csv


class Reader:

    TOKEN_REGEX = re.compile(r'[A-Za-z]+|[0-9]+|[,.!?:]')

    def __init__(self, filename):
        self._filename = filename

    def __iter__(self):
        with open(self._filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for article in reader:
                title, url, content, authors, image, error, uuid = article
                tokens = self._tokenize(content)
                try:
                    assert uuid and len(uuid) > 0
                    assert len(tokens) >= 300
                except AssertionError:
                    continue
                yield uuid, tokens

    @classmethod
    def _tokenize(cls, text):
        tokens = re.findall(cls.TOKEN_REGEX, text)
        tokens = [x.lower() for x in tokens]
        return tokens
