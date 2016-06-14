import re
import csv


class Reader:

    TOKEN_REGEX = re.compile(r'[A-Za-z]+|[0-9]+|[,.!?:]')

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for article in reader:
                title, url, content, authors, image, error, uuid = article
                if not uuid or len(uuid) == 0:
                    continue
                tokens = self._tokenize(content)
                yield uuid, tokens

    @classmethod
    def _tokenize(cls, text):
        tokens = re.findall(cls.TOKEN_REGEX, text)
        tokens = [x.lower() for x in tokens]
        return tokens
