import re
import csv


class Reader:

    TOKEN_REGEX = re.compile(r'[A-Za-z]+')

    def __init__(self, filename):
        self._filename = filename

    def __iter__(self):
        with open(self._filename, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            next(reader)
            for article in reader:
                uuid = article['bookmarkID']
                tokens = self._tokenize(article['content'])
                try:
                    assert uuid and len(uuid) > 0
                    assert tokens
                except AssertionError:
                    print('Skip invalid article')
                    continue
                yield uuid, tokens

    @classmethod
    def _tokenize(cls, text):
        tokens = re.findall(cls.TOKEN_REGEX, text)
        tokens = [x.lower() for x in tokens]
        return tokens
