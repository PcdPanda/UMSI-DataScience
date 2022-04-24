import mrjob
from mrjob.job import MRJob
import re
WORD_RE = re.compile(r"\b[A-Za-z]+\b") #regular expression
class WordCounter(MRJob):
    OUTPUT_PROTOCOL = mrjob.protocol.TextProtocol
    def mapper(self, _, line):
        try:
            bill_type = line.split("\t")[1]# bill type
            title = line.split("\t")[2]# title
            if title is not None:
                matches = WORD_RE.findall(title)
                for word in matches:
                    if len(word) >= 4 and len(word) <= 15:
                        yield [bill_type, word.lower()], 1
        except Exception:
            pass
     
       
    def combiner(self, key, counts):
        yield key, sum(counts)

    def reducer(self, key, counts):
        yield "{}\t{}".format(key[0], key[1]), str(sum(counts))


if __name__ == '__main__':
    WordCounter.run()
