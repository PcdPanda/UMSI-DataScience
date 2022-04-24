import mrjob
from mrjob.job import MRJob 
import re
WORD_RE = re.compile(r"\b[A-Za-z]+\b")

class MRMostUsedWords(MRJob): 
   OUTPUT_PROTOCOL = mrjob.protocol.TextProtocol

   def mapper(self, _, line):
      try:
         line = line.split("\t")
         year = line[-4]
         for part in line:
            for word in WORD_RE.findall(part):
               if len(word) >= 4:
                  yield "{}\t{}".format(year, word.lower()), 1
      except Exception:
         pass

   def combiner(self, key, count):
      yield key, sum(count)

   def reducer(self, key, count):
      yield key, str(sum(count))
 
if __name__ == "__main__": 
   MRMostUsedWords.run() 