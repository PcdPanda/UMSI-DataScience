import mrjob
from mrjob.job import MRJob 
from mrjob.step import MRStep
import re
WORD_RE = re.compile(r"\b[A-Za-z]+\b")

class MRMostUsedWords(MRJob): 
   OUTPUT_PROTOCOL = mrjob.protocol.TextProtocol 
   def mapper_get_words(self, _, line): 
        try:
            for part in line.split("\t"):
                for word in WORD_RE.findall(part):
                    if len(word) >= 4:
                        yield word.lower(), 1
        except Exception:
            pass

   def combiner_count_words(self, word, counts): 
       yield word, sum(counts)
 
   def reducer_count_words(self, word, counts): 
       yield len(word), (sum(counts), word)
 
   def reducer_find_max_word(self, length, word_count_pairs): 
        counts, word = max(word_count_pairs)
        try:
            yield str(length), "{}\t{}".format(word, counts)
        except Exception:
            pass
 
   def steps(self): 
       return [ 
           MRStep(mapper=self.mapper_get_words, 
                  combiner=self.combiner_count_words, 
                  reducer=self.reducer_count_words), 
           MRStep(reducer=self.reducer_find_max_word) 
       ] 
 
 
if __name__ == "__main__": 
   MRMostUsedWords.run() 