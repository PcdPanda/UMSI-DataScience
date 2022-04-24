'''
An old version was created by Dr. Yuhang Wang. Solutions created by Josh Gardner.
'''

import mrjob
from mrjob.job import MRJob
import re

WORD_RE = #regular expression

class WordCounter(MRJob):
    OUTPUT_PROTOCOL = mrjob.protocol.TextProtocol
   
    def mapper(self, _, line):
      try:
       # # +++your code here+++
        bill_type = # bill type
        title = # title
        if title is not None:
            matches = WORD_RE.findall(title)
            # # +++your code here+++
      except:
        pass
     
       
    def combiner(self, key, counts):
        # +++your code here+++

    def reducer(self, key, counts):
        # +++your code here+++
      


if __name__ == '__main__':
    WordCounter.run()
