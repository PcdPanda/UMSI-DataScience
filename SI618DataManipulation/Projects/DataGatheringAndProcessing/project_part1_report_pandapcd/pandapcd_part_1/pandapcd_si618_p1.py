from PreProcess import preprocess
from BackTest import BackTest
from Analysis import analyze, save

if __name__ == "__main__":
    dfs = preprocess()
    ret = BackTest(dfs) 
    save(*analyze(dfs, ret))