"""A deterministic illustration of later self-consistency over reasoning traces."""

# Reading guide: follow the named helpers in data-flow order, then inspect the
# assertions at the bottom. Change one toy input at a time and rerun the file.
from collections import Counter
def answer(trace): return int(trace.rsplit('=',1)[1].strip())
def main():
 traces=['3 + 4 = 7','start at 3, add 4: 7 = 7','3 + 4 = 8','three plus four gives 7 = 7','3 + 4 = 7']
 votes=Counter(map(answer,traces)); winner,count=votes.most_common(1)[0]
 print('votes:',dict(votes),'winner:',winner)
 assert winner==7 and count==4
 print('ok: a majority of independently formatted traces yields the known answer')
if __name__=='__main__':main()

