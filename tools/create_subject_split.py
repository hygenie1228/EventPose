"""Create a reproducible 20-ID test split without participant leakage."""
import json,random
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).absolute().parents[1]
def main():
 r=json.loads((ROOT/'exp/261023/subject_split/split_proposal.json').read_text());rows=r['sessions_metadata'];parent={}
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 for row in rows:
  for x in row['participant_ids']:parent[find(x)]=find(row['primary_id'])
 groups={}
 for row in rows:groups.setdefault(find(row['primary_id']),[]).append(row)
 groups=list(groups.values());rng=random.Random(261023);best=None
 for trial in range(300):
  order=list(range(len(groups)));rng.shuffle(order);dp={(0,0):[]}
  for i in order:
   count=len({x for row in groups[i] for x in row['participant_ids']})
   for (n,s),selected in list(dp.items()):
    key=(n+count,s+len(groups[i]))
    if key[0]<=20 and key[1]<=70 and key not in dp:dp[key]=selected+[i]
  for (n,s),selected in dp.items():
   if n!=20:continue
   test=[row for i in selected for row in groups[i]];c=Counter(x['action'] for x in test);total=Counter(x['action'] for x in rows)
   score=abs(s-64)*10+sum(abs(c[k]-v*s/283) for k,v in total.items())+abs(sum(x['source_I_available'] for x in test)-171*s/283)
   if best is None or score<best[0]:best=(score,test)
 test={x['session'] for x in best[1]};result={'seed':261023,'basis':'GT metadata/session_participant_ids; participant-disjoint, co-occurring IDs stay together','train':{},'test':{}}
 for name,select in [('train',[x for x in rows if x['session'] not in test]),('test',[x for x in rows if x['session'] in test])]:
  result[name]={'subject_ids':sorted({i for x in select for i in x['participant_ids']},key=int),'sessions':sorted(x['session'] for x in select)}
  result[name]['num_subjects']=len(result[name]['subject_ids']);result[name]['num_sessions']=len(select)
 assert result['test']['num_subjects']==20
 assert not set(result['train']['subject_ids'])&set(result['test']['subject_ids'])
 assert len(set(result['train']['sessions'])|set(result['test']['sessions']))==283
 (ROOT/'exp/261023/subject_split/split_metadata.json').write_text(json.dumps(result,indent=2)+'\n')
 (ROOT/'etri_processed/split.json').write_text(json.dumps({k:result[k]['sessions'] for k in ('train','test')},indent=2)+'\n')
 for name in ['train','test']:print(name,result[name]['num_subjects'],result[name]['num_sessions'],result[name]['subject_ids'])
if __name__=='__main__':main()
