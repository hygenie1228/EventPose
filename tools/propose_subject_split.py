"""Propose participant-disjoint splits using only small canonical GT metadata."""
import json
from collections import Counter
from pathlib import Path
import h5py
ROOT=Path(__file__).absolute().parents[1]
OUT=ROOT/'exp/261023/subject_split'

def main():
    rows=[];parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for entry in json.loads((ROOT/'exp/261023/all_gt_processed/session_inputs.json').read_text()):
        name=entry['session']
        with h5py.File(ROOT/'dataset'/name/'RGBD/device0_mamma.h5','r') as f:
            ids=[x.decode() if isinstance(x,bytes) else str(x) for x in f['metadata/session_participant_ids'][()]]
        assert ids[0]==name.split('_')[3]
        for x in ids:parent[find(x)]=find(ids[0])
        rows.append({'session':name,'participant_ids':ids,'primary_id':ids[0],'action':name.split('_')[4],'source_I_available':bool(entry['source_I'])})
    groups={}
    for r in rows:groups.setdefault(find(r['primary_id']),[]).append(r)
    groups=sorted(groups.values(),key=lambda g:min(int(x) for r in g for x in r['participant_ids']))
    def choose(candidates,target):
        dp={0:[]}
        for i in candidates:
            for n,ix in list(dp.items())[::-1]:
                new=n+len(groups[i])
                if new<=target and new not in dp:dp[new]=ix+[i]
        n=min(dp,key=lambda n:abs(n-target))
        return set(dp[n])
    def plan(three):
        remaining=set(range(len(groups)))
        test=choose(sorted(remaining),29 if three else 57);remaining-=test
        val=choose(sorted(remaining),28) if three else set();remaining-=val
        result={}
        for key,ix in [('train',remaining),('val',val),('test',test)]:
            if not ix and key=='val':continue
            rr=[r for i in sorted(ix) for r in groups[i]]
            result[key]={'sessions':len(rr),'subjects':len({x for r in rr for x in r['participant_ids']}),'primary_subjects':len({r['primary_id'] for r in rr}),'subject_ids':sorted({x for r in rr for x in r['participant_ids']},key=int),'action_counts':dict(Counter(r['action'] for r in rr)),'source_I_sessions':sum(r['source_I_available'] for r in rr),'session_names':[r['session'] for r in rr]}
        sets=[set(v['subject_ids']) for v in result.values()]
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
        assert sum(v['sessions'] for v in result.values())==len(rows)
        return result
    report={'count_basis':'GT metadata/session_participant_ids; identities not inferred from images','sessions':len(rows),'subjects':len(parent),'primary_subjects':len({r['primary_id'] for r in rows}),'interaction_only_ids':sorted(set(parent)-{r['primary_id'] for r in rows},key=int),'connected_groups':len(groups),'action_counts':dict(Counter(r['action'] for r in rows)),'train_test':plan(False),'train_val_test':plan(True),'sessions_metadata':rows}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'split_proposal.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ['sessions_metadata','train_test','train_val_test']})
    for k in ['train_test','train_val_test']:
        print(k)
        for name,d in report[k].items():print(name,{x:y for x,y in d.items() if x!='session_names'})
if __name__=='__main__':main()
