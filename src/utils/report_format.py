"""Highlight best numeric values in Markdown metric tables."""
def highlight_table_bests(lines):
    output=list(lines);i=0
    while i+1<len(output):
        if not output[i].startswith('|') or not output[i+1].startswith('|---'):
            i+=1;continue
        header=[x.strip().lower() for x in output[i].strip('|').split('|')]
        end=i+2
        while end<len(output) and output[end].startswith('|'):end+=1
        rows=[[x.strip().replace('**','') for x in line.strip('|').split('|')] for line in output[i+2:end]]
        if len(rows)>1:
            for column,title in enumerate(header):
                try:values=[float(row[column].rstrip('%')) for row in rows]
                except (ValueError,IndexError):continue
                higher=any(word in title for word in ['fps','throughput','coverage','matched gt'])
                best=(max if higher else min)(values)
                for row,value in zip(rows,values):
                    if value==best:row[column]='**'+row[column]+'**'
            output[i+2:end]=['| '+' | '.join(row)+' |' for row in rows]
        i=end
    return output
