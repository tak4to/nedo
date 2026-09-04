import json,sys,math
def load(t):
    return {r['scene']:r for r in json.load(open(f'ab_{t}.json'))}
base_tags=sys.argv[1].split(',')
bs=[load(t) for t in base_tags]
scenes=sorted(set.intersection(*[set(b) for b in bs]))
def bmean(s,k): return sum(b[s][k] for b in bs)/len(bs)
for tag in sys.argv[2].split(','):
    a=load(tag)
    ss=[s for s in scenes if s in a]
    d=[(a[s]['norm']-bmean(s,'norm'),s) for s in ss]
    dp=[a[s]['pct']-bmean(s,'pct') for s in ss]
    n=len(d); m=sum(x for x,_ in d)/n
    sd=math.sqrt(sum((x-m)**2 for x,_ in d)/(n-1))
    w=sum(1 for x,_ in d if x>1e-9); l=sum(1 for x,_ in d if x<-1e-9)
    print(f'{tag:12s} n={n} dnorm={m:+6.2f} (se {sd/math.sqrt(n):.2f})  dpct={sum(dp)/n:+5.2f}  W/L={w}/{l}')
    d.sort()
    print('   worst:', ' '.join(f'{s}{x:+.1f}' for x,s in d[:5]))
    print('   best :', ' '.join(f'{s}{x:+.1f}' for x,s in d[-5:]))
