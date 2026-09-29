"""Scene set for the soft-bag change (hotfix7): every optimize scene we have.

The change only touches Agent.optimize(), so online scenes play exactly as
before and are left out. Writes scenes_softA.json with
  * scenes_offtest.json   64  task A as specified: optimize, k=1, 1-2
                              containers, shelf / priority-container mixes
  * the optimize scenes of the four catalogue pools (pool / test / ktest /
    shelftest, 44), renamed <pool>:<scene> since test and shelftest share
    names
  * the optimize scenes of scenes_prepacked.json (task A with bags already
    loaded, 12)
"""
import json

POOLS = {'pool': 'scenes_pool.json', 'test': 'scenes_test.json',
         'ktest': 'scenes_ktest.json', 'shelftest': 'scenes_shelftest.json'}


def main():
    out = {}
    for k, v in json.load(open('scenes_offtest.json')).items():
        out[f'offtest:{k}'] = v
    for pool, f in POOLS.items():
        for k, v in json.load(open(f)).items():
            if v['agent']['optimize']:
                out[f'{pool}:{k}'] = v
    for k, v in json.load(open('scenes_prepacked.json')).items():
        if v['agent']['optimize']:
            out[f'prepacked:{k}'] = v
    json.dump(out, open('scenes_softA.json', 'w'))
    groups = {}
    for k in out:
        groups[k.split(':')[0]] = groups.get(k.split(':')[0], 0) + 1
    print('wrote', len(out), 'scenes', groups)


if __name__ == '__main__':
    main()
