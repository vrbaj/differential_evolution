"""Compare explicit scalar configurations with the published v0.1.0 artifact.

This checks regression compatibility, not agreement with an external algorithm.
"""
from pathlib import Path
import io
import json
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = '''
import json, sys
sys.path.insert(0,sys.argv[1])
import differential_evolution as de
runs=[]
for seed in (0,1,7):
    for name in ('DE','jDE','SHADE','LSHADE'):
        options=dict(objective=de.sphere_function,bounds=[(-5.,5.)]*5,
                     population_size=30,max_evaluations=1001,seed=seed,
                     record_snapshots=True,snapshot_interval=7)
        if name=='DE':
            result=de.DifferentialEvolution(**options,mutation=de.Rand1(.8),
                crossover=de.BinomialCrossover(.9)).run()
        elif name=='jDE':
            components=de.jde_rand_1_bin()
            result=de.DifferentialEvolution(**options,mutation=components.mutation,
                crossover=components.crossover).run()
        else:
            result=getattr(de,name)(**options).run()
        values=result.to_dict()
        values.pop('metadata')
        runs.append(values)
print(json.dumps(runs,sort_keys=True))
'''


def main():
    archive=subprocess.check_output(['git','archive','v0.1.0'],cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='defoundry-release-') as directory:
        with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
            handle.extractall(directory,filter='data')
        baseline=subprocess.check_output([sys.executable,'-I','-S','-c',PROGRAM,directory],cwd=directory,text=True)
        current=subprocess.check_output([sys.executable,'-I','-S','-c',PROGRAM,str(ROOT)],cwd=directory,text=True)
    assert json.loads(baseline)==json.loads(current)
    print('12 explicit DE/jDE/SHADE/L-SHADE scalar runs match v0.1.0 histories, populations and snapshots exactly')


if __name__=='__main__':
    main()
