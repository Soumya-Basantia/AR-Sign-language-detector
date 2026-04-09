import os

dirs = [d for d in os.listdir('data/sequences') if os.path.isdir(os.path.join('data/sequences', d))]
for d in sorted(dirs):
    count = len([f for f in os.listdir(os.path.join('data/sequences', d)) if f.endswith('.npy')])
    print(f'{d}: {count} sequences')