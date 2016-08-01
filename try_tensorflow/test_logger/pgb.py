from tqdm import tqdm
from time import sleep
from tqdm import trange

# pbar = tqdm(['a', 'b', 'c', 'd'])
# for char in pbar:
#     pbar.set_description("Processing %s" % char)
#     sleep(1)


# outbar = tqdm(range(10))
#
# for i in outbar:
#     innerbar = tqdm(range(20), leave=False)
#     for j in innerbar:
#         # print('\n', i, j)
#         sleep(1)
#     print(i)

# with tqdm(total=100) as pbar:
#     for i in range(10):
#         pbar.update(10)
#         sleep(1)

for j in trange(5, desc='2nd loop'):
    for k in trange(100, desc='3nd loop'):
        sleep(0.1)
    print('It\'s nice')
