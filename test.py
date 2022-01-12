# from libpysal.weights import lat2W
# import libpysal
# import numpy as np
# from inequality.gini import Gini_Spatial


# w = lat2W(4,5)
# print(w.neighbors)


# f=libpysal.io.open(libpysal.examples.get_path("mexico.csv"))
# # print(type(f))
# # print(dir(f))
# print(f.header)
# vnames=["pcgdp%d"%dec for dec in range(1940,2010,10)]
# y=np.transpose(np.array([f.by_col[v] for v in vnames]))

# print(len(y[:,0]))
# print(len(y[0]))

# regimes=np.array(f.by_col('hanson98'))
# print(regimes)
# # print(regimes)
# # w = libpysal.weights.block_weights(regimes)
# # print(dir(w))
# # np.random.seed(12345)

# gs = Gini_Spatial(y[:,0],w)
# print(gs.p_sim)
# gs.p_sim

f