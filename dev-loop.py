from minian.cnmf import update_temporal_cvxpy
from minian.cnmf import update_spatial_perpx

def main():
	c, s, b, c0 = update_temporal_cvxpy(YrA, g, tn, **kwargs)

	res = update_spatial_perpx(y[h, w, :], alpha[h, w], sub[h, w, :], C_store, f)

if __name__ == '__main__':
	main()