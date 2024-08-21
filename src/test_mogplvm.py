# Tests with partial coverage for the MOGPLVM

import pytest
import torch
from src.mogplvm import MOGPLVM
from src.utils.tensor_utils import invert_covariance_matrix, make_grid
from src.data import Dataset, ObservedComponents, SpectralData

@pytest.fixture(name = "bass")
def bamm_fixture():
    torch.manual_seed(1234)
    torch.set_default_dtype(torch.float64)
    bass = MOGPLVM(
        beta=torch.ones(10)*10,
        gamma=torch.ones(1)/10,
        sigma2=torch.Tensor([0.01]),
        sigma2_s=torch.ones(1)*10,
        v_x = torch.randn(5, 10),
        v_l = torch.linspace(-10,10,2).reshape(-1,1)
    )
    return bass

@pytest.fixture(name="dataset")
def dataset_fixture():
    spectral_data = SpectralData(
        torch.arange(10).reshape(-1,1),
        torch.randn(10,10),
    )
    r = torch.rand(10,1)
    components = ObservedComponents(torch.hstack((r, 1-r)))
    return Dataset(spectral_data, components, torch.randn(10,10), torch.ones(10,1))

def compare(a,b):
    return torch.all(torch.isclose(a,b))

def kernel_between_points(x1,x2,length_scale, variance, ):
    diff = x1 - x2
    scaled_L2_diff = (diff**2/length_scale).sum()
    return variance*torch.exp(-scaled_L2_diff/2)

class TestBAMM:

    def test_ELBO(self, bass, dataset):
        bass.elbo([dataset])
    
    def test_kernel(self, bass: MOGPLVM):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1234)
        x1 = torch.randn(100, 10)
        x2 = torch.randn(50,10)
        weights = torch.abs(torch.randn(10))
        scale = torch.randn(1)
        expected_results = torch.zeros(100,50)
        for i in range(100):
            for j in range(50):
                for k in range(10):
                    expected_results[i,j] += (x1[i,k] - x2[j,k])**2/weights[k]
        expected_results = scale*torch.exp(-1/2*expected_results)
        assert torch.all(expected_results - bass.ard_kernel(x1,x2,weights,scale) < 1e-12)

    def test_get_K_vv(self, bass: MOGPLVM):
        torch.set_default_dtype(torch.float64)
        num_inducing = bass.v.shape[0]
        length_scale = torch.hstack((
            bass.beta,
            bass.gamma
        ))
        correct = torch.zeros(num_inducing,num_inducing)
        for i in range(num_inducing):
            for j in range(num_inducing):
                correct[i,j] = kernel_between_points(bass.v[i], bass.v[j], length_scale,  bass.sigma2_s,)

        K = bass.get_K_vv() 
        assert compare(K, correct)
            
    def test_xi_0(self, bass, dataset):
        xi0 = bass.get_xi_0(dataset)
        assert xi0 == bass.sigma2_s*((dataset.get_r()**2).sum())*10

    #TODO
    # def test_xi1(self,bass: BASS, dataset):
    #     N = dataset.num_data_points
    #     M = dataset.num_wavelengths
    #     L = bass.num_inducing_points
    #     wavelengths = dataset.spectral_data.wavelengths
    #     expected_xi1 = torch.zeros(N,M,L)
    #     det_beta = bass.beta.prod()
    #     for i in range(N):
    #         scale = bass.beta + dataset.Sigma_x[i]
    #         det_scale = scale.prod()
    #         for j in range(M):
    #             for k in range(L):
    #                 expected_xi1[i,j,k] = bass.sigma2_s*torch.sqrt(det_beta/det_scale)*kernel_between_points(dataset.mu_x[i], bass.v[k, :-1], scale,1)*kernel_between_points(wavelengths[j], bass.v[k, -1], bass.gamma, 1 )
    #     xi1 = bass.get_xi_1(dataset)
    #     assert torch.all(torch.isclose(xi1, expected_xi1))       

    # def test_get_psi1_x(self, bamm: BASS):
    #     N = bamm.num_data_points
    #     L_x = bamm.num_inducing_points_in_latent_space
    #     expected_psi1_x = torch.zeros(N,L_x)
    #     det_beta = bamm.beta.prod()
    #     for i in range(N):
    #         scale = bamm.beta + bamm.Sigma_x[i]
    #         det_scale = scale.prod()
    #         for k in range(L_x):
    #             expected_psi1_x[i,k] = torch.sqrt(det_beta/det_scale)*kernel_between_points(bamm.mu_x[i], bamm.v_x[k], scale,1)
    #     psi_1_x = bamm.get_psi_1_x()
    #     assert torch.all(torch.isclose(psi_1_x, expected_psi1_x))

    # def test_get_xi(
    #         self,
    #         bass: BASS,
    # ):  
    #     D = torch.randn(bass.num_data_points, 10)
    #     R = torch.randn(bass.num_data_points, bass.num_mixture_components)
    #     wavelengths = torch.arange(10).reshape(-1,1)
    #     psi1 = bass.get_psi_1(wavelengths)
    #     expected_results = torch.zeros(bass.num_mixture_components*bass.num_inducing_points)
    #     for i in range(bass.num_data_points):
    #         for j in range(wavelengths.shape[0]):
    #             for c in range(bass.num_mixture_components):
    #                 for k in range(bass.num_inducing_points):
    #                     expected_results[c*bass.num_inducing_points + k] +=  D[i,j] * R[i,c] * psi1[i, j, k]
    #     assert compare(expected_results, bass.get_xi_1(D,R, wavelengths))

    # #TODO: THIS TEST SHOULD NOT PASS. There is an extra factor of -1/2 in the final term which should not be there and slowed me down for weeks. 
    # def test_get_psi_2(
    #                 self,
    #                 bamm: BASS
    #             ):
    #     torch.set_default_dtype(torch.float64)
    #     expected_result = torch.zeros(bamm.num_inducing_points*bamm.num_mixture_components, bamm.num_inducing_points*bamm.num_mixture_components)
    #     R = torch.randn(bamm.num_data_points,bamm.num_mixture_components)
    #     wavelengths = torch.linspace(-3,3, 5).reshape(-1,1)        
    #     psi2 = bamm.get_xi_2(R, wavelengths)
    #     for i, mu in enumerate(bamm.mu_x):
    #         for j, l in enumerate(wavelengths):
    #             for k1, v1 in enumerate(bamm.v):
    #                 for k2, v2 in enumerate(bamm.v):
    #                     for c1, r1 in enumerate(R[i]):
    #                         for c2, r2 in enumerate(R[i]):
    #                             expected_result[c1*bamm.num_inducing_points + k1,c2*bamm.num_inducing_points+k2] += r1*r2*bamm.sigma2_s[0]**2*kernel_between_points(l, v1[-1], bamm.gamma, 1) * kernel_between_points(l, v2[-1], bamm.gamma,1)*torch.sqrt(bamm.beta.prod()/(2*bamm.Sigma_x[i] + bamm.beta).prod()) * torch.exp(-1/4*((v1[:-1] - v2[:-1])**2/bamm.beta).sum()) * torch.exp(-1/2*((mu - (v1[:-1]+v2[:-1])/2)**2/(bamm.beta + 2*bamm.Sigma_x[i])).sum())
    #     assert compare(expected_result, psi2)
                                

    # def test_get_psi_2_x(
    #         self,
    #         bamm: BASS
    #     ):
    #     expected_result = torch.zeros(
    #                 bamm.mu_x.shape[0], 
    #                 bamm.v_x.shape[0], 
    #                 bamm.v_x.shape[0]
    #             )
    #     for i,mu in enumerate(bamm.mu_x):
    #         for j,v1 in enumerate(bamm.v_x):
    #             for k,v2 in enumerate(bamm.v_x):
    #                 expected_result[i,j,k] = torch.sqrt(bamm.beta.prod()/(2*bamm.Sigma_x[i] + bamm.beta).prod())*torch.exp(-1/4*((v1-v2)**2/bamm.beta).sum())*torch.exp(-((mu - (v1 + v2)/2)**2/(bamm.beta + 2*bamm.Sigma_x[i])).sum())
    #     psi_2_x = bamm.get_psi_2_x()
    #     assert compare(expected_result, psi_2_x)

    # def test_get_K_lvl(
    #         self,
    #         bamm: BASS
    #     ):
    #     wavelengths = torch.linspace(-3,3, 50).reshape(-1,1)
    #     expected_result = torch.zeros(50, bamm.v_l.shape[0])
    #     for i,w in enumerate(wavelengths):
    #         for j,vl in enumerate(bamm.v_l):
    #             expected_result[i,j] = kernel_between_points(w,vl, bamm.gamma, 1)
    #     K_lvl = bamm.get_K_lvl(wavelengths)
    #     assert compare(expected_result, K_lvl)
    
    # #TODO: proper testing 
    # def test_get_sample_mean(self, bamm:BASS):
    #     wavelengths = torch.linspace(-3,3, 20).reshape(-1,1)
    #     x = torch.ones(10,10)
    #     new_points = make_grid(x, wavelengths)
    #     K_vv = bamm.get_K_vv()
    #     K_vv_inv = invert_covariance_matrix(K_vv)
    #     K_xv =  torch.zeros(200, bamm.num_inducing_points)
    #     for i, point in enumerate(new_points):
    #         for j, v in enumerate(bamm.v):
    #             K_xv[i,j] = kernel_between_points(point, v, torch.concat([bamm.beta, bamm.gamma]), bamm.sigma2_s)
    #     expected = (K_xv @ K_vv_inv @ bamm.mu_u.T).reshape(10, 20, bamm.num_mixture_components)
    #     sample_mean = bamm.get_sample_mean(wavelengths, x)
    #     assert compare(expected, sample_mean)
        

    # def test_sample(self):
    #     raise NotImplementedError("I should get this one too")
    


    # #Note: This test may fail for some values of beta - I believe this is due to numerical problems 
    # def test_variance_posive(self, bamm: BASS):
    #     bamm._log_beta = torch.nn.Parameter(torch.ones(10)*10)
    #     bamm._log_gamma = torch.nn.Parameter(torch.ones(1)*10)
    #     psi_2 = bamm.get_xi_2()
    #     psi_0 = bamm.get_xi_0()

    #     K_c_vv = bamm.get_K_vv()
    #     K_c_vv_inv = invert_covariance_matrix(K_c_vv)

    #     #TODO: correct for different kernels in each component 
    #     K_vv = torch.block_diag(*[K_c_vv]*bamm.num_mixture_components)
    #     K_vv_inv = torch.block_diag(*[K_c_vv_inv]*bamm.num_mixture_components)

    #     K_vxvx = bamm.kernel(bamm.v[:, :-1],bamm.v[:, :-1], bamm.beta)
    #     K_vxvx_inv = torch.inverse(K_vxvx)
    #     psi_2_x = bamm.get_psi_2_x().sum(axis=0)

    #     assert psi_0 > torch.trace(K_vv_inv @ psi_2)

    # def test_get_K_vxxvx(self, bamm: BASS):
    #     expected_result = torch.zeros(bamm.v_x.shape[0], bamm.v_x.shape[0])
    #     for i, v1 in enumerate(bamm.v_x):
    #         for j, v2 in enumerate(bamm.v_x):
    #             expected_result[i, j] = kernel_between_points(v1,v2, bamm.beta, 1)
    #     K_vxvx = bamm.get_K_vxvx()   
    #     assert compare(K_vxvx, expected_result)

    # def test_get_K_vlvl(self, bamm: BASS):
    #     expected_result = torch.zeros(bamm.v_l.shape[0], bamm.v_l.shape[0])
    #     for i, v1 in enumerate(bamm.v_l):
    #         for j, v2 in enumerate(bamm.v_l):
    #             expected_result[i, j] = kernel_between_points(v1,v2, bamm.beta, 1)
    #     K_vlvl = bamm.get_K_vlvl()   
    #     assert compare(K_vlvl, expected_result)


    # def test_sigma_u(self, bamm):
    #     sigma_u = bamm.sigma_u

    # def test_mu_u(self, bamm):
    #     sigma_u = bamm.mu_u
