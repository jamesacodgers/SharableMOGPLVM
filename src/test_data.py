# Tests with partial coverage of the data classes

import pytest
import torch

from src.data import ObservedComponents, SpectralData, Images


class TestSpectralData:
    def test_validate_inputs_correct(self):
        with pytest.raises(ValueError):
            bad_wavelengths = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
            SpectralData(bad_wavelengths, bad_wavelengths)

    def test_waveleghts_and_spectra_match(self):
        with pytest.raises(ValueError):
            wavelengths = torch.Tensor([1, 2, 3, 4])
            bad_spectra = torch.Tensor([[1, 2, 3], [1, 2, 3]])
            SpectralData(wavelengths, bad_spectra)

    def test_component_shape_case_1(self):
        with pytest.raises(ValueError):
            wavelengths = torch.Tensor([1, 2, 3, 4])
            spectra = torch.randn(4, 2)
            components = ObservedComponents(torch.randn(2, 3))
            
            SpectralData(wavelengths, spectra)


    def test_mean_center_by_spectra(self):
        torch.set_default_dtype(torch.float64)
        wavelengths = torch.linspace(-10,10,500).reshape(-1,1)
        shift = torch.randn(1000)
        observations = torch.sin(shift[:, None] + wavelengths.T)
        data = SpectralData(wavelengths, observations)
        data.snv()
        assert torch.all(data.spectra.mean(axis = 1).abs()) < 1e-10



    def test_snv(self):
        torch.set_default_dtype(torch.float64)
        wavelengths = torch.linspace(-10,10,500).reshape(-1,1)
        shift = torch.randn(1000)
        observations = torch.sin(shift[:, None] + wavelengths.T)
        data = SpectralData(wavelengths, observations)
        data.snv()
        assert torch.all(data.spectra.mean(axis = 1).abs()) < 1e-10
        assert torch.all(data.spectra.var(axis=1) - 1 < 1e-10)

    def test_cut_by_wavelengths(self):
        torch.set_default_dtype(torch.float64)
        wavelengths = torch.linspace(-10,10,500).reshape(-1,1)
        shift = torch.randn(1000)
        observations = torch.sin(shift[:, None] + wavelengths.T)
        data = SpectralData(wavelengths, observations)
        wavelength_mask = torch.logical_and(-5 < wavelengths, wavelengths < 5)
        observation_mask = wavelength_mask.reshape(1,-1).repeat(1000,1)
        data.trim_wavelengths(-5,5)
        assert torch.all(data.wavelengths == wavelengths[wavelength_mask].reshape(-1,1))
        assert torch.all(data.spectra == observations[observation_mask].reshape(1000, -1))

    def test_select_wavelengths_in_output_range(self):
        torch.set_default_dtype(torch.float64)
        wavelengths = torch.linspace(-10,10,500).reshape(-1,1)
        shiftUp = torch.randn(1000)
        observations = torch.sin(wavelengths.flatten()) + shiftUp[:, None]
        data = SpectralData(wavelengths, observations)
        data.filter_by_intensity(min_intensity = -2, max_intensity = 2)
        assert torch.all(data.spectra<2)
        assert torch.all(data.spectra>-2)


class TestComponents:
    def test_validate_inputs(self):
        with pytest.raises(ValueError):
            r = torch.randn(100,100,100)
            components = ObservedComponents(r)

    def test_observed_outer(self):
        torch.set_default_dtype(torch.float64)
        r = torch.randn(100,10)
        expected = torch.zeros(100,10,10)
        for i in range(100): 
            for j in range(10):
                for k in range(10):
                    expected[i,j,k] = r[i,j]*r[i,k]
        components = ObservedComponents(r)
        router = components.get_r_outer()
        assert torch.all(expected == router)

    def test_observed_prior_term(self):
        r = torch.rand(100, 10)
        r = r/r.sum(axis=1)[:, None]
        dist = torch.distributions.Dirichlet(torch.ones(10))
        p_single_data_point = 12.8018     # Taken as single measurement
        components = ObservedComponents(r)
        assert -p_single_data_point*100 + components.get_prior_term() <1e-5


class TestImage:
    def test_init(self):
        images = Images(torch.randn(100,2,2))
        assert torch.all(images.inputs == torch.Tensor([[0,0], [0,1], [1,0], [1,1]]))

    def test_get_output(self):
        images = Images(torch.Tensor([[[1,2],[3,4]],[[5,6],[7,8]]]))
        assert torch.all(images.get_outputs() == torch.Tensor([[1,2,3,4],[5,6,7,8]]))
        

