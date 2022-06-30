import torch
import pytest
from adaptive_sampling.colvars.kearsley import Kearsley
from adaptive_sampling.colvars.utils import *


@pytest.mark.parametrize(
    "a, b, expected_nofit, expected_fit",
    [
        (
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [9, 9, 7],
                    [9, 9, 8],
                    [9, 9, 9],
                ]
            ),
            torch.tensor(
                [
                    [30.50347534, -20.16089091, -7.42752623],
                    [30.77704903, -21.02339348, -7.27823201],
                    [31.3215374, -21.99452332, -7.15703548],
                    [42.05988643, -23.50924264, -15.59516355],
                    [42.27217891, -24.36478643, -15.59064995],
                    [42.66080502, -25.27318759, -15.386241],
                ]
            ),
            26.6020,
            0.0979,
        )
    ],
)
def test_kearsley(a, b, expected_nofit, expected_fit):
    k = Kearsley()
    a = torch.flatten(a).float()
    b = torch.flatten(b).float()

    rmsd_nofit = rmsd(a, b)
    rmsd_kearsley = k.fit(a, b)

    assert float(rmsd_nofit) == pytest.approx(expected_nofit, rel=1.0e-3)
    assert float(rmsd_kearsley) == pytest.approx(expected_fit, rel=1.0e-3)


@pytest.mark.parametrize(
    "a, b, expected_nofit, expected_fit",
    [
        (
            torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [9, 9, 7],
                    [9, 9, 8],
                    [9, 9, 9],
                ]
            ),
            torch.tensor(
                [
                    [30.50347534, -20.16089091, -7.42752623],
                    [30.77704903, -21.02339348, -7.27823201],
                    [31.3215374, -21.99452332, -7.15703548],
                    [42.05988643, -23.50924264, -15.59516355],
                    [42.27217891, -24.36478643, -15.59064995],
                    [42.66080502, -25.27318759, -15.386241],
                ]
            ),
            26.6020,
            0.0979,
        )
    ],
)
def test_kabsch(a, b, expected_nofit, expected_fit):
    a = torch.flatten(a).float()
    b = torch.flatten(b).float()

    rmsd_nofit = rmsd(a, b)
    rmsd_kabsch = kabsch_rmsd(a, b)

    assert float(rmsd_nofit) == pytest.approx(expected_nofit, rel=1.0e-3)
    assert float(rmsd_kabsch) == pytest.approx(expected_fit, rel=1.0e-3)


@pytest.mark.parametrize(
    "images, interpolated_images",
    [
        (
            [
                torch.tensor([0.0, 0, 0, 0, 0, 0]),
                torch.tensor([1.0, 0, 0, 0, 0, 0]),
            ],
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ),
        )
    ],
)
def test_interpolate(images, interpolated_images):
    i = interpolate_coordinates(images, n_interpol=5)
    i = torch.stack(i)
    print(i)
    assert torch.allclose(i, interpolated_images, rtol=1.0e-5)
