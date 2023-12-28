from typing import Iterable

import numpy as np
import torch


def contains_object(x, obj):
    if isinstance(x, Iterable):
        return contains_object(next(iter(x)), obj)
    else:
        return isinstance(x, obj)


# n-dim generalisation of circle and sphere
def round_init_data(centre, r, batch_size, on_border=False):
    """
    :param centre:
    :param r:
    :param batch_size:
    :return:
    """
    dim = len(centre)
    if dim == 1:
        return segment([centre[0] - r, centre[0] + r], batch_size)
    elif dim == 2:
        return circle_init_data(centre, r, batch_size, on_border=on_border)
    elif dim == 3:
        return sphere_init_data(centre, r, batch_size, on_border=on_border)
    else:
        return n_dim_sphere_init_data(centre, r, batch_size, on_border=on_border)


def segment(dims, batch_size):
    return (dims[1] - dims[0]) * torch.rand(batch_size, 1) + dims[0]


# generates data for (X - centre)**2 <= radius
def sphere_init_data(centre, r, batch_size, on_border=False):
    """
    :param centre: list/tupe/tensor containing the 3 coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    # spherical coordinates
    # x = r sin(theta) cos(phi)
    # y = r sin(theta) sin(phi)
    # z = r cos(theta)
    theta = (2 * np.pi) * torch.rand(batch_size, 1)
    phi = np.pi * torch.rand(batch_size, 1)
    r = np.sqrt(r)
    if on_border:
        radius = r * torch.ones(batch_size, 1)
    else:
        radius = r * torch.rand(batch_size, 1)
    x_coord = radius * np.sin(theta) * np.cos(phi)
    y_coord = radius * np.sin(theta) * np.sin(phi)
    z_coord = radius * np.cos(theta)
    offset = torch.cat([x_coord, y_coord, z_coord], dim=1)

    return torch.tensor(centre) + offset


# generates points in a n-dim sphere: X**2 <= radius**2
# adapted from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
# method 20: Muller generalised
def n_dim_sphere_init_data(centre, radius, batch_size, on_border=False):
    dim = len(centre)
    u = torch.randn(
        batch_size, dim
    )  # an array of d normally distributed random variables
    norm = torch.sum(u**2, dim=1) ** (0.5)
    if on_border:
        r = radius * torch.ones(batch_size, dim) ** (1.0 / dim)
    else:
        r = radius * torch.rand(batch_size, dim) ** (1.0 / dim)
    x = torch.div(r * u, norm[:, None]) + torch.tensor(centre)

    return x


# generates data for (X - centre)**2 <= radius
def circle_init_data(centre, r, batch_size, on_border=False):
    """
    :param centre: list/tuple/tensor containing the 'n' coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    border_batch = int(batch_size / 10)
    internal_batch = batch_size - border_batch
    r = np.sqrt(r)
    angle = (2 * np.pi) * torch.rand(internal_batch, 1)
    if on_border:
        radius = r * torch.ones(internal_batch, 1)
    else:
        radius = r * torch.rand(internal_batch, 1)
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    offset = torch.cat([x_coord, y_coord], dim=1)

    angle = (2 * np.pi) * torch.rand(border_batch, 1)
    x_coord = r * np.cos(angle)
    y_coord = r * np.sin(angle)
    offset_border = torch.cat([x_coord, y_coord], dim=1)
    offset = torch.cat([offset, offset_border])

    return torch.tensor(centre) + offset


def square_init_data(domain, batch_size, on_border=False):
    """
    :param domain: list = [lower_bounds, upper_bounds]
                    lower_bounds, upper_bounds are lists
    :param batch_size: int
    :return:
    """

    r1 = torch.tensor(domain[0])
    r2 = torch.tensor(domain[1])
    square_uniform = (r1 - r2) * torch.rand(batch_size, len(domain[0])) + r2
    return square_uniform


def _set_assertion(required: object, actual: object, name: object) -> object:
    assert required == actual, (
        f"Required {name} {required} do not match actual domains {actual}. "
        f"Missing: {required - actual}, Not required: {actual - required}"
    )
