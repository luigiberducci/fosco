from .system import ControlAffineControllableDynamicalModel


def make_system(id: str) -> ControlAffineControllableDynamicalModel:
    if id == "single_integrator":
        from systems.single_integrator import SingleIntegrator
        return SingleIntegrator
    elif id == "double_integrator":
        from systems.double_integrator import DoubleIntegrator
        return DoubleIntegrator
    else:
        raise NotImplementedError(f"System {id} not implemented")
