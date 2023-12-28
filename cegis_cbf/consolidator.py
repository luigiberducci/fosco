import torch


class Consolidator:
    def get(self, cex, datasets, **kwargs):
        datasets = self.add_ces_to_data(cex, datasets)
        return {"datasets": datasets}

    def add_ces_to_data(self, cex, datasets):
        for lab, cex in cex.items():
            if cex != []:
                x = cex
                datasets[lab] = torch.cat([datasets[lab], x], dim=0).detach()
            print(lab, datasets[lab].shape)
        return datasets


def make_consolidator(**kwargs) -> Consolidator:
    """
    Factory method for consolidator.

    :param kwargs:
    :return:
    """
    return Consolidator(**kwargs)
