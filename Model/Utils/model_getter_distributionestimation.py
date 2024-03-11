import numpy as np

from ..ZEstimator import dic_z_estimator
from ..Energy import get_energy, get_explicit_bias, FullEBM
from ..Proposals import get_base_dist, get_proposal
from .init_proposals import (
    init_energy_to_gaussian,
    init_energy_to_proposal,
    init_proposal_to_data,
)


def get_model(cfg, complete_dataset, complete_masked_dataset, loader_train):
    input_size = complete_dataset.get_dim_input()

    # Get energy function :
    print("Get energy function")
    fnn = get_energy(input_size, cfg)
    print("Get energy function... end")



    if cfg.base_distribution.proposal_name == "proposal":
        # Get proposal :
        if cfg.proposal.proposal_name is not None:
            print("Get proposal")
            proposal = get_proposal(
                cfg=cfg,
                input_size=input_size,
                dataset=[
                    complete_dataset.dataset_train,
                    complete_dataset.dataset_val,
                    complete_dataset.dataset_test,
                ],
                f_theta=energy,
                base_dist=None,
            )
            print("Get proposal... end")
        else:
            raise ValueError("No proposal given")
    else:
        base_dist = get_base_dist(
            cfg=cfg,
            proposal=None,
            input_size=input_size,
            dataset=[
                complete_dataset.dataset_train,
                complete_dataset.dataset_val,
                complete_dataset.dataset_test,
            ],
        )
        if cfg.proposal.proposal_name is not None:
            proposal = get_proposal(
                cfg=cfg,
                input_size=input_size,
                dataset=[
                    complete_dataset.dataset_train,
                    complete_dataset.dataset_val,
                    complete_dataset.dataset_test,
                ],
                f_theta=energy,
                base_dist=base_dist,
            )
        else :
            raise ValueError("No proposal given")

    # Get base_dist :
    base_dist = get_base_dist(
        cfg=cfg,
        proposal=proposal,
        input_size=input_size,
        dataset=complete_dataset.dataset_train,
    )
    if cfg.base_distribution.proposal_name == "proposal":
        assert proposal == base_dist, "Proposal and base_dist should be the same"

    # if base_dist is None and ('ebm_pretraining' not in args_dict.keys() or args_dict['ebm_pretraining'] is None):
    if base_dist is None and cfg.energy.ebm_pretraining is None:
        print("Careful, no base_dist given, the energy might not be well initialized")

    if cfg.proposal_training.proposal_pretraining == "data":
        print("Init proposal")
        proposal = init_proposal_to_data(
            proposal=proposal, input_size=input_size, dataloader=loader_train, cfg=cfg
        )
        print("Init proposal... end")

    if cfg.energy.ebm_pretraining == "gaussian":
        print("Init energy to standard gaussian")
        energy = init_energy_to_gaussian(
            energy, input_size, complete_dataset.dataset_train, cfg
        )
        print("Init energy to standard gaussian... end")
    elif cfg.energy.ebm_pretraining == "proposal":
        print("Init energy to proposal")
        energy = init_energy_to_proposal(
            energy, proposal, input_size, complete_dataset.dataset_train, cfg
        )
        print("Init energy to proposal... end")

    explicit_bias = get_explicit_bias(cfg)

    ebm = FullEBM(fnn, base_dist, explicit_bias)

    # Get z_estimator :
    print("Get EBM")
    
    
    z_estimator = dic_z_estimator[cfg.ebm.z_estimator_name](
        energy=energy,
        proposal=proposal,
        cfg_ebm=cfg.ebm,
        nb_sample_init_bias=cfg.explicit_bias.nb_sample_init_bias,
    )
    print("Get EBM... end")

    return ebm, z_estimator
