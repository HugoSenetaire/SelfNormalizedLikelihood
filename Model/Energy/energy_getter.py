from .linear import fc_energy

dic_energy = {
    'fc': fc_energy
}

def get_energy(input_size, args_dict):
    if 'energy_name' not in args_dict:
        raise ValueError("Energy name not specified")
    if args_dict['energy_name'] not in dic_energy:
        raise ValueError("Energy name not valid")
    energy = dic_energy[args_dict['energy_name']]
    if 'energy_params' not in args_dict:
        energy = energy(input_size=input_size)
    else:
        energy = energy(input_size=input_size, **args_dict['energy_params'])
    return energy