def calculate_current_noise_annealing(
    current_step, noise_annealing_init, noise_annealing_gamma
):
    assert noise_annealing_gamma < 1
    return noise_annealing_init * (noise_annealing_gamma**current_step)
