def calculate_current_noise_annealing(
    global_step, noise_annealing_init, noise_annealing_gamma
):
    assert noise_annealing_gamma < 1
    return noise_annealing_init * (noise_annealing_gamma**global_step)
