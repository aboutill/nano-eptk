import os
import ants


def create_jacobian_determinant(
        input_fixed_path,
        input_moving_path,
        output_jac_path,
        sigma=1.0,
    ):
    
    # TODO Hnadle week to week template
    output_dir = os.path.dirname(output_jac_path)
    os.makedirs(output_dir, exist_ok=True)
    
    fixed = ants.image_read(input_fixed_path)
    moving = ants.image_read(input_moving_path)
    warp = ants.registration(fixed=fixed , moving=moving, type_of_transform='SyNRA')
    jac = ants.create_jacobian_determinant_image(fixed, warp['fwdtransforms'][0], do_log=1)
    jac = ants.smooth_image(jac, sigma=sigma, sigma_in_physical_coordinates=False)
    jac.image_write(output_jac_path)