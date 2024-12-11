
# generate an ANTs registration command for non-linear registration with the Mutual Information metric
def generate_registration_command_SyN_MI(ants_reg_command, output_prefix, fixed, moving):
    output_1 = output_prefix
    output_2 = output_prefix + "Warped.nii.gz"
    output_3 = output_prefix + "InverseWarped.nii.gz"
    return  "%s -n 20 --verbose 0 --dimensionality 3 --float 0 --collapse-output-transforms 1 --output [ %s,%s,%s ] --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ] --initial-moving-transform [ %s,%s,1 ] --transform Rigid[ 0.1 ] --metric MI[ %s,%s,1,32,Regular,0.25 ] --convergence [ 1000x500x250x100,1e-6,10 ] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform Affine[ 0.1 ] --metric MI[ %s,%s,1,32,Regular,0.25 ] --convergence [ 1000x500x250x100,1e-6,10 ] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform SyN[ 0.1,3,0 ] --metric MI[ %s,%s,1,32,Regular,1.0 ] --convergence [ 100x70x50x20,1e-8,10 ] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox" % (ants_reg_command, output_1, output_2, output_3, fixed, moving, fixed, moving, fixed, moving, fixed, moving)

# generate an ANTs registration command for affine registration
def generate_registration_command_affine(ants_reg_command, output_prefix, fixed, moving):
    output_1 = output_prefix
    output_2 = output_prefix + "_reg.nii.gz"
    output_3 = output_prefix + "_inversereg.nii.gz"
    return "%s -n 20 --verbose 0 --dimensionality 3 --float 0 --collapse-output-transforms 1 --output [ %s,%s,%s ] --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ] --initial-moving-transform [ %s,%s,1 ] --transform Rigid[ 0.1 ] --metric MI[ %s,%s,1,32,Regular,0.25 ] --convergence [ 1000x500x250x100,1e-6,10 ] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform Affine[ 0.1 ] --metric MI[ %s,%s,1,32,Regular,0.25 ] --convergence [ 1000x500x250x100,1e-6,10 ] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox" % (ants_reg_command, output_1, output_2, output_3, fixed, moving, fixed, moving, fixed, moving)

