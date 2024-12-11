from subprocess import call
from tempfile import TemporaryDirectory
import numpy as np
import os

# run tckgen with ACT options
def perform_act(wm_fod_fname, tt_img_fname, gmwmi_fname, output_tck_fname, n_streamlines, downsampling_ratio=10):
    call("tckgen -seeds %s -minlength 5 -maxlength 250 -cutoff 0.06 -downsample %s -seed_gmwmi %s "
         "-act %s -backtrack -crop_at_gmwmi %s %s -force -nthreads 20" % (n_streamlines, downsampling_ratio,
                                                                          gmwmi_fname, tt_img_fname, wm_fod_fname,
                                                                          output_tck_fname),
         shell=True)

# weight tractogram using SIFT2
def perform_sift2(tractogram_fname, tt_fname, wm_fod_fname, weights_output_fname, mu_output_fname):
    call("tcksift2 -act %s -out_mu %s %s %s %s -force" % (tt_fname, mu_output_fname, tractogram_fname, wm_fod_fname, weights_output_fname), shell=True)

# sum SIFT2 weights between ROIs
def construct_connectome(tractogram_fname, atlas_fname, sift_weights_fname, sift_mu_fname, output_matrix, output_assignments):
    with TemporaryDirectory() as tmpdir:
        output_csv = os.path.join(tmpdir, "matrix.csv")
        call("tck2connectome -assignment_radial_search 4 -symmetric -zero_diagonal -tck_weights_in %s -out_assignments %s %s %s %s -force" %
             (sift_weights_fname, output_assignments, tractogram_fname, atlas_fname, output_csv), shell=True)

        struct_mat = np.loadtxt(output_csv, delimiter=',')
        struct_mat = np.loadtxt(sift_mu_fname) * struct_mat
        np.save(output_matrix, struct_mat)
