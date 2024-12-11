import os
from subprocess import call, DEVNULL
import numpy as np
import tempfile
from shutil import copyfile
import nibabel as nib


from registration_funcs import generate_registration_command_SyN_MI, generate_registration_command_affine
from tt_construction import extract_dgm_dwireg, generate_gmwmi, hack_5tt
from tractography import perform_act, construct_connectome, perform_sift2
from slant_funcs import run_slant, slant_to_atlas


ants_path = "/path/to/ants/bin/"
ants_registration_path = "antsRegistration"
ants_apply_registration_path = "antsApplyTransforms"

# customize which steps to perform
DO_DWI_PREPROC = True
DO_BIASCORRECT = True
DO_STRIP = True
DO_REGRID_B0 = True
DO_DWI_REG = True
DO_PREPOST_REG = True
DO_SLANT = True
DO_SLANT_REG = True
DO_RESPONSES = True
DO_RESPONSE_AVERAGING = True
DO_CSD = True
DO_5tt = True
DO_TRACTOGRAPHY = True
DO_SIFT = True
DO_CONNECTOME = True

# re-do and overwrite if results of intermediate steps already exist?
OVERWRITE = False

# if true, prints log to terminal
VERBOSE = False
stdout = None if VERBOSE else DEVNULL

# path to subjects folder
datapath = "/path/to/data"

# input subject folders are structured as:
#       - subj_name
#           - pre
#               - raw
#                   - T1.nii
#                   - DWI.nii
#                   - DWI.bval
#                   - DWI.bvec
#           - post
#               - raw
#                   - T1.nii
#                   - DWI.nii
#                   - DWI.bval
#                   - DWI.bvec


# list of subjects, default is all folders in data path
subset = os.listdir(datapath)

for pat in subset:
    pat_dir = os.path.join(datapath, pat)

    print("processing patient " + pat)

    # process both pre-op and post-op sessions
    for sess in ['pre', 'post']:
        print("session " + sess)
        sess_dir = os.path.join(pat_dir, sess)

        # input images are in /subject/session/raw/
        T1 = os.path.join(sess_dir, "raw", "T1.nii")

        T1_stripped = os.path.join(preproc_dir, "T1_stripped.nii.gz")
        T1_stripped_mask = os.path.join(preproc_dir, "T1_stripped_mask.nii.gz")
        if DO_STRIP and (not os.path.exists(T1_stripped) or OVERWRITE):
            # skull stripping with hd-bet
            print("skull stripping")
            call("hd-bet -i %s -o %s" % (T1, T1_stripped), shell=True, stdout=stdout, stderr=stdout)

            # retry with cpu
            if not os.path.exists(T1_stripped):
                call("hd-bet -i %s -o %s -device cpu" % (T1, T1_stripped), shell=True)

        DWI = os.path.join(sess_dir, "raw", "DWI.nii")
        bvals = os.path.join(sess_dir, "raw", "DWI.bval")
        bvecs = os.path.join(sess_dir, "raw", "DWI.bvec")

        preproc_dir = os.path.join(sess_dir, "preproc")
        if not os.path.exists(preproc_dir):
            os.mkdir(preproc_dir)

        # affinely register stripped T1 mask onto DWI as initial mask for preprocessing
        DWI_initial_mask = os.path.join(preproc_dir, "DWI_mask_initial.nii")
        if not os.path.exists(DWI_initial_mask) or OVERWRITE:
            with tempfile.TemporaryDirectory as tmpdir:
                tmp_b0 = os.path.join(tmpdir, "b0.nii.gz")
                call("dwiextract %s - -bzero | mrmath - mean %s -axis 3 -force" % (DWI, tmp_b0),
                     shell=True, stdout=stdout)


                reg_command = generate_registration_command_affine(ants_registration_path,
                                                               tmpdir, tmp_b0, T1_stripped)
                call(reg_command, shell=True, stdout=stdout)

                tmp_affine = os.path.join(tmpdir, "0GenericAffine.mat")

                call("%s -d 3 -i %s -r %s -o %s -n NearestNeighbor -t %s" % (ants_apply_registration_path,
                                                                            T1_stripped_mask,
                                                                             tmp_b0,
                                                                             DWI_initial_mask,
                                                                             tmp_affine),
                     shell=True, stdout=stdout)


        csd_dir = os.path.join(sess_dir, "csd")
        if not os.path.exists(csd_dir):
            os.mkdir(csd_dir)

        DWI_preproc = os.path.join(preproc_dir, "DWI_fslpreproc.mif")

        if DO_DWI_PREPROC and (not os.path.exists(DWI_preproc) or OVERWRITE):
            # preprocess DWI images
            print("denoising DWI")
            DWI_denoised = os.path.join(preproc_dir, "DWI_denoised.mif")
            call("dwidenoise %s %s -force -mask %s" % (DWI, DWI_denoised, DWI_initial_mask),
                 shell=True, stdout=stdout, stderr=stdout)

            print("running dwifslpreproc")
            eddy_out_dir = os.path.join(preproc_dir, "eddy")
            call("dwifslpreproc %s -fslgrad %s %s %s -rpe_none -pe_dir j -nthreads 20 -force -eddyqc_text %s -eddy_mask %s" % (
                                                                                                    DWI_denoised,
                                                                                                    bvecs, bvals,
                                                                                                    DWI_preproc,
                                                                                                    eddy_out_dir,
                                                                                                    DWI_initial_mask),
                 shell=True, stdout=stdout, stderr=stdout)

        # bias correct DWI
        DWI_biascorrected = os.path.join(preproc_dir, "DWI_biascorrected.mif")
        if DO_BIASCORRECT and (not os.path.exists(DWI_biascorrected) or OVERWRITE):
            print("running bias correct")
            call("dwibiascorrect ants -mask %s %s %s -force" % (
            DWI_initial_mask, DWI_preproc, DWI_biascorrected),
                 shell=True, stdout=stdout, stderr=stdout)

        # regrid DWI and b0 to 1mm isotropic
        b0_upsampled_fname = os.path.join(sess_dir, "b0.nii.gz")
        if DO_REGRID_B0 and (not os.path.exists(b0_upsampled_fname) or OVERWRITE):
            print("regridding b0")
            DWI_biascorrected_mask_automatic = os.path.join(preproc_dir, "DWI_mask_biascorrected_automatic.nii.gz")
            call("dwi2mask %s %s -force" % (DWI_biascorrected, DWI_biascorrected_mask_automatic), shell=True, stdout=stdout)

            DWI_preproc_upsampled_fname = os.path.join(sess_dir, "DWI_preproc.mif")
            call("mrgrid %s regrid %s -voxel 1 -force" % (DWI_biascorrected, DWI_preproc_upsampled_fname),
                 shell=True, stdout=stdout)

            call("dwiextract %s - -bzero | mrmath - mean %s -axis 3 -force" % (DWI_preproc_upsampled_fname,
                                                                               b0_upsampled_fname),
                 shell=True, stdout=stdout)
            DWI_mask_upsampled_fname = os.path.join(sess_dir, "DWI_mask.nii.gz")
            call("mrgrid %s regrid %s -voxel 1 -force" % (DWI_biascorrected_mask_automatic, DWI_mask_upsampled_fname),
                 shell=True, stdout=stdout)

        reg_dir = os.path.join(sess_dir, "reg")
        if not os.path.exists(reg_dir):
            os.mkdir(reg_dir)
        T1_dwireg = os.path.join(sess_dir, "T1_dwireg.nii.gz")
        dwireg_affine = os.path.join(reg_dir, "T1_dwi_affine.mat")
        dwireg_warp = os.path.join(reg_dir, "T1_dwi_warp.nii.gz")
        # register T1 to DWI for later atlas transformation
        if DO_DWI_REG and (not os.path.exists(T1_dwireg) or OVERWRITE):
            print("registering T1 to DWI")
            T1_stripped_mask_fname = os.path.join(preproc_dir, "T1_stripped_mask.nii.gz")
            T1_inverted_fname = os.path.join(reg_dir, "T1_inverted.nii.gz")

            # we invert the skull-stripped T1 and non-linearly register the T1 onto the b0 image to account for distortion effects
            strip_mask = nib.load(T1_stripped_mask_fname).get_fdata() == 1
            T1_nib = nib.load(T1_stripped)
            T1_data = T1_nib.get_fdata()
            T1_avg = np.average(T1_data[strip_mask])
            T1_demeaned = T1_data - T1_avg
            T1_inverted = T1_avg - T1_demeaned
            T1_inverted[np.logical_not(strip_mask)] = 0

            nib.save(nib.Nifti1Image(T1_inverted, T1_nib.affine), T1_inverted_fname)

            with tempfile.TemporaryDirectory() as tmpdirname:
                T1_inverted_dwireg_tmp = os.path.join(tmpdirname, "ants_")

                call("antsRegistrationSyN.sh -d 3 -f %s -m %s -o %s -n 20" % (b0_upsampled_fname, T1_inverted_fname,
                                                                              T1_inverted_dwireg_tmp),
                     shell=True, stdout=stdout)

                ants_affine = os.path.join(tmpdirname, "ants_0GenericAffine.mat")
                ants_warp = os.path.join(tmpdirname, "ants_1Warp.nii.gz")
                call("%s -d 3 -i %s -r %s -o %s -t %s %s" % (ants_registration_path, T1, b0_upsampled_fname, T1_dwireg,
                                                                                                    ants_warp,
                                                                                                    ants_affine),
                     shell=True, stdout=stdout)

                copyfile(ants_affine, dwireg_affine)
                copyfile(ants_warp, dwireg_warp)

        parc_dir = os.path.join(sess_dir, "parc")
        if not os.path.exists(parc_dir):
            os.mkdir(parc_dir)

        slant_atlas = os.path.join(parc_dir, "slant_atlas.nii.gz")
        if DO_SLANT and (not os.path.exists(slant_atlas) or OVERWRITE):
            # parcellations with SLANT
            print("running SLANT")
            slant_seg = os.path.join(parc_dir, "slant_seg.nii.gz")
            run_slant(T1, slant_seg)

            # convert whole-brain segmentation generated by SLANT into cortical and subcortical atlas
            slant_to_atlas(slant_seg, slant_atlas)

        wm_response_fname = os.path.join(csd_dir, "wm_response.txt")
        if DO_RESPONSES and (not os.path.exists(wm_response_fname) or OVERWRITE):
            # get diffusion responses
            print("calculating diffusion responses")
            dwi_preproc_fname = os.path.join(sess_dir, "DWI_preproc.mif")
            mask_fname = os.path.join(sess_dir, "DWI_mask.nii.gz")
            gm_response_fname = os.path.join(csd_dir, "gm_response.txt")
            csf_response_fname = os.path.join(csd_dir, "csf_response.txt")

            call("dwi2response dhollander %s %s %s %s -force -mask %s" % (dwi_preproc_fname,
                                                                          wm_response_fname,
                                                                          gm_response_fname,
                                                                          csf_response_fname,
                                                                          mask_fname),
                 shell=True)#, stderr=stdout, stdout=stdout)

    pre_postreg_T1 = os.path.join(pat_dir, "post", "reg", "T1_pre_postreg.nii.gz")
    if DO_PREPOST_REG and (not os.path.exists(pre_postreg_T1 or OVERWRITE)):
        # register preop T1 to postop T1
        print("registering preop T1 to postop")
        pre_T1 = os.path.join(pat_dir, "pre", "raw", "T1.nii")
        post_T1 = os.path.join(pat_dir, "post", "raw", "T1.nii")

        pre_postreg_warp = os.path.join(pat_dir, "post", "reg", "T1_pre_postreg_warp.nii.gz")
        pre_postreg_affine = os.path.join(pat_dir, "post", "reg", "T1_pre_postreg_affine.mat")

        with tempfile.TemporaryDirectory() as tmpdirname:
            prepostreg_tmp = os.path.join(tmpdirname, "ants_")

            reg_command = generate_registration_command_SyN_MI(ants_path,
                                                               prepostreg_tmp, post_T1, pre_T1)
            call(reg_command, shell=True, stdout=stdout)

            temp_affine = prepostreg_tmp + "0GenericAffine.mat"
            temp_warp = prepostreg_tmp + "1Warp.nii.gz"
            temp_postreg = prepostreg_tmp + "Warped.nii.gz"

            call("cp %s %s" % (temp_affine, pre_postreg_affine), shell=True)
            call("cp %s %s" % (temp_warp, pre_postreg_warp), shell=True)
            call("cp %s %s" % (temp_postreg, pre_postreg_T1), shell=True)

    slant_seg_post_dwireg = os.path.join(pat_dir, "post", "parc", "slant_seg_dwireg.nii.gz")
    if DO_SLANT_REG and (not os.path.exists(slant_seg_post_dwireg or OVERWRITE)):
        print("registering slant to postop")

        # register pre-op SLANT atlas to pre-op DWI space
        slant_atlas_pre = os.path.join(pat_dir, "pre", "parc", "slant_atlas.nii.gz")
        pre_b0 = os.path.join(pat_dir, "pre", "b0.nii.gz")
        slant_atlas_pre_dwireg = os.path.join(pat_dir, "pre", "parc", "slant_atlas_dwireg.nii.gz")
        pre_dwireg_warp = os.path.join(pat_dir, "pre", "reg", "T1_dwi_warp.nii.gz")
        pre_dwireg_affine = os.path.join(pat_dir, "pre", "reg", "T1_dwi_affine.mat")
        call("%s -d 3 -i %s -r %s -o %s -n NearestNeighbor -t %s %s" % (ants_apply_registration_path,
                                                                        slant_atlas_pre,
                                                                         pre_b0,
                                                                         slant_atlas_pre_dwireg,
                                                                         pre_dwireg_warp,
                                                                         pre_dwireg_affine),
             shell=True, stdout=stdout)

        slant_seg_pre = os.path.join(pat_dir, "pre", "parc", "slant_seg.nii.gz")
        slant_seg_pre_dwireg = os.path.join(pat_dir, "pre", "parc", "slant_seg_dwireg.nii.gz")
        call("%s -d 3 -i %s -r %s -o %s -n NearestNeighbor -t %s %s" % (ants_apply_registration_path,
                                                                        slant_seg_pre,
                                                                         pre_b0,
                                                                         slant_seg_pre_dwireg,
                                                                         pre_dwireg_warp,
                                                                         pre_dwireg_affine),
             shell=True, stdout=stdout)

        # register pre-op SLANT atlas to post-op DWI
        slant_atlas_post = os.path.join(pat_dir, "post", "parc", "slant_atlas.nii.gz")
        T1_post = os.path.join(pat_dir, "post", "raw", "T1.nii")
        pre_postreg_warp = os.path.join(pat_dir, "post", "reg", "T1_pre_postreg_warp.nii.gz")
        pre_postreg_affine = os.path.join(pat_dir, "post", "reg", "T1_pre_postreg_affine.mat")
        call("%s -d 3 -i %s -r %s -o %s -n NearestNeighbor -t %s %s" % (ants_apply_registration_path,
                                                                        slant_atlas_pre,
                                                                         T1_post,
                                                                         slant_atlas_post,
                                                                         pre_postreg_warp,
                                                                         pre_postreg_affine),
             shell=True, stdout=stdout)

        slant_seg_post = os.path.join(pat_dir, "post", "parc", "slant_seg.nii.gz")
        call("%s -d 3 -i %s -r %s -o %s -n NearestNeighbor -t %s %s" % (ants_apply_registration_path,
                                                                        slant_seg_pre,
                                                                         T1_post,
                                                                         slant_seg_post,
                                                                         pre_postreg_warp,
                                                                         pre_postreg_affine),
             shell=True, stdout=stdout)

        post_b0 = os.path.join(pat_dir, "post", "b0.nii.gz")
        slant_atlas_post_dwireg = os.path.join(pat_dir, "post", "parc", "slant_atlas_dwireg.nii.gz")
        post_dwireg_warp = os.path.join(pat_dir, "post", "reg", "T1_dwi_warp.nii.gz")
        post_dwireg_affine = os.path.join(pat_dir, "post", "reg", "T1_dwi_affine.mat")
        call("%s -d 3 -i %s -r %s -o %s -n NearestNeighbor -t %s %s" % (ants_apply_registration_path,
                                                                        slant_atlas_post,
                                                                         post_b0,
                                                                         slant_atlas_post_dwireg,
                                                                         post_dwireg_warp,
                                                                         post_dwireg_affine),
             shell=True, stdout=stdout)

        call("%s -d 3 -i %s -r %s -o %s -n NearestNeighbor -t %s %s" % (ants_apply_registration_path,
                                                                        slant_seg_post,
                                                                         post_b0,
                                                                         slant_seg_post_dwireg,
                                                                         post_dwireg_warp,
                                                                         post_dwireg_affine),
             shell=True, stdout=stdout)


avg_wm_response_fname = os.path.join(datapath, "pat_avg_wm_response.txt")
avg_gm_response_fname = os.path.join(datapath, "pat_avg_gm_response.txt")
avg_csf_response_fname = os.path.join(datapath, "pat_avg_csf_response.txt")
if DO_RESPONSE_AVERAGING:
    # to obtain meaningful results in the patient group we need to average diffusion responses of all subjects
    avg_wm_response = np.zeros((2, 6))
    avg_gm_response = np.zeros((2))
    avg_csf_response = np.zeros((2))
    for pat in subset:
        print(pat)
        pat_dir = os.path.join(datapath, pat)

        for sess in ['pre', 'post']:
            print(sess)
            sess_dir = os.path.join(pat_dir, sess)
            csd_dir = os.path.join(sess_dir, "csd")
            wm_response_fname = os.path.join(csd_dir, "wm_response.txt")
            gm_response_fname = os.path.join(csd_dir, "gm_response.txt")
            csf_response_fname = os.path.join(csd_dir, "csf_response.txt")

            avg_wm_response += np.loadtxt(wm_response_fname)
            avg_gm_response += np.loadtxt(gm_response_fname)
            avg_csf_response += np.loadtxt(csf_response_fname)

    np.savetxt(avg_wm_response_fname, avg_wm_response / (2 * len(subset)))
    np.savetxt(avg_gm_response_fname, avg_gm_response / (2 * len(subset)))
    np.savetxt(avg_csf_response_fname, avg_csf_response / (2 * len(subset)))


for pat in subset:
    print(pat)
    pat_dir = os.path.join(datapath, pat)
    for sess in ['pre', 'post']:
        print(sess)
        sess_dir = os.path.join(pat_dir, sess)
        csd_dir = os.path.join(sess_dir, "csd")
        output_wm_fod_fname = os.path.join(csd_dir, "wm_fod_avg_response.mif")
        output_gm_fod_fname = os.path.join(csd_dir, "gm_fod_avg_response.mif")
        output_csf_fod_fname = os.path.join(sess_dir, "csf_fod_avg_response.mif")
        dwi_preproc_fname = os.path.join(sess_dir, "DWI_preproc.mif")
        mask_fname = os.path.join(sess_dir, "DWI_mask.nii.gz")

        if DO_CSD and (not os.path.exists(output_wm_fod_fname) or OVERWRITE):
            # get FODs with SS3T-CSD
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_wm_fod = os.path.join(tmpdir, "tmp_wm_fod.mif")
                tmp_gm_fod = os.path.join(tmpdir, "tmp_gm_fod.mif")
                tmp_csf_fod = os.path.join(tmpdir, "tmp_csf_fod.mif")

                print("performing SS3T-CSD")
                call(
                    "/home/lars/external_code/MRtrix3Tissue/bin/ss3t_csd_beta1 %s %s %s %s %s %s %s -mask %s -force -nthreads 20" % (
                        dwi_preproc_fname,
                        avg_wm_response_fname,
                        tmp_wm_fod,
                        avg_gm_response_fname,
                        tmp_gm_fod,
                        avg_csf_response_fname,
                        tmp_csf_fod,
                        mask_fname), shell=True)

                # normalize intensities to allow comparison between subjects
                call("/home/lars/external_code/MRtrix3Tissue/bin/mtnormalise %s %s %s %s %s %s -mask %s -force" % (
                    tmp_wm_fod,
                    output_wm_fod_fname,
                    tmp_gm_fod,
                    output_gm_fod_fname,
                    tmp_csf_fod,
                    output_csf_fod_fname,
                    mask_fname), shell=True)

        tt_dir = os.path.join(sess_dir, "tt")
        if not os.path.exists(tt_dir):
            os.mkdir(tt_dir)

        hacked_gmwmi_fname = os.path.join(tt_dir, "hacked_gmwmi.nii.gz")
        if DO_5tt and (not os.path.exists(hacked_gmwmi_fname) or OVERWRITE):
            # generate 5-tissue image for ACT, but based on SS3T estimates for WM, GM and CSF (cf. Smolders et al. in CDMRI 2024)
            print("hacking 5tt")
            T1_path = os.path.join(sess_dir, "raw", "T1.nii")
            b0_path = os.path.join(sess_dir, "b0.nii.gz")

            fivett_path = os.path.join(tt_dir, "5tt.nii.gz")
            fivett_dwireg_path = os.path.join(tt_dir, "5tt_dwireg.nii.gz")
            dgm_dwireg_path = os.path.join(tt_dir, "dgm_dwireg.nii.gz")
            output_hacked_5tt = os.path.join(tt_dir, "hacked_5tt.nii.gz")
            threett_path = os.path.join(tt_dir, "3tt.nii.gz")

            t1_dwireg_ants_warp_fname = os.path.join(sess_dir, "reg", "T1_dwi_warp.nii.gz")
            t1_dwireg_ants_affine_fname = os.path.join(sess_dir, "reg", "T1_dwi_affine.mat")
            # take deep gray matter segmentation (thalamus etc.) from T1-based 5ttgen and register to DWI space
            extract_dgm_dwireg(T1_path, b0_path, t1_dwireg_ants_warp_fname, t1_dwireg_ants_affine_fname, fivett_path, fivett_dwireg_path, dgm_dwireg_path)

            # hack together a 5tt image, sourcing WM, GM and CSF estimates from DWI data and the deep GM from T1 data
            hack_5tt(output_wm_fod_fname, output_gm_fod_fname, output_csf_fod_fname, dgm_dwireg_path, mask_fname, output_hacked_5tt, threett_path)

            # generate the GM-WM interface needed for ACT
            generate_gmwmi(output_hacked_5tt, hacked_gmwmi_fname)

        tract_dir = os.path.join(sess_dir, "tracts")
        if not os.path.exists(tract_dir):
            os.mkdir(tract_dir)
        filtered_tck_fname = os.path.join(tract_dir, "tractogram_4m.tck")
        hacked_5tt_img_fname = os.path.join(tt_dir, "hacked_5tt.nii.gz")
        wm_fod_fname = os.path.join(csd_dir, "wm_fod_avg_response.mif")
        if DO_TRACTOGRAPHY and (not os.path.exists(filtered_tck_fname) or OVERWRITE):
            # generate 4 million streamlines whole brain tractogram with ACT
            print("constructing tractogram")
            hacked_gmwmi_fname = os.path.join(tt_dir, "hacked_gmwmi.nii.gz")
            unfiltered_tck_fname = os.path.join(tract_dir, "tractogram_4m_unfiltered.tck")

            perform_act(wm_fod_fname, hacked_5tt_img_fname, hacked_gmwmi_fname, unfiltered_tck_fname, 4000000)

        weights_fname = os.path.join(tract_dir, "SIFT2_4m_weights.txt")
        mu_fname = os.path.join(tract_dir, "SIFT2_4m_mu.txt")
        if DO_SIFT and (not os.path.exists(weights_fname) or OVERWRITE):
            # generate weights with SIFT2
            print("calculating SIFT2 weights")
            perform_sift2(filtered_tck_fname, hacked_5tt_img_fname, wm_fod_fname, weights_fname, mu_fname)

        output_matrix_fname = os.path.join(tract_dir, "structural_matrix_slant_4m.npy")
        if DO_CONNECTOME and (not os.path.exists(output_matrix_fname) or OVERWRITE):
            # construct connectomes by summing SIFT2 weights between each ROI in the SLANT atlas
            print("constructing connectome")
            atlas_fname = os.path.join(sess_dir, "parc", "slant_atlas_dwireg.nii.gz")
            output_assignments_fname = os.path.join(tract_dir, "structural_assignments_slant_4m.txt")
            construct_connectome(filtered_tck_fname, atlas_fname, weights_fname, mu_fname, output_matrix_fname,
                                 output_assignments_fname)
