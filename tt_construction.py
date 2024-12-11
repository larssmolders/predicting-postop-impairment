import os
from subprocess import call, DEVNULL
import nibabel as nib
import numpy as np
from tempfile import TemporaryDirectory
from nilearn.image import resample_to_img

# extract deep gray matter (DGM) segmentation from 5ttgen and register to DWI space
def extract_dgm_dwireg(T1_fname, b0_fname, t1_dwireg_ants_warp_fname, t1_dwireg_ants_affine_fname, fivett_fname, fivett_dwireg_fname, output_dgm_dwireg_fname):
    with TemporaryDirectory() as tmpdir:
        # generate T1-based 5tt image
        call("5ttgen fsl %s %s -force -nthreads 20" % (T1_fname, fivett_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)

        # extract the DGM tissue type
        dgm_fname = os.path.join(tmpdir, "dgm.nii.gz")
        call("mrconvert %s -coord 3 1 %s" % (fivett_fname, dgm_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)

        # register to DWI space
        call("antsApplyTransforms -d 3 -i %s -r %s -o %s -t %s %s" % (dgm_fname, b0_fname, output_dgm_dwireg_fname, t1_dwireg_ants_warp_fname, t1_dwireg_ants_affine_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)


# construct a 5tt image from DWI FODs and T1-based deep gray matter
def hack_5tt(wm_fod_fname, gm_fod_fname, csf_fod_fname, dgm_dwireg_fname, mask_fname, output_hacked_5tt, ttt_output, keep_temp_files=True):
    with TemporaryDirectory() as tmpdir:
        if keep_temp_files:
            tmpdir = os.path.join(os.path.dirname(output_hacked_5tt), "intermediates")
            if not os.path.exists(tmpdir):
                os.mkdir(tmpdir)
        if wm_fod_fname.endswith(".mif"):
            new_wm_fod_fname = os.path.join(tmpdir, "wm_fod.nii.gz")
            call("mrconvert %s %s -force" % (wm_fod_fname, new_wm_fod_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)
            wm_fod_fname = new_wm_fod_fname
        if gm_fod_fname.endswith(".mif"):
            new_gm_fod_fname = os.path.join(tmpdir, "gm_fod.nii.gz")
            call("mrconvert %s %s -force" % (gm_fod_fname, new_gm_fod_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)
            gm_fod_fname = new_gm_fod_fname
        if csf_fod_fname.endswith(".mif"):
            new_csf_fod_fname = os.path.join(tmpdir, "csf_fod.nii.gz")
            call("mrconvert %s %s -force" % (csf_fod_fname, new_csf_fod_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)
            csf_fod_fname = new_csf_fod_fname
        if mask_fname.endswith(".mif"):
            new_mask_fname = os.path.join(tmpdir, "mask.nii.gz")
            call("mrconvert %s %s -force" % (mask_fname, new_mask_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)
            mask_fname = new_mask_fname

        # extract the first WM coefficient, corresponding to the WM FOD amplitude
        wm_fod_0_fname = os.path.join(tmpdir, "wm_fod_0.nii.gz")
        call("mrconvert %s -coord 3 0 %s -force" % (wm_fod_fname, wm_fod_0_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)
        # create 3-tissue image (WM, GM, CSF) based on DWI
        call("mrcat %s %s %s %s -force" % (wm_fod_0_fname, gm_fod_fname, csf_fod_fname, ttt_output), shell=True, stderr=DEVNULL, stdout=DEVNULL)

        wm_fod_0 = nib.load(wm_fod_0_fname)
        gm_fod = nib.load(gm_fod_fname)
        csf_fod = nib.load(csf_fod_fname)
        mask = nib.load(mask_fname).get_fdata() == 1.0
        diffusion_3tt_img = np.array([wm_fod_0.get_fdata(), gm_fod.get_fdata(), csf_fod.get_fdata()])

        # take max. amplitude across tissues for each voxel as segmentation label
        max_img = np.argmax(diffusion_3tt_img, axis=0)
        max_img[mask] = max_img[mask] + 1.0
        wm_max_fname = os.path.join(tmpdir, "wm_max.nii.gz")
        gm_max_fname = os.path.join(tmpdir, "gm_max.nii.gz")
        nib.save(nib.Nifti1Image((max_img == 1.0).astype(float), affine=wm_fod_0.affine), wm_max_fname)
        nib.save(nib.Nifti1Image((max_img == 2.0).astype(float), affine=wm_fod_0.affine), gm_max_fname)

        # apply connected components filter to find and fix holes in WM
        gm_max_cc_fname = os.path.join(tmpdir, "gm_max_cc.nii.gz")
        call("maskfilter %s connect -largest %s -force" % (gm_max_fname, gm_max_cc_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)

        # fill GM regions that are disconnected from the main (cortical) GM with WM
        gm_holes = nib.load(gm_max_fname).get_fdata() - nib.load(gm_max_cc_fname).get_fdata()
        wm_filled_holes = nib.load(wm_max_fname).get_fdata() + gm_holes
        wm_filled_holes_fname = os.path.join(tmpdir, "wm_filled_holes.nii.gz")
        nib.save(nib.Nifti1Image(wm_filled_holes, affine=wm_fod_0.affine), wm_filled_holes_fname)
        wm_filled_holes_cc_fname = os.path.join(tmpdir, "wm_filled_holes_cc.nii.gz")
        call("maskfilter %s connect -largest %s -force" % (wm_filled_holes_fname, wm_filled_holes_cc_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)

        # concatenate the tissue types into a 5tt image
        gm_max_cc = nib.load(gm_max_cc_fname).get_fdata()
        dgm = nib.load(dgm_dwireg_fname).get_fdata()
        wm_filled_holes_cc = nib.load(wm_filled_holes_cc_fname).get_fdata()
        csf = csf_fod.get_fdata()
        empty = np.zeros(wm_fod_0.shape)

        if len(dgm.shape) != len(gm_max_cc.shape):
            dgm = dgm[..., np.newaxis]

        if dgm.shape != gm_max_cc.shape:
            dgm_nii = nib.load(dgm_dwireg_fname)
            gm_max_cc_nii = nib.load(gm_max_cc_fname)
            dgm = resample_to_img(dgm_nii, gm_max_cc_nii, interpolation="nearest").get_fdata()

        hacked_5tt_img = np.array([gm_max_cc, dgm, wm_filled_holes_cc, csf, empty])
        hacked_5tt_img = np.moveaxis(hacked_5tt_img, 0, 3)[..., 0]

        summed_intensities = np.sum(hacked_5tt_img, axis=3)
        # set empty holes to WM
        if mask.shape != summed_intensities.shape:
            mask = mask[..., np.newaxis]
        holes = np.logical_and(summed_intensities == 0.0, mask)[..., 0]

        hacked_5tt_img[holes, 2] = 1.0

        # normalize intensities
        summed_intensities = np.sum(hacked_5tt_img, axis=3)
        hacked_5tt_img[..., 0] = hacked_5tt_img[..., 0] / summed_intensities
        hacked_5tt_img[..., 1] = hacked_5tt_img[..., 1] / summed_intensities
        hacked_5tt_img[..., 2] = hacked_5tt_img[..., 2] / summed_intensities
        hacked_5tt_img[..., 3] = hacked_5tt_img[..., 3] / summed_intensities
        hacked_5tt_img[..., 4] = hacked_5tt_img[..., 4] / summed_intensities

        # ensure unmasked area is 0
        hacked_5tt_img[np.logical_not(mask)] = 0.0
        hacked_5tt_img[dgm[..., 0] > 0.5, 1] = 1.0
        hacked_5tt_img[dgm[..., 0] > 0.5, 0] = 0.0
        hacked_5tt_img[dgm[..., 0] > 0.5, 2] = 0.0
        hacked_5tt_img[dgm[..., 0] > 0.5, 3] = 0.0
        hacked_5tt_img[dgm[..., 0] > 0.5, 4] = 0.0

        nib.save(nib.Nifti1Image(hacked_5tt_img, affine=wm_fod_0.affine), output_hacked_5tt)

def generate_gmwmi(tt_img_fname, gmwmi_output_fname):
    call("5tt2gmwmi %s %s -force" % (tt_img_fname, gmwmi_output_fname), shell=True, stderr=DEVNULL, stdout=DEVNULL)
