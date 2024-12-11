import nibabel as nib
import numpy as np
from subprocess import call, DEVNULL
import os
from shutil import rmtree
from GPUtil import getAvailable

# conversion dict for atlas ROI labels -> SLANT segmentation labels
slant_labels = {
    1: 23, 2: 30, 3: 31, 4: 32, 5: 36, 6: 37, 7: 35, 8: 47, 9: 48, 10: 55, 11: 56, 12: 57, 13: 58, 14: 59, 15: 60,
    16: 75, 17: 76, 18: 100, 19: 101, 20: 102, 21: 103, 22: 104, 23: 105, 24: 106, 25: 107, 26: 108, 27: 109, 28: 112,
    29: 113, 30: 114, 31: 115, 32: 116, 33: 117, 34: 118, 35: 119, 36: 120, 37: 121, 38: 122, 39: 123, 40: 124, 41: 125,
    42: 128, 43: 129, 44: 132, 45: 133, 46: 134, 47: 135, 48: 136, 49: 137, 50: 138, 51: 139, 52: 140, 53: 141, 54: 142,
    55: 143, 56: 144, 57: 145, 58: 146, 59: 147, 60: 148, 61: 149, 62: 150, 63: 151, 64: 152, 65: 153, 66: 154, 67: 155,
    68: 156, 69: 157, 70: 160, 71: 161, 72: 162, 73: 163, 74: 164, 75: 165, 76: 166, 77: 167, 78: 168, 79: 169, 80: 170,
    81: 171, 82: 172, 83: 173, 84: 174, 85: 175, 86: 176, 87: 177, 88: 178, 89: 179, 90: 180, 91: 181, 92: 182, 93: 183,
    94: 184, 95: 185, 96: 186, 97: 187, 98: 190, 99: 191, 100: 192, 101: 193, 102: 194, 103: 195, 104: 196, 105: 197,
    106: 198, 107: 199, 108: 200, 109: 201, 110: 202, 111: 203, 112: 204, 113: 205, 114: 206, 115: 207
}

# atlas labels corresponding to DMN and FPN derived by overlap with Yeo's 7 networks
DMN_indices = np.array([17, 18, 23, 24, 35, 36, 47, 48, 51, 52, 63, 64, 65, 66, 73, 74, 75, 76, 77, 78, 97, 98, 108])
FPN_indices = np.array([21, 22, 53, 54, 71, 72, 101, 102, 111, 112])
thalamus_indices = [13, 14]

DMN_names = ["right anterior cingulate gyrus", "left anterior cingulate gyrus",
             "right angular gyrus", "left angular gyrus", "right frontal pole", "left frontal pole",
             "right lateral orbital gyrus", "left lateral orbital gyrus", "right medial frontal cortex",
             "left medial frontal cortex", "right medial superior frontal gyrus", "left medial superior frontal gyrus",
             "right middle temporal gyrus", "left middle temporal gyrus", "right pars orbitalis",
             "left pars orbitalis", "right posterior cingulate gyrus", "left posterior cingulate gyrus", "right precuneus",
             "left precuneus", "right lateral superior frontal gyrus", "left lateral superior frontal gyrus", "left superior temporal gyrus"]

FPN_names = ["right anterior orbital gyrus", "left anterior orbital gyrus",
             "right middle frontal gyrus", "left middle frontal gyrus", "right pars opercularis",
             "left pars opercularis", "right supramarginal gyrus", "left supramarginal gyrus",
             "right pars triangularis", "left pars triangularis"]

thalamus_names = ["right thalamus", "left thalamus"]


# run SLANT segmentation using docker (https://github.com/MASILab/SLANTbrainSeg)
def run_slant(input_t1, output_seg):
    tmpdir = os.path.join(os.path.dirname(input_t1), "slant_temp")
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    in_tmp = os.path.join(tmpdir, "in")
    out_tmp = os.path.join(tmpdir, "out")

    # prepare temporary directory structure
    if not os.path.exists(in_tmp):
        os.mkdir(in_tmp)
    if not os.path.exists(out_tmp):
        os.mkdir(out_tmp)

    # copy raw T1 file to input dir
    call("cp %s %s" % (input_t1, in_tmp), shell=True)
    if not input_t1.endswith(".gz"):
        call("gzip -f %s" % os.path.join(in_tmp, "T1.nii"), shell=True)
    call("ls %s" % in_tmp, shell=True)

    # find a GPU
    device_str = ",".join([str(a) for a in getAvailable()])

    if device_str == "":
        device_str = "1"
    print("running SLANT on device %s" % device_str)

    # run SLANT with docker
    call("docker run -it --rm --gpus device=%s -v %s:/INPUTS/ -v %s:/OUTPUTS/ masidocker/public:deep_brain_seg_v1_1_0 /bin/bash -c \"/extra/run_deep_brain_seg.sh; chmod 777 -R /OUTPUTS/; chmod 777 -R /INPUTS/\"" % (device_str, in_tmp, out_tmp),
         shell=True, stdout=DEVNULL, stderr=DEVNULL)

    slant_out_name = os.path.join(out_tmp, "FinalResult", os.path.basename(input_t1).split('.')[0] + "_seg.nii.gz")

    # copy output and remove temp dirs
    call("cp %s %s" % (slant_out_name, output_seg), shell=True)
    rmtree(tmpdir)

# convert SLANT segmentation (including e.g. WM and ventricles) into atlas of ROIs
def slant_to_atlas(input_seg_fname, output_fname):
    seg = nib.load(input_seg_fname)
    seg_data = seg.get_fdata()
    output_img = np.zeros(seg_data.shape)
    # use slant_labels dict to convert segmentation into atlas
    for i in slant_labels.keys():
        vox = np.argwhere(seg_data == slant_labels[i])
        output_img[vox[:, 0], vox[:, 1], vox[:, 2]] = i

    nib.save(nib.Nifti1Image(output_img, seg.affine), output_fname)
