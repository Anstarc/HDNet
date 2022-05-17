import os, glob, sys


# Print an error message and quit
def printError(message):
    print('ERROR: {}'.format(message))
    sys.exit(-1)


def main(contain_img=True):
    # Where to look for Cityscapes
    cityscapesPath = os.environ['CITYSCAPES_DATASET']
    # how to search for all ground truth
    searchTrainFine = os.path.join(cityscapesPath, "gtFine", "train", "*", "*_gt*_labelTrainIds.png")
    searchValFine = os.path.join(cityscapesPath, "gtFine", "val", "*", "*_gt*_labelTrainIds.png")
    searchTestFine = os.path.join(cityscapesPath, "gtFine", "test", "*", "*_gt*_labelTrainIds.png")
    if contain_img:
        searchTrainImg = os.path.join(cityscapesPath, "leftImg8bit", "train", "*", "*_leftImg8bit.png")
        searchValImg = os.path.join(cityscapesPath, "leftImg8bit", "val", "*", "*_leftImg8bit.png")
        searchTestImg = os.path.join(cityscapesPath, "leftImg8bit", "test", "*", "*_leftImg8bit.png")

    # search files
    filesTrainFine = glob.glob(searchTrainFine)
    filesTrainFine.sort()
    filesValFine = glob.glob(searchValFine)
    filesValFine.sort()
    filesTestFine = glob.glob(searchTestFine)
    filesTestFine.sort()
    if contain_img:
        filesTrainImg = glob.glob(searchTrainImg)
        filesTrainImg.sort()
        filesValImg = glob.glob(searchValImg)
        filesValImg.sort()
        filesTestImg = glob.glob(searchTestImg)
        filesTestImg.sort()

    # quit if we did not find anything
    if not filesTrainFine:
        printError("Did not find any gtFine/train files.")
    if not filesValFine:
        printError("Did not find any gtFine/val files.")
    if contain_img:
        if not filesTrainImg:
            printError("Did not find any leftImg8bit/train files.")
        if not filesValImg:
            printError("Did not find any leftImg8bit/val files.")
        if not filesTestImg:
            printError("Did not find any leftImg8bit/test files.")

    # assertion
    if contain_img:
        assert len(filesTrainImg) == len(filesTrainFine), \
            "Error %d (filesTrainImg) != %d (filesTrainFine)" % (len(filesTrainImg), len(filesTrainFine))
        assert len(filesValImg) == len(filesValFine), \
            "Error %d (filesValImg) != %d (filesValFine)" % (len(filesValImg), len(filesValFine))
        assert len(filesTestImg) == 1525 and len(filesTestImg) == len(filesTestFine), "Error %d (filesTestImg) != 1525" % len(filesTestImg)
        # files = filesTrainFine + filesValFine + filesTrainCoarse + filesValCoarse + filesExTrainCoarse
        # assert len(files) == 26948, "Error %d (gtFiles) != 26948" % len(files)
        print('Train:', len(filesTrainFine))
        print('Val:', len(filesValFine))
        print('Test:', len(filesTestImg))

    # create txt
    dir_path = os.path.join(cityscapesPath, 'dataset')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("---create test.txt---")
    with open(os.path.join(dir_path, 'test.txt'), 'w') as f:
        for l in filesTestFine:
            f.write(l[len(cityscapesPath):] + '\n')
    print("---create train_fine.txt---")
    with open(os.path.join(dir_path, 'train_fine.txt'), 'w') as f:
        for l in zip(filesTrainImg, filesTrainFine):
            assert l[0][len(cityscapesPath+'/leftImg8bit/'):-len('_leftImg8bit.png')] \
                   == l[1][len(cityscapesPath+'/gtFine/'):-len('_gtFine_labelTrainIds.png')], \
                "%s != %s" % (l[0][len(cityscapesPath+'/leftImg8bit/'):-len('_leftImg8bit.png')], \
                              l[1][len(cityscapesPath+'/gtFine/'):-len('_gtFine_labelTrainIds.png')])
            f.write(l[0][len(cityscapesPath):] + ' ' + l[1][len(cityscapesPath):] + '\n')
    print("---create val_fine.txt---")
    with open(os.path.join(dir_path, 'val_fine.txt'), 'w') as f:
        for l in zip(filesValImg, filesValFine):
            assert l[0][len(cityscapesPath+'/leftImg8bit/'):-len('_leftImg8bit.png')] \
                   == l[1][len(cityscapesPath+'/gtFine/'):-len('_gtFine_labelTrainIds.png')], \
                "%s != %s" % (l[0][len(cityscapesPath+'/leftImg8bit/'):-len('_leftImg8bit.png')], \
                              l[1][len(cityscapesPath+'/gtFine/'):-len('_gtFine_labelTrainIds.png')])
            f.write(l[0][len(cityscapesPath):] + ' ' + l[1][len(cityscapesPath):] + '\n')

# call the main
if __name__ == "__main__":
    os.environ['CITYSCAPES_DATASET'] = '/home/data/Cityscapes/'
    main()
