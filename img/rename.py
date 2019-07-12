import os 
import random

def renameAllinOne():
    i = 0
    folder = "FPS"
    fdir = "./" + folder

    #name = folder + "_" + str(j) + "/"
    ldir = os.listdir(fdir)
    random.shuffle(ldir)
    print(ldir)
    for filename in ldir: 
        dst = fdir + "/" + folder + "_" + format(i, '05') + ".png"
        src = fdir + "/" + filename 
        os.rename(src, dst) 
        i += 1
def renameSmallinOne():
    end = "./RTS"
    game = "./RTS"
    folder = "./RTS/RTS"
    fdir = "./" + folder
    i = 0
    for j in range(2):
        fdir =  folder + "_" + str(j)
        ldir = os.listdir(fdir)
        random.shuffle(ldir)
        for filename in ldir: 
            dst =  folder + "_" + format(i, '08') + ".png"
            src = fdir + "/" + filename 
            print(dst)
            print(src)
            os.rename(src, dst) 
            i += 1

#renameSmallinOne()
renameAllinOne()