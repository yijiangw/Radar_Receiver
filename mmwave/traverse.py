'''
@Author: Yijiang Wang
@Date: 2020-06-11 23:20:05
@LastEditTime: 2020-06-11 23:42:01
@LastEditors: Yijiang Wang
@FilePath: /mmwave/traverse.py
@Description: 
'''
# Import the os module, for the os.walk function
import os

from FirstFrameForSync import findSyncFrame
# Set the directory you want to start from
rootDir = os.path.abspath(os.getcwd())
suffix = ".bin"
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname.find(suffix)>0:
            findSyncFrame(dirName+"/"+fname)


