from xml.dom import minidom
import os


PATH =  os.getcwd() + "/VOCdevkit/VOC2010/Annotations"
allFiles = []
print(PATH)

for root, dirs, filenames in os.walk(PATH):
    for f in filenames:
        if f.endswith('.xml'):
            allFiles.append(os.path.join(root, f))

#print(allFiles)

mainDic = {}


for f in allFiles:
    xmldoc = minidom.parse(f)
    width = xmldoc.getElementsByTagName('width')
    height = xmldoc.getElementsByTagName('height')

    name = xmldoc.getElementsByTagName('name')
    file_name = xmldoc.getElementsByTagName('filename')
    ymin = xmldoc.getElementsByTagName('ymin')
    ymax = xmldoc.getElementsByTagName('ymax')
    xmin = xmldoc.getElementsByTagName('xmin')
    xmax = xmldoc.getElementsByTagName('xmax')
    k = 0
    print(len(name))
    dic = {}
    for i in name:
        j = i.firstChild.nodeValue
        dim = []
        dim.extend([xmin[k].firstChild.nodeValue, xmax[k].firstChild.nodeValue, ymin[k].firstChild.nodeValue, ymax[k].firstChild.nodeValue])
        
        dic[j] = dim
        mainDic[file_name[0].firstChild.nodeValue] = dic
        k+=1
        #print(j)



