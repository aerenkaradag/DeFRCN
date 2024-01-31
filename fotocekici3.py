from ctypes import Array
import os
# import numpy as np
import xml.etree.ElementTree as ET
import cv2
import shutil
import random
from random import randint
import linecache
import argparse
import glob
import os
import zipfile
from bs4 import BeautifulSoup
parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, required=True)
parser.add_argument('--cevat', action='store_true')
parser.add_argument('--random', type=bool, required=False)
parser.add_argument('--sirali', type=bool, required=False)
parser.add_argument('--custom', action='store_true')
parser.add_argument('--no-custom', dest='feature', action='store_false')
parser.add_argument('--video', type=str, required = True)
#parser.add_argument('--resimsayisi', type=int, required = True)

def get_object_dimensions(tree, object_name):
    # Iterate over all 'object' elements in the XML tree
    for obj in tree.iter('object'):
        # If the object's name matches the given name
        if obj.find('name').text == object_name:
            # Extract and return the bounding box dimensions
            bndbox = obj.find('bndbox')
            width = int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)
            height = int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
            return width, height
    # If no matching object was found, return None
    return None

def add_background_object(tree, image_width, image_height, object_area):
    # Calculate a random width and height for the new object, ensuring the area matches
    new_width = random.randint(1, image_width)
    new_height = int(object_area / new_width)

    # Calculate random coordinates for the new object
    xmin = 3#random.randint(0, image_width - new_width)
    ymin = 3#random.randint(0, image_height - new_height)
    xmax = xmin + new_width
    ymax = ymin + new_height

    # Create the new 'object' element and its child elements
    new_obj = ET.SubElement(tree.getroot(), 'object')
    ET.SubElement(new_obj, 'name').text = "background"
    ET.SubElement(new_obj, 'pose').text = "Unspecified"
    ET.SubElement(new_obj, 'truncated').text = "0"
    ET.SubElement(new_obj, 'difficult').text = "0"
    bndbox = ET.SubElement(new_obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(xmin)
    ET.SubElement(bndbox, 'ymin').text = str(ymin)
    ET.SubElement(bndbox, 'xmax').text = str(xmax)
    ET.SubElement(bndbox, 'ymax').text = str(ymax)

def add_background_to_xml(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)

    # Get the image width and height from the XML
    size = tree.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    # Get the dimensions of the 'newobject' object
    dimensions = get_object_dimensions(tree, 'newobject')

    if dimensions is not None:
        # Calculate the area of the 'newobject' object
        width, height = dimensions
        object_area = width * height

        # Add a new object with the name "background" to the XML
        add_background_object(tree, image_width, image_height, object_area)

        # Write the changes back to the XML file
        tree.write(xml_path)
    else:
        print(f"No object named 'newobject' found in {xml_path}")

def sayi_sec(gercekresim):
    return randint(1, gercekresim)

def dogrusayi(sayi):
  src = os.path.join(os.getcwd(),"assets","groundtruths","groundtruth{}.txt".format(args.video)) 
  count = 0
  a = 0
  file1 = open(src, 'r')
  Line = file1.readlines()
  #print(len(Line))
  print("sayi: {}".format(sayi))
  for lines in Line:
    txt = lines.split(',')
    a = a + 1
    if(txt[0] != "nan" and int(float(txt[0])) != 0):
      count = count + 1
    if count == sayi:
      print("dogru sayi: {}".format(a))
      return a




def sayi_sec1(gercekresim, sayi):
  a = 0
  count = 0
  while a < gercekresim:
    a = a + 1
    txt = linecache.getline(os.path.join(os.getcwd(),"assets", "groundtruths","groundtruth{}.txt".format(args.video)), a)
    txt = txt.split(",")
    if(txt[0] != "nan" and int(float(txt[0])) != 0):
      count = count + 1
  
  resimnumarasi = 1
  c = []
  print("count: {}".format(count))
  b = 0

  while b < args.count:
    resimnumarasi = int(resimnumarasi + ((count - 1) / sayi))
    print(resimnumarasi)
    c.append(dogrusayi(resimnumarasi))
    b = b + 1
  assert None not in c
  return c


def dogrudizim(number):
    if (number < 10):
        return "00000" + str(number)
    elif (number < 100):
        return "0000" + str(number)
    elif (number < 1000):
        return "000" + str(number)
    elif (number < 5000):
        return "00" + str(number)
    else:
        return "" + str(number)


def okumaca(number, video_name, num):
    src = os.path.join(os.getcwd(),"assets", "sample.xml")
    src_emp = os.path.join(os.getcwd(),"assets", "sample_empty.xml")
    c = []

    a = 0
    while (number > 0):
        a = a + 1
        txt = linecache.getline(os.path.join(os.getcwd(),"assets", "groundtruths","groundtruth{}.txt".format(video_name)), a)
        
        txt = txt.split(",")
        #,print(int(float(txt[0])) == 0)
        if(str(txt[0]) != "nan" and int(float(txt[0])) != 0):
          #print(a)
          c.append(a)
          txt[0] = (int(float(txt[0])))
          txt[1] = (int(float(txt[1])))
          txt[2] = (int(float(txt[2])))
          txt[3] = (int(float(txt[3][:-1])))
          dest = os.path.join("VOC2007", "Annotations", dogrudizim(num) + ".xml")
          shutil.copyfile(src, dest)
          tree = ET.parse(dest)
          root = tree.getroot()
          root.find("filename").text = dogrudizim(num) + ".jpg"
          root.find("source").find("flickrid").text = str(randint(0, 100000))
          root = root.find("object")
          root.find("name").text = video_name
          root = root.find("bndbox")
          root.find("xmin").text = str(txt[0])
          root.find("ymin").text = str(txt[1])
          root.find("xmax").text = str(txt[0] + txt[2])
          root.find("ymax").text = str(txt[1] + txt[3])
          tree.write(dest)
          num = num + 1
        else: 
          c.append(a)
          dest = os.path.join("VOC2007", "Annotations", dogrudizim(num) + ".xml")
          
          shutil.copyfile(src_emp, dest)
          tree = ET.parse(dest)
          root = tree.getroot()
          root.find("filename").text = dogrudizim(num) + ".jpg"
          root.find("source").find("flickrid").text = str(randint(0, 100000))
          tree.write(dest)
          num = num + 1
        number = number - 1
    print(num)    
    return c, num 


def ad_ayarla(number, c, video_name, num):
    a = 0
    f = open(f'{video_name}.txt', 'w')
    print("gen: {}".format(len(c)))
    while (number > a):
        src = os.path.join("assets", "video_resim", video_name ,str(c[a]).zfill(8) + ".jpg")
        dest = os.path.join("VOC2007", "JPEGImages", str(dogrudizim(num)) + ".jpg")

        f.write(str(dogrudizim(num)) + "\n")
        shutil.copyfile(src, dest)
        a = a + 1
        num = num + 1
    f.write(str(dogrudizim(num)) + "\n")
    return num

def main(args):
    num = 9975
    count = 0
    if args.cevat:
        dest = os.path.join(os.getcwd(), "datasets", "VOC2007")
        for number in [1]:
          #print("girdigirdi")
          src = os.path.join("datasets", os.path.join("VOC2007", "JPEGImages", str(dogrudizim(number)) + ".jpg"))
          dst = os.path.join(dest, "JPEGImages", str(dogrudizim(num + count) + ".jpg"))
          shutil.copyfile(src, dst)

          src = os.path.join("datasets", os.path.join("VOC2007", "Annotations", str(dogrudizim(number)) + ".xml"))
          dst = os.path.join(dest, "Annotations", str(dogrudizim(num+ count)) + ".xml")
          shutil.copyfile(src,dst)

          num = num + 1
          count = count + 1
    else:
      src = os.path.join(os.getcwd(), "assets", "sample.xml")
      src_emp = os.path.join(os.getcwd(), "assets", "sample_empty.xml")
      
      img = cv2.imread(os.path.join("assets", "video_resim", args.video,"00000001.jpg"))
      
      height, width = img.shape[0], img.shape[1]
      tree = ET.parse(src)
      tree2 = ET.parse(src_emp)
      root = tree.getroot()
      root2 = tree2.getroot()
      root = root.find("size")
      root2 = root2.find("size")
      root.find("width").text = str(width)
      root.find("height").text = str(height)
      root2.find("width").text = str(width)
      root2.find("height").text = str(height)
      tree.write(src)
      tree2.write(src_emp)
      rand = 0	
      num = 9975
      foto = 0
      foto = args.count
      if args.custom:
        if os.path.isdir("VOC2007"):
            shutil.rmtree("VOC2007")
        if os.path.isdir("vocsplit"):
            shutil.rmtree("vocsplit")
        os.mkdir("vocsplit")
        os.mkdir("VOC2007")
        os.mkdir("VOC2007/Annotations")
        os.mkdir("VOC2007/JPEGImages")
        os.mkdir("VOC2007/ImageSets")
        os.mkdir("VOC2007/ImageSets/Layout")
        os.mkdir("VOC2007/ImageSets/Main")
        print("klasörler oluşturuldu")
        video_list = os.listdir("assets/video_resim")
        start_numer = 1
        for video_name in video_list:
            resimsayisi = len(os.listdir(os.path.join(os.getcwd(), "assets", "video_resim", video_name)))
            c , start_num = okumaca(resimsayisi, video_name, start_numer)
            num = ad_ayarla(resimsayisi - 1, c, video_name, start_numer)

            # Iterate over the numbers 0 through 29
            for i in range(30):
                # Create the name of the seed folder
                folder_name = f"seed{i}"
                # Create the full path to the seed folder
                folder_path = os.path.join(os.getcwd(), "vocsplit", folder_name)
                # Create the seed folder
                os.makedirs(folder_path, exist_ok=True)
                # Iterate over the numbers 1 through 10 for each text file
                for shot in range(1, 11):
                    # Create the full path to the text file
                    file_path = os.path.join(folder_path, f"box_{shot}shot_{video_name}_train.txt")
                    
                    # Create and write to the text file
                    with open(file_path, "w") as f:
                        for img_num in range(start_numer, shot + start_numer):
                            # The content to be written in each text file
                            content = f"datasets/VOC2007/JPEGImages/{str(img_num).zfill(6)}.jpg\n"
                            f.write(content)
            
            start_numer = start_num
            # Check if "9.txt" already exists
            if os.path.exists(os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout", "train.txt")):
                with open(f'{video_name}.txt', 'r') as source_file:
                    data = source_file.read()
                with open(os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout", "train.txt"), 'a') as dest_file:
                    dest_file.write(data)
            else:
                # If it doesn't exist, create it
                shutil.copy2(f'{video_name}.txt', os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout", "train.txt"))

            shutil.copyfile(os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout", "train.txt"), os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout", "trainval.txt"))
            shutil.copyfile(os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout", "train.txt"), os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout", "test.txt"))
            shutil.copyfile(os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout", "train.txt"), os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout", "val.txt"))            
            shutil.move(f"{video_name}.txt", os.path.join(os.path.join(os.getcwd(), "VOC2007", "ImageSets" ), "Main"))
            shutil.copyfile(os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Main", f"{video_name}.txt"), os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Main", "train.txt"))
            shutil.copyfile(os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Main", f"{video_name}.txt"), os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Main", "trainval.txt"))
            shutil.copyfile(os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Main", f"{video_name}.txt"), os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Main", "test.txt"))
            shutil.copyfile(os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Main", f"{video_name}.txt"), os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Main", "val.txt"))
            # Define source and destination directories
            # source_directory = os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Main")
            # destination_directory = os.path.join(os.getcwd(), "VOC2007", "ImageSets", "Layout")
            # # Define the filenames
            # source_filename = f"{video_name}.txt"
            # destination_filenames = ["test.txt", "train.txt", "val.txt", "trainval.txt"]
            # # Copy and rename the files
            # for destination_filename in destination_filenames:
            #     source_path = os.path.join(source_directory, source_filename)
            #     destination_path = os.path.join(destination_directory, destination_filename)
            #     destination_path2 = os.path.join(source_directory, destination_filename)
            #     shutil.copyfile(source_path, destination_path)
            #     shutil.copyfile(source_path, destination_path2)
        print("Files copied and renamed successfully.")
      if foto != 0:
          a = 0
          dest = os.path.join(os.getcwd(), "datasets", "VOC2007")
          if os.path.isdir(dest):
              shutil.rmtree(dest)
          dest2 = os.path.join("datasets","VOC2007")
          shutil.move("VOC2007",dest2)

          path2 = os.path.join("datasets", "vocsplit")
          if os.path.isdir(path2):
              shutil.rmtree(path2)
          dest2 = os.path.join("datasets","vocsplit")
          shutil.move("vocsplit",dest2)

          if args.random:
              rand = [1] #sayi_sec(args.count)
          if args.sirali:
              print("buradayım")
              rand = sayi_sec1(resimsayisi,args.count) 
          print(rand)
          
          count = 0
          for number in rand:
            #print("girdigirdi")
            src = os.path.join("datasets", os.path.join("VOC2007", "JPEGImages", str(dogrudizim(number)) + ".jpg"))
            dst = os.path.join(dest, "JPEGImages", str(dogrudizim(num + count) + ".jpg"))
            shutil.copyfile(src, dst)

            src = os.path.join("datasets", os.path.join("VOC2007", "Annotations", str(dogrudizim(number)) + ".xml"))
            dst = os.path.join(dest, "Annotations", str(dogrudizim(num+ count)) + ".xml")
            shutil.copyfile(src,dst)

            num = num + 1
            count = count + 1
      #add_background_to_xml(dst)
            
def download_videos():
    with open('./assets/videos.txt', 'r') as file:
        for line in file:
            video_name, video_url = line.strip().split(": ")
            os.system(f'curl -L "{video_url}" -o "{video_name}.zip"')
            os.makedirs(f"assets/video_resim/{video_name}", exist_ok=True)
            with zipfile.ZipFile(f"{video_name}.zip", 'r') as zip_ref:
                zip_ref.extractall(f"assets/video_resim/{video_name}")
            os.remove(f"{video_name}.zip")
    print("videos downloaded")


def download_groundtruths():
    # for LT videos: https://drive.google.com/uc?id=1-q1UafYST5_24s6UJpLCnnIX3CrbuhVy
    # for ST videos: https://drive.google.com/file/d/1YdNAtXlzlnOqTTkvum9onRG16J-dt0E0
    os.system('gdown "https://drive.google.com/uc?id=1-q1UafYST5_24s6UJpLCnnIX3CrbuhVy" -O groundtruths.zip')
    with zipfile.ZipFile('groundtruths.zip', 'r') as zip_ref:
        zip_ref.extractall('assets/')
    os.remove('groundtruths.zip')
    print("groundtruth downloaded")

if __name__ == "__main__":
    download_groundtruths()
    download_videos()
    args = parser.parse_args()
    main(args)