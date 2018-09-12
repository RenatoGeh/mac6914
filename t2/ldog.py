import cv2
import os.path
import sys
import numpy as np

WRITE_IMGS = False
BASE = None
EXT = None

def parse_args():
  if (2 > len(sys.argv) > 3) or ("-h" in sys.argv):
    print("Usage " + sys.argv[0] + " img [-w] [-h]")
    print("  Arguments:")
    print("    img - image to apply filters")
    print("  Options:")
    print("    -w  - write images to disk")
    print("    -h  - displays this help message")
    return None, None
  base, ext = os.path.splitext(sys.argv[1])
  global WRITE_IMGS
  if "-w" in sys.argv:
    WRITE_IMGS = True
  print(WRITE_IMGS)
  return base, ext

def _ask_prompt(prompt, Options):
  print(prompt, end=' ')
  n = len(Options)
  for i, o in enumerate(Options):
    default = i == n-1
    if default:
      print('[', end='')
    print(o, end='')
    if default:
      print(']')
    else:
      print('/', end='')

def ask(prompt, Options):
  while True:
    _ask_prompt(prompt, Options)
    c = input()
    if c == '':
      return Options[-1]
    for i, o in enumerate(Options):
      if c == o or o.startswith(c):
        return o

def open_ask(prompt, n, Args):
  while True:
    print(prompt)
    s = input().split()
    if len(s) == n:
      break
  if type(Args) == type:
    t = Args
    for i, v in enumerate(s):
      s[i] = t(v)
    return s
  m = len(Args)
  lt = Args[0]
  for i, v in enumerate(s):
    if i < m:
      lt = Args[i]
    s[i] = lt(v)
  return s

def write_image(ap, img):
  if WRITE_IMGS:
    cv2.imwrite(BASE + "_" + ap + EXT, img)

def wait_key(c):
  if type(c) == str:
    while cv2.waitKey(0) != ord(c):
      pass
    return None
  elif type(c) == list:
    while True:
      k = cv2.waitKey(0)
      v = chr(k)
      if v in c:
        return v
  return None

def view_image(title, img):
  cv2.imshow(title, img)
  print("Press 'q' to kill all windows.")
  wait_key('q')
  cv2.destroyAllWindows()

def add_image(title, img):
  cv2.imshow(title, img)

def flush_imview():
  print("Press 'q' to kill all windows.")
  wait_key('q')
  cv2.destroyAllWindows()

def clear_imview():
  cv2.destroyAllWindows()

def read_image(path):
  print("Reading image from " + path + "...")
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    sys.exit(2)
  if ask("View image?", ['y', 'n']) == 'y':
    view_image("Original", img)
  return img

def apply_gaussian(img):
  print("Applying gaussian blur...")
  s = open_ask("Gaussian blur kernel sizes (n, m) and (p, q)? Input format: n m p q", 4, int)
  bImg = cv2.GaussianBlur(img, (s[0], s[1]), 0)
  pImg = cv2.GaussianBlur(img, (s[2], s[3]), 0)
  if ask("View images?", ['y', 'n']) == 'y':
    add_image("Gaussian blur with kernel " + str(s[0:2]), bImg)
    add_image("Gaussian blur with kernel " + str(s[2:4]), pImg)
    flush_imview()
  return bImg, pImg

def apply_laplacian(img, gImg):
  print("Applying Laplacian...")
  iL = cv2.Laplacian(img, cv2.CV_64F)
  gL = cv2.Laplacian(gImg, cv2.CV_64F)
  if ask("View and compare Laplacians?", ['y', 'n']) == 'y':
    add_image("Laplacian applied over original", iL)
    add_image("Laplacian applied over gaussian", gL)
    flush_imview()
  return iL, gL

def apply_difference(gImg, hImg):
  print("Applying Difference of Gaussians...")
  D = hImg - gImg
  if ask("View difference of gaussians?", ['y', 'n']) == 'y':
    view_image("Difference of gaussians", D)
  return D

def compare_log_dog(lImg, logImg, dogImg):
  if ask("Compare Laplacian, LoG and DoG?", ['y', 'n']) == 'y':
    add_image("Laplacian", lImg)
    add_image("Laplacian of Gaussian", logImg)
    add_image("Difference of Gaussian", dogImg)
    flush_imview()

def create_raw_pyramid(img, n):
  print("Creating raw pyramid...")
  P = []
  L = img
  for i in range(n):
    d = tuple((np.asarray(L.shape)/2).astype(int))
    L = cv2.resize(L, d)
    P.append(L)
  return P

def create_gaussian_pyramid(img, n):
  print("Creating gaussian pyramid...")
  s = open_ask("Gaussian kernel size before subsampling? Input format: n m", 2, int)
  P = []
  L = img
  for i in range(n):
    G = cv2.GaussianBlur(L, tuple(s), 0)
    d = tuple((np.asarray(L.shape)/2).astype(int))
    L = cv2.resize(G, d)
    P.append(L)
  return P

def compare_pyramids(img):
  print("Comparing raw and gaussian pyramids...")
  n = open_ask("Pyramid height (number of levels)? Input format: m", 1, int)[0]
  rP, gP = create_raw_pyramid(img, n), create_gaussian_pyramid(img, n)
  i = 0
  print("Press j to go down a pyramid level.")
  print("Press i to go up a pyramid level.")
  print("Press q when you're done.")
  while True:
    add_image("Raw pyramid", rP[i])
    add_image("Gaussian pyramid", gP[i])
    c = wait_key(['j', 'k', 'q'])
    if c == 'j':
      i = -min(0, -i+1)
    elif c == 'k':
      i = min(n-1, i+1)
    else:
      clear_imview()
      break
  return rP, gP

def save_pyramid(P, ap):
  if WRITE_IMGS:
    for i, p in enumerate(P):
      write_image(ap + "_" + str(i), p)

def run():
  I = read_image(BASE + EXT)
  # write_image("gray", I)
  # G1, G2 = apply_gaussian(I)
  # write_image("G1", G1)
  # write_image("G2", G2)
  # iL, gL = apply_laplacian(I, G1)
  # write_image("L", iL)
  # write_image("LoG", gL)
  # D = apply_difference(G1, G2)
  # write_image("DoG", D)
  # compare_log_dog(iL, gL, D)
  rP, gP = compare_pyramids(I)
  save_pyramid(rP, "raw_pyramid")
  save_pyramid(gP, "gauss_pyramid")

if __name__ == "__main__":
  BASE, EXT = parse_args()
  if BASE is None:
    sys.exit(1)
  run()
