import cv2
import os.path
import sys

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

def write_image(ap, img):
  if WRITE_IMGS:
    cv2.imwrite(BASE + "_" + ap + EXT, img)

def view_image(title, img):
  cv2.imshow(title, img)
  print("Press 'q' to kill all windows.")
  while cv2.waitKey(0) != ord('q'):
    pass
  cv2.destroyAllWindows()

def add_image(title, img):
  cv2.imshow(title, img)

def flush_imview():
  print("Press 'q' to kill all windows.")
  while cv2.waitKey(0) != ord('q'):
    pass
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
  while True:
    print("Gaussian blur kernel sizes (n, m) and (p, q)? Input format: n m p q")
    s = input().split()
    if len(s) == 4:
      break
  bImg = cv2.GaussianBlur(img, (int(s[0]), int(s[1])), 0)
  pImg = cv2.GaussianBlur(img, (int(s[2]), int(s[3])), 0)
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
  print("Applying Difference og Gaussians...")
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

def run():
  I = read_image(BASE + EXT)
  write_image("gray", I)
  G1, G2 = apply_gaussian(I)
  write_image("G1", G1)
  write_image("G2", G2)
  iL, gL = apply_laplacian(I, G1)
  write_image("L", iL)
  write_image("LoG", gL)
  D = apply_difference(G1, G2)
  write_image("DoG", D)
  compare_log_dog(iL, gL, D)

if __name__ == "__main__":
  BASE, EXT = parse_args()
  if BASE is None:
    sys.exit(1)
  run()
