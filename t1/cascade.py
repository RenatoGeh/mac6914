import numpy as np
import cv2
import sys

if len(sys.argv) != 2:
  print("Usage " + sys.argv[0] + " img")
  print("  img - image to detect eyes and faces")
  sys.exit(1)

def union(a, b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a, b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w < 0 or h < 0: return (0, 0, 0, 0)
  return (x, y, w, h)

def inter_ratio(A, B):
  dx = min(A[0]+A[2], B[0]+B[2])-max(A[0], B[0])
  dy = min(A[1]+A[3], B[1]+B[3])-max(A[1], B[1])
  sI = 0
  if dx >= 0 and dy >= 0:
    sI = dx*dy
  sU = A[2]*A[3]+B[2]*B[3]-sI
  print(A, B)
  print(sI, sU)
  return sI/sU
  # sI = max(0, min(A[2], B[2]) - max(A[0], B[0]))*max(0, min(A[3], B[3]) - max(A[1], B[1]))
  # sA = abs(A[0]-A[2])*abs(A[1]-A[3])
  # sB = abs(B[0]-B[2])*abs(B[1]-B[3])
  # sU = sA + sB - sI
  # I = intersection(A, B)
  # U = union(A, B)
  # sI = I[2]*I[3]
  # sU = U[2]*U[3]
  # print(sI, sU, sI/sU)
  # return sI/sU

def any_intersects(R, A):
  for r in R:
    i = inter_ratio(r, A)
    print(i)
    if i >= 0.1:
      return True
  return False

filename = sys.argv[1]

DATA_PATH = "/usr/share/opencv/haarcascades/"

face = [ \
        # cv2.CascadeClassifier(DATA_PATH+'haarcascade_frontalface_default.xml'),\
        # cv2.CascadeClassifier(DATA_PATH+'haarcascade_frontalface_alt.xml'),\
        cv2.CascadeClassifier(DATA_PATH+'haarcascade_frontalface_alt2.xml'),\
       ]
eye = cv2.CascadeClassifier(DATA_PATH+'haarcascade_eye.xml')

img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

F = []
for f in face:
  F.extend(f.detectMultiScale(gray, 1.3, 1))

R = []
for (x, y, w, h) in F:
  cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  R.append((x, y, w, h))
  rg = gray[y:y+h, x:x+w]
  rc = img[y:y+h, x:x+w]
  E = eye.detectMultiScale(rg)
  for (a, b, c, d) in E:
    cv2.rectangle(rc, (a, b), (a+c, b+d), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.imwrite("detect_"+filename, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
