(ns facial-rec.core
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]))


(require-python 'pyximport)
(pyximport/install )
(require-python 'cv2)
(require-python '[rcnn.cython :as rc])
(require-python '[rcnn.cython.bbox :as rc])
(require-python '[retinaface :as rface :reload])
