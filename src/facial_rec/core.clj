(ns facial-rec.core
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]))


(require-python '[distutils.core :refer [setup]])
(require-python '[numpy :as np])
(require-python 'pyximport)
(pyximport/install :setup_args {:include_dirs
                                (py/->py-list [(np/get_include)])})
(require-python 'cv2)
(require-python '[rcnn.cython :as rc])
(require-python '[rcnn.cython.bbox :as rc])
(require-python '[retinaface :as rface :reload])
