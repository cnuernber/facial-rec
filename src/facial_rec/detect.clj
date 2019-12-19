(ns facial-rec.detect
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]))

(require-python '[distutils.core :refer [setup]])
(require-python '[numpy :as np])
(require-python 'pyximport)
(pyximport/install :setup_args {:include_dirs
                                (py/->py-list [(np/get_include)])})
(require-python '[retinaface :as rface])
(require-python 'cv2)
(require-python '[builtins :refer [slice]])


(defonce model (rface/RetinaFace "models/detection/R50" 0 -1))


(defn detect-faces
  [img-path]
  (py/with-gil-stack-rc-context
    (if-let [cv-img (cv2/imread img-path)]
      (when-let [detection (py/$a model detect cv-img 0.8)]
        (let [[faces landmarks] detection]
          (->> (mapv (fn [face landmark]
                       (let [face-bbox (->> (take 4 face)
                                            (map (comp long #(Math/round (double %)))))
                             confidence (last face)]
                         {:confidence confidence
                          :bbox {:top-left (vec (take 2 face-bbox))
                                 :bottom-right (vec (drop 2 face-bbox))}
                          :landmarks (mapv #(mapv int %) landmark)}))
                     faces landmarks))))
      (throw (Exception. (format "Unable to open image %s" img-path))))))


(defn render-faces!
  "Draw the face detection result on an image.  Presumably the same image the data came from."
  [img face-detection-result & {:keys [bbox-color landmark-color]
                                :or {bbox-color [0 255 0]
                                     landmark-color [255 0 0]}}]
  (doseq [{:keys [bbox landmarks]} face-detection-result]
    (cv2/rectangle img (:top-left bbox) (:bottom-right bbox) bbox-color 2)
    (doseq [landmark landmarks]
      (cv2/circle img landmark 1 landmark-color 2)))
  img)


(defn crop-faces
  "Crop out faces.  For each face detection result, return a new image.
  Returned images are scaled to a specific size (the size needed by the
  facial feature engine)."
  [img face-detection-result
   & {:keys [face-size]
      :or {face-size [112 112]}}]
  ;;Alignment is also possible (we have landmarks) but not going to go there for
  ;;demo.  Note function is cv2.warpAffine.
  (->> face-detection-result
       (mapv (fn [{:keys [bbox]}]
               (let [{:keys [top-left bottom-right]} bbox
                     [min-x min-y] top-left
                     [max-x max-y] bottom-right]
                 (->
                  (py/get-item img [(slice min-y max-y) (slice min-x max-x)])
                  (cv2/resize [112 112])))))))
