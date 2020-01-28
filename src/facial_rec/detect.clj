(ns facial-rec.detect
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :refer [py. py.- py..] :as py]
            [clojure.tools.logging :as log]))

(require-python '[distutils.core :refer [setup]])
(require-python '[numpy :as np])
(require-python 'pyximport)
(pyximport/install :setup_args {:include_dirs
                                (py/->py-list [(np/get_include)])})
(require-python '[retinaface :as rface])
(require-python 'cv2)
(require-python '[builtins :refer [slice]])
(require-python '[skimage.transform :as trans])


(defonce model (rface/RetinaFace "models/detection/R50" 0 -1))


(defn detect-faces
  [img-path]
  (py/with-gil-stack-rc-context
    (if-let [cv-img (cv2/imread img-path)]
      (when-let [detection (py. model detect cv-img 0.8)]
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


(def ideal-face-landmarks
  (np/array [[30.2946, 51.6963]
             [65.5318, 51.5014]
             [48.0252, 71.7366]
             [33.5493, 92.3655]
             [62.7299, 92.2041]]
            :dtype np/float32))


(defn affine-warp-mat
  [landmarks]
  (try
    (let [landmark-ary (np/array landmarks :dtype np/float32)
          sim-trans (trans/SimilarityTransform)
          success? (py. sim-trans estimate landmark-ary ideal-face-landmarks)]
      (when success?
        (-> (py.- sim-trans params)
            (py/get-item [(slice 0 2) (slice nil)]))))
    (catch Throwable e
      (log/warnf e (format "Similarity transform failed for landmarks: %s"
                           landmarks))
      nil)))


(defn crop-faces
  "Crop out faces.  For each face detection result, return a new image.  Returned images
  are scaled to a specific size (the size needed by the facial feature engine)."
  [img face-detection-result
   & {:keys [face-size align?]
      :or {face-size [112 112]
           align? true}}]
  ;;Alignment is also possible (we have landmarks) but not going to go there for
  ;;demo.  Note function is cv2.warpAffine.
  (->> face-detection-result
       (mapv (fn [{:keys [bbox landmarks]}]
               (let [{:keys [top-left bottom-right]} bbox
                     [min-x min-y] top-left
                     [max-x max-y] bottom-right
                     affine-mat (when align?
                                  (affine-warp-mat landmarks))]
                 (if affine-mat
                   (cv2/warpAffine img affine-mat face-size :borderValue 0.0)
                   ;;Fallthrough in the case the estimate mechanism fails or
                   ;;the user doesn't want alignment.
                   (->
                    (py/get-item img [(slice min-y max-y) (slice min-x max-x)])
                    (cv2/resize [112 112]))))))))
