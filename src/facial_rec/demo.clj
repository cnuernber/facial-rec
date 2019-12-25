(ns face-rec.demo
  (:require [facial-rec.detect :as detect]
            [facial-rec.face-feature :as face-feature]
            [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]
            [tech.io :as io]
            [tech.v2.datatype.functional :as dfn])
  (:import [java.io File]
           [java.util UUID]))



(io/make-parents "faces/face.jpg")
(require-python 'cv2)


(defn filename->faces
  [fname]
  (py/with-gil-stack-rc-context
    (let [detection (detect/detect-faces fname)
          cropped-faces (detect/crop-faces (cv2/imread fname) detection)]
      (mapv (fn [detection-result face-img]
              (let [face-id (UUID/randomUUID)
                    dest-fname (format "faces/%s.jpg" face-id)
                    dest-feature-fname (format "file://faces/%s.nippy" face-id)
                    _ (cv2/imwrite dest-fname face-img)
                    feature (face-feature/face->feature dest-fname)
                    metadata (merge detection-result
                                    {:id face-id
                                     :src-file fname
                                     :feature feature})]
                (io/put-nippy! dest-feature-fname metadata)
                metadata))
            detection cropped-faces))))


(defn find-annotate-faces!
  []
  (py/with-gil-stack-rc-context
    (->> (file-seq (io/file "dataset"))
         (remove #(.isDirectory ^File %))
         (mapcat (fn [^File src-img]
                   (filename->faces (.toString src-img))))
         vec)))


(def annotations
  (memoize
   (fn []
     (->> (file-seq (io/file "faces"))
          (map #(.toString ^File %))
          (filter #(.endsWith ^String % "nippy"))
          (map (comp (juxt :id identity) io/get-nippy))
          (into {})))))


(def annotations-by-file
  (memoize
   #(group-by :src-file (vals (annotations)))))


(defn nearest
  [ann-id]
  (let [{:keys [feature] :as target-annotation} (get (annotations) ann-id)]
    (->> (vals (annotations))
         (map #(assoc % :distance-squared (dfn/distance-squared feature (:feature %))))
         (sort-by :distance-squared)
         (map #(dissoc % :feature)))))
