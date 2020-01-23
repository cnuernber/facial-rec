(ns face-rec.demo
  (:require [facial-rec.detect :as detect]
            [facial-rec.face-feature :as face-feature]
            [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]
            [tech.io :as io]
            [tech.v2.datatype.functional :as dfn]
            [clojure.tools.logging :as log])
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
                    dest-fname (format "faces/%s.png" face-id)
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


(defn delete-previously-found-faces!
  []
  (->> (file-seq (io/file "faces"))
       (remove #(.isDirectory ^File %))
       (map (fn [f]
              (.delete f)))
       (dorun)))


(defn find-annotate-faces!
  []
  (log/info "finding faces")
  (delete-previously-found-faces!)
  (py/with-gil-stack-rc-context
    (->> (file-seq (io/file "dataset"))
         (remove #(.isDirectory ^File %))
         (mapcat (fn [^File src-img]
                   (filename->faces (.toString src-img))))
         vec
         (#(do (log/infof "Found %d faces" (count %))
               %)))))


(defn annotations
  []
  (->> (file-seq (io/file "faces"))
       (map #(.toString ^File %))
       (filter #(.endsWith ^String % "nippy"))
       (map (comp (juxt :id identity) io/get-nippy))
       (into {})))


(defn annotations-by-file
  []
  (group-by :src-file (vals (annotations))))


(defn nearest
  [ann-id]
  (let [{:keys [feature] :as target-annotation} (get (annotations) ann-id)]
    (->> (vals (annotations))
         (map #(assoc % :distance-squared (dfn/distance-squared feature (:feature %))))
         (sort-by :distance-squared)
         (map #(dissoc % :feature)))))


(defn- display-face-img
  [{:keys [id] :as entry}]
  (format "![face-img](faces/%s.png) " id))


(defn- display-distance-and-face-img
  [{:keys [id distance-squared] :as entry}]
  (format "%02d %s"
          (long (Math/sqrt (double distance-squared)))
          (display-face-img entry)))


(defn output-face-results!
  [& [all-faces]]
  (let [all-faces (or all-faces (find-annotate-faces!))]
    (spit "results.md"
          (with-out-str
            (println "## Results")
            (println "| face-img | 5 nearest |")
            (println "|-----|------|")
            (->> all-faces
                 (map (fn [{:keys [id] :as entry}]
                        (println "|" (display-face-img entry)
                                 "|" (->> (nearest id)
                                          (take 5)
                                          (map display-distance-and-face-img)
                                          (reduce str))
                                 "|")))
                 (dorun))))))


(comment
  ;;Stress testing the system
  (dotimes [iter 100]
    (println "running")
    (find-annotate-faces!))
  )
