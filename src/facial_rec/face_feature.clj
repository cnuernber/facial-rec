(ns facial-rec.face-feature
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]
            [tech.v2.datatype :as dtype]))



(require-python '(mxnet mxnet.ndarray mxnet.module mxnet.io mxnet.model))
(require-python 'cv2)
(require-python '[numpy :as np])


(defn load-model
  [& {:keys [model-path checkpoint]
      :or {model-path "models/recognition/model"
           checkpoint 0}}]
  (let [[sym arg-params aux-params] (mxnet.model/load_checkpoint model-path checkpoint)
        all-layers (py/$a sym get_internals)
        target-layer (py/get-item all-layers "fc1_output")
        ;;TODO - I haven't overloaded enough of the IFn invoke methods for this to work without
        ;;using call-kw
        model (py/call-kw mxnet.module/Module [] {:symbol target-layer :context (mxnet/cpu) :label_names nil})]
    (py/$a model bind :data_shapes [["data" [1 3 112 112]]])
    (py/$a model set_params arg-params aux-params)
    model))

(defonce model (load-model))



(defn face->feature
  [img-path]
  (py/with-gil-stack-rc-context
    (if-let [new-img (cv2/imread img-path)]
      (let [new-img (cv2/cvtColor new-img cv2/COLOR_BGR2RGB)
            new-img (np/transpose new-img [2 0 1])
            input-blob (np/expand_dims new-img :axis 0)
            data (mxnet.ndarray/array input-blob)
            batch (mxnet.io/DataBatch :data [data])]
        (py/$a model forward batch :is_train false)
        (-> (py/$a model get_outputs)
            first
            (py/$a asnumpy)
            (py/as-tensor)
            (#(dtype/make-container :java-array :float32 %))))
      (throw (Exception. (format "Failed to load img: %s" img-path))))))
