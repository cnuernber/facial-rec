# facial-rec

Demonstration of some pretty good facial rec tech.

![detection phase](detector_test.jpg)


## Usage

### Get the data

```console 
scripts/get-data
```


### Start up a REPL


```console
scripts/run-conda-docker
```


### Find/Annotate Faces


```clojure
(require '[facial-rec.demo :as demo])
(demo/find-annotate-faces!)
```

Now there are cutout faces in the faces subdir.
You can do nearest searches in the demo namespace and
see how well this network does.


I didn't do alignment of the faces, and it is unclear if the
feature network requires BGR or RGB images.  All told, I think
the results are middling.


## License

Copyright © 2019 Chris Nuernberger

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
http://www.eclipse.org/legal/epl-2.0.
