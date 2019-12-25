# facial-rec

Demonstration of some pretty good facial rec tech.

![detection phase](detector_test.jpg)


## How It Works

At a high level, facial recognition consists of two steps: detection and embedding.

Detection takes a large image and produces a list of faces in the image.  This piece is
generally takes 


## Usage


This system is built to show a realistic example of a cutting-edge system.  As such
it rests on four components:
1.  docker
2.  Conda
3.  Python
4.  Clojure


The most advanced piece of the demo is actually the facial detection component.
Luckily, it was nicely wrapped.  To get it working we needed cython working and
there is some [good information](src/facial_rec/detect.clj) there if you want to 
use a system that is based partially on cython.


### Get the data

This script mainly downloads the models used for detection and feature embedding.

```console 
scripts/get-data
```


### Start up a REPL


```console
scripts/run-conda-docker
```

The port is printed out in a line like:

```console
nREPL server started on port 44507 on host localhost - nrepl://localhost:44507
```

Now in emacs, vim or somewhere connect to the
exposed port on locahost.


### Find/Annotate Faces


```clojure
(require '[facial-rec.demo :as demo])
;;long pause as things compile
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
