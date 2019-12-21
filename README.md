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
