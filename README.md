# Face-Reidentification
Face Reidentification using Facenet + Adding temporal consistency between frames in video

Modified code from:

https://github.com/timesler/facenet-pytorch

Additional code was added to improve accuracy on video inputs, using the fact that the bounding boxes between consecutive frames should be similar.



To run the code:

```
python infer.py
```

