# Memory Box
Demo (click on GIF for the full video) :   
[![Memory Box working](https://media.giphy.com/media/3ov9jSrPznNGCmzaGA/giphy.gif)](https://www.youtube.com/watch?v=iU0qsqgCEkc)

# Inspiration
Wished for a dropbox which works with videos! Well wait no more!! Search moments that you relish, you want to share!

# What it does
The app looks through a library of videos, samples frames every few seconds, and generates natural language captions using a CNN+LSTM for each video frame, written in PyTorch. Next, all the captions are indexed using lucene and users can search this index using a web interface.

# How we built it
We will explain in the presentation

# Challenges we ran into
Setting up the infrastructure for video feature extraction. Setting up the indexing and searching aspects.

# Accomplishments that we're proud of

# What we learned
Bunch of technologies like Lucene, PyTorch, LSTM and CNN.

# What's next for MemoryBox
Scaling to 1000s of videos database
