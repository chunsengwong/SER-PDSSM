# SER-PDSSM

MLP_ExtraTree.py: 
SER model that is trained by MLP with RFE(ExtraTreeClassifier) technique
x= emotional speech features (MFCCs, Chroma, Mel spectrogram)
y= targeted emotions (big five: sad, happy, angry, disgusted, fearful)

SER-PDSSM:
SER model that is integrated with PDSSM (puntuation-driven speech segmentation model) to predict emotions from long-duration speech.
the main tool of PDSSM is Google spech-to-text library, PDSSM treats punctuation marks ('.','?','!') as indicators. 

